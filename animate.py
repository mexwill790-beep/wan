import os
import io
import json
import time
import re
import tempfile
from datetime import datetime, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from gradio_client import Client

SPACE_ID = "Wan-AI/Wan2.2-Animate"

# Fixed settings you requested
WAN_MODE = "wan2.2-animate-mix"   # MIX
WAN_QUALITY = "wan-pro"          # PRO (720p)

PROCESSED_TAG = "__PROCESSED"

# Hugging Face Space constraints (from the Space UI text) are best-effort enforced there;
# we still do light checks.
MAX_VIDEO_MB = 200


def env_required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def build_drive():
    sa_json = env_required("GDRIVE_SA_JSON")
    info = json.loads(sa_json)
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(drive, folder_id: str, mime_prefix: str | None = None):
    # Query: files in folder, not trashed
    q = f"'{folder_id}' in parents and trashed = false"
    if mime_prefix:
        q += f" and mimeType contains '{mime_prefix}'"

    files = []
    page_token = None
    while True:
        resp = drive.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, size)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def pick_latest_image(files):
    # Choose most recently modified/created image file
    def parse_dt(f):
        # prefer modifiedTime if present, else createdTime
        t = f.get("modifiedTime") or f.get("createdTime")
        return datetime.fromisoformat(t.replace("Z", "+00:00"))

    image_candidates = [
        f for f in files
        if (f.get("mimeType", "").startswith("image/"))
           and not f.get("name", "").startswith(".")
    ]
    if not image_candidates:
        return None
    image_candidates.sort(key=parse_dt, reverse=True)
    return image_candidates[0]


def is_processed(name: str) -> bool:
    return PROCESSED_TAG in name


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def download_file(drive, file_id: str, out_path: str):
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.FileIO(out_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()


def upload_file_to_folder(drive, local_path: str, folder_id: str, out_name: str):
    file_metadata = {"name": out_name, "parents": [folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    created = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name",
        supportsAllDrives=True
    ).execute()
    return created["id"]


def rename_file(drive, file_id: str, new_name: str):
    drive.files().update(
        fileId=file_id,
        body={"name": new_name},
        fields="id, name",
        supportsAllDrives=True
    ).execute()


def find_best_api_name(client: Client) -> str | None:
    """
    Spaces can change API endpoint names.
    We inspect available endpoints and pick something that looks like the main generator.
    """
    try:
        api = client.view_api(all_endpoints=True)
    except Exception:
        # fallback: many spaces use "/predict"
        return "/predict"

    endpoints = api.get("named_endpoints", {}) if isinstance(api, dict) else {}
    if not endpoints:
        return "/predict"

    # Heuristic: prefer endpoints with 4 inputs (image, video, model_id, model)
    best = None
    for api_name, meta in endpoints.items():
        inputs = meta.get("parameters", [])
        if len(inputs) == 4:
            best = api_name
            break

    # If not found, just take the first endpoint
    if not best:
        best = next(iter(endpoints.keys()))
    return best


def call_wan2_2(client: Client, api_name: str, image_path: str, video_path: str, retries: int = 5):
    """
    Calls the HF Space through gradio_client. Retries on queue/temporary errors.
    Returns local path to output file.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            result = client.predict(
                image_path,
                video_path,
                WAN_MODE,
                WAN_QUALITY,
                api_name=api_name
            )
            # Result format can vary: string path, dict, list/tuple
            # We normalize to a local filepath.
            if isinstance(result, str) and os.path.exists(result):
                return result

            if isinstance(result, (list, tuple)):
                for item in result:
                    if isinstance(item, str) and os.path.exists(item):
                        return item

            if isinstance(result, dict):
                # sometimes {"video": "..."} or {"path": "..."}
                for k in ["video", "path", "output", "file"]:
                    v = result.get(k)
                    if isinstance(v, str) and os.path.exists(v):
                        return v

            # If it's a URL, we cannot reliably download without extra handling.
            raise RuntimeError(f"Unexpected HF output type: {type(result)} / {result}")

        except Exception as e:
            last_err = e
            sleep_s = min(60, 5 * attempt)
            print(f"[HF] Attempt {attempt}/{retries} failed: {e} | sleeping {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError(f"HF call failed after {retries} retries: {last_err}")


def main():
    pic_folder = env_required("PIC_FOLDER_ID")
    ref_folder = env_required("REF_FOLDER_ID")
    out_folder = env_required("OUT_FOLDER_ID")
    hf_token = os.environ.get("HF_TOKEN")

    drive = build_drive()

    # 1) Find latest character image
    pic_files = list_files_in_folder(drive, pic_folder, mime_prefix="image/")
    latest_img = pick_latest_image(pic_files)
    if not latest_img:
        raise RuntimeError("No image found in PIC_FOLDER_ID folder.")

    # 2) List ref videos (skip processed)
    ref_files = list_files_in_folder(drive, ref_folder, mime_prefix="video/")
    todo = [f for f in ref_files if not is_processed(f.get("name", ""))]
    todo.sort(key=lambda f: (f.get("modifiedTime") or f.get("createdTime") or ""))

    if not todo:
        print("No unprocessed videos found. Nothing to do.")
        return

    # 3) Connect to HF Space
    if hf_token:
        client = Client(SPACE_ID, hf_token=hf_token)
    else:
        client = Client(SPACE_ID)
    api_name = find_best_api_name(client)
    print(f"Using HF Space endpoint: {api_name}")

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, safe_filename(latest_img["name"]))
        print(f"Downloading character image: {latest_img['name']}")
        download_file(drive, latest_img["id"], img_path)

        for f in todo:
            name = f["name"]
            size = int(f.get("size") or 0)
            size_mb = size / (1024 * 1024) if size else 0

            if size_mb and size_mb > MAX_VIDEO_MB:
                print(f"Skipping {name} (too large: {size_mb:.1f} MB)")
                continue

            print(f"\nProcessing: {name} ({size_mb:.1f} MB)")
            video_path = os.path.join(tmp, safe_filename(name))
            download_file(drive, f["id"], video_path)

            # Call HF
            out_local = call_wan2_2(client, api_name, img_path, video_path)

            # Upload output
            base = os.path.splitext(name)[0]
            out_name = f"{base}__MIX__PRO.mp4"
            out_id = upload_file_to_folder(drive, out_local, out_folder, out_name)
            print(f"Uploaded output: {out_name} (id={out_id})")

            # Rename input as processed
            new_in_name = f"{base}{PROCESSED_TAG}.mp4"
            rename_file(drive, f["id"], new_in_name)
            print(f"Renamed input to: {new_in_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
