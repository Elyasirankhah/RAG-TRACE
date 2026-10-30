import json
import os
import sys
import time
import urllib.error
import urllib.request
\
DATASET = "ds004504"
TAG = "1.0.9"
ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ROOT, DATASET)
LOG = os.path.join(ROOT, "download_ds004504.log")
QUERY = """
query($datasetId: ID!, $tag: String!) {
  snapshot(datasetId: $datasetId, tag: $tag) {
    id
    files(recursive: true) { filename size directory urls }
  }
}
"""
\
\
def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
def file_list():
    body = json.dumps({"query": QUERY, "variables": {"datasetId": DATASET, "tag": TAG}}).encode()
    request = urllib.request.Request(
        "https://openneuro.org/crn/graphql",
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    snapshot = payload["data"]["snapshot"]
    files = [item for item in snapshot["files"] if not item.get("directory")]
    return snapshot["id"], files
def download(url: str, dest: str, size: int) -> None:
    temporary = dest + ".part"
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            got = os.path.getsize(temporary)
            if got != size:
                raise IOError(f"size {got} != {size}")
            os.replace(temporary, dest)
            return
        except Exception as exc:
            if os.path.exists(temporary):
                os.remove(temporary)
            if attempt == 5:
                raise
            log(f"retry {attempt} {os.path.basename(dest)}: {exc}")
            time.sleep(2 * attempt)
def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    snapshot_id, files = file_list()
    total = sum(item["size"] for item in files)
    log(f"{snapshot_id}: {len(files)} files, {total / 1024**3:.2f} GB")
    done = skipped = 0
    done_bytes = 0
    for index, item in enumerate(files, start=1):
        rel = item["filename"].replace("/", os.sep)
        dest = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isfile(dest) and os.path.getsize(dest) == item["size"]:
            skipped += 1
            done_bytes += item["size"]
            continue
        log(f"[{index}/{len(files)}] {item['filename']} ({item['size'] / 1024**2:.1f} MB)")
        download(item["urls"][0], dest, item["size"])
        done += 1
        done_bytes += item["size"]
        if done % 10 == 0 or item["size"] > 5 * 1024 * 1024:
            log(f"progress {done_bytes / 1024**3:.2f} / {total / 1024**3:.2f} GB")
    log(f"complete downloaded={done} skipped={skipped} files={len(files)}")
    return 0
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
