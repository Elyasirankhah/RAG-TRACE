import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config

os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

BUCKET = "physionet-open"
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "download_ptbxl.log")
DATASETS = (
    ("ptb-xl/1.0.3/", os.path.join(ROOT, "ptb-xl")),
    ("ptb-xl-plus/1.0.1/", os.path.join(ROOT, "ptb-xl-plus")),
)
WORKERS = 32


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def client():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=20,
            read_timeout=60,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=WORKERS + 4,
        ),
    )


def list_objects(s3, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            objects.append((key, int(item["Size"])))
    return objects


def fetch(s3, key: str, size: int, prefix: str, dest_root: str) -> tuple[str, int]:
    rel = key[len(prefix) :].replace("/", os.sep)
    dest = os.path.join(dest_root, rel)
    if os.path.isfile(dest) and os.path.getsize(dest) == size:
        return "skip", size
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temporary = dest + ".part"
    for attempt in range(1, 5):
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            if len(body) != size:
                raise IOError(f"size {len(body)} != {size}")
            with open(temporary, "wb") as handle:
                handle.write(body)
            os.replace(temporary, dest)
            return "ok", size
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            if attempt == 4:
                raise
            time.sleep(attempt)
    return "ok", size


def download_prefix(s3, prefix: str, dest_root: str) -> None:
    objects = list_objects(s3, prefix)
    total = sum(size for _, size in objects)
    log(f"{prefix} {len(objects)} files, {total / 1024**3:.2f} GB")
    done = skipped = 0
    done_bytes = 0
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch, s3, key, size, prefix, dest_root): key
            for key, size in objects
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                status, size = future.result()
            except Exception as exc:
                failures.append(f"{key}: {exc}")
                log(f"FAIL {key}: {exc}")
                continue
            if status == "skip":
                skipped += 1
            else:
                done += 1
            done_bytes += size
            finished = done + skipped + len(failures)
            if finished % 2000 == 0 or finished == len(objects):
                log(
                    f"{prefix} {finished}/{len(objects)} "
                    f"{done_bytes / 1024**3:.2f}/{total / 1024**3:.2f} GB"
                )
    if failures:
        raise RuntimeError(f"{len(failures)} files failed in {prefix}")
    log(f"{prefix} complete downloaded={done} skipped={skipped}")


def main() -> int:
    s3 = client()
    for prefix, dest in DATASETS:
        os.makedirs(dest, exist_ok=True)
        download_prefix(s3, prefix, dest)
    log("complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
