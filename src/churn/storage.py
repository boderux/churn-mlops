"""MinIO / S3 Object Storage integration for data, reports, and artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from churn.config import ARTIFACT_DIR, CHAMPION_DIR, resolve

log = logging.getLogger(__name__)

# Bucket names (configurable via environment variables)
BUCKET_DATA = os.getenv("MINIO_DATA_BUCKET", "churn-data")
BUCKET_REPORTS = os.getenv("MINIO_REPORTS_BUCKET", "churn-reports")
BUCKET_ARTIFACTS = os.getenv("MINIO_ARTIFACTS_BUCKET", "churn-artifacts")
BUCKET_MLFLOW = os.getenv("MINIO_MLFLOW_BUCKET", "mlflow")


def get_endpoint_url() -> str:
    """Resolve MinIO / S3 endpoint URL."""
    return (
        os.getenv("MINIO_ENDPOINT_URL")
        or os.getenv("MLFLOW_S3_ENDPOINT_URL")
        or os.getenv("S3_ENDPOINT_URL")
        or "http://localhost:9000"
    )


def get_s3_client():
    """Create a boto3 S3 client configured for MinIO."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log.warning("boto3 is not installed; MinIO storage client unavailable.")
        return None

    endpoint = get_endpoint_url()
    access_key = os.getenv("AWS_ACCESS_KEY_ID", os.getenv("MINIO_ROOT_USER", "minioadmin"))
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"))
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def is_minio_available() -> bool:
    """Check if MinIO is reachable."""
    client = get_s3_client()
    if client is None:
        return False
    try:
        client.list_buckets()
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("MinIO is not reachable at %s: %s", get_endpoint_url(), exc)
        return False


def ensure_bucket(bucket: str) -> bool:
    """Ensure a bucket exists in MinIO; create it if missing."""
    client = get_s3_client()
    if client is None:
        return False
    try:
        existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if bucket not in existing:
            client.create_bucket(Bucket=bucket)
            log.info("Created MinIO bucket: %s", bucket)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to check/create bucket %s: %s", bucket, exc)
        return False


def upload_file(local_path: Path | str, bucket: str, s3_key: str | None = None) -> str | None:
    """Upload a single file to MinIO. Returns s3://<bucket>/<key> on success."""
    p = Path(local_path)
    if not p.is_file():
        log.warning("File not found for upload: %s", p)
        return None

    client = get_s3_client()
    if client is None:
        return None

    ensure_bucket(bucket)
    key = s3_key or p.name
    try:
        client.upload_file(str(p), bucket, key)
        s3_uri = f"s3://{bucket}/{key}"
        log.info("Uploaded %s -> %s", p, s3_uri)
        return s3_uri
    except Exception as exc:  # noqa: BLE001
        log.warning("Upload failed for %s to %s/%s: %s", p, bucket, key, exc)
        return None


def download_file(bucket: str, s3_key: str, local_path: Path | str) -> Path | None:
    """Download a file from MinIO to local destination."""
    out = Path(local_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()
    if client is None:
        return None

    try:
        client.download_file(bucket, s3_key, str(out))
        log.info("Downloaded s3://%s/%s -> %s", bucket, s3_key, out)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("Download failed for s3://%s/%s: %s", bucket, s3_key, exc)
        return None


def upload_dir(local_dir: Path | str, bucket: str, prefix: str = "") -> list[str]:
    """Recursively upload a local directory to a MinIO bucket."""
    d = Path(local_dir)
    if not d.is_dir():
        log.debug("Directory does not exist for upload: %s", d)
        return []

    client = get_s3_client()
    if client is None:
        return []

    ensure_bucket(bucket)
    uploaded: list[str] = []
    clean_prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""

    for item in d.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(d).as_posix()
            key = f"{clean_prefix}{rel_path}"
            try:
                client.upload_file(str(item), bucket, key)
                uploaded.append(f"s3://{bucket}/{key}")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed uploading %s to %s/%s: %s", item, bucket, key, exc)

    if uploaded:
        log.info("Uploaded %d files from %s to s3://%s/%s", len(uploaded), d, bucket, clean_prefix)
    return uploaded


def sync_data_to_minio(params: dict[str, Any] | None = None) -> list[str]:
    """Sync raw and processed datasets to MinIO bucket `churn-data`."""
    if not is_minio_available():
        log.info("MinIO unavailable, skipping data sync.")
        return []

    uploaded = []
    # Sync raw data
    raw_path = resolve("data/raw")
    if raw_path.exists():
        uploaded.extend(upload_dir(raw_path, BUCKET_DATA, prefix="raw"))

    # Sync processed data
    proc_path = resolve("data/processed")
    if proc_path.exists():
        uploaded.extend(upload_dir(proc_path, BUCKET_DATA, prefix="processed"))

    log.info("Synced data to MinIO: %d files", len(uploaded))
    return uploaded


def sync_reports_to_minio() -> list[str]:
    """Sync all reports (Evidently, Fairness, Explainability, PNGs) to MinIO `churn-reports`."""
    if not is_minio_available():
        log.info("MinIO unavailable, skipping reports sync.")
        return []

    reports_dir = ARTIFACT_DIR / "reports"
    uploaded = upload_dir(reports_dir, BUCKET_REPORTS, prefix="reports")

    # Also upload root gate decision / candidate metrics if present
    for json_name in ["gate_decision.json", "candidate_metrics.json"]:
        file_path = ARTIFACT_DIR / json_name
        if file_path.exists():
            uri = upload_file(file_path, BUCKET_REPORTS, s3_key=f"reports/{json_name}")
            if uri:
                uploaded.append(uri)

    log.info("Synced reports to MinIO: %d files", len(uploaded))
    return uploaded


def sync_artifacts_to_minio() -> list[str]:
    """Sync champion model bundle and gate metrics to MinIO `churn-artifacts`."""
    if not is_minio_available():
        log.info("MinIO unavailable, skipping artifacts sync.")
        return []

    uploaded = []
    if CHAMPION_DIR.exists():
        uploaded.extend(upload_dir(CHAMPION_DIR, BUCKET_ARTIFACTS, prefix="champion"))

    for item in ["gate_decision.json", "candidate_metrics.json"]:
        p = ARTIFACT_DIR / item
        if p.exists():
            uri = upload_file(p, BUCKET_ARTIFACTS, s3_key=item)
            if uri:
                uploaded.append(uri)

    log.info("Synced model artifacts to MinIO: %d files", len(uploaded))
    return uploaded


def sync_all_to_minio(params: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Sync datasets, reports, and artifacts to MinIO."""
    return {
        "data": sync_data_to_minio(params),
        "reports": sync_reports_to_minio(),
        "artifacts": sync_artifacts_to_minio(),
    }
