from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from churn import storage


def test_endpoint_resolution():
    with patch.dict("os.environ", {"MINIO_ENDPOINT_URL": "http://minio:9000"}):
        assert storage.get_endpoint_url() == "http://minio:9000"


def test_is_minio_available_false_when_offline():
    with patch("churn.storage.get_s3_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = Exception("Connection refused")
        mock_get_client.return_value = mock_client
        assert storage.is_minio_available() is False


def test_is_minio_available_true_when_online():
    with patch("churn.storage.get_s3_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {"Buckets": [{"Name": "mlflow"}]}
        mock_get_client.return_value = mock_client
        assert storage.is_minio_available() is True


def test_ensure_bucket():
    with patch("churn.storage.get_s3_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {"Buckets": []}
        mock_get_client.return_value = mock_client

        assert storage.ensure_bucket("churn-data") is True
        mock_client.create_bucket.assert_called_once_with(Bucket="churn-data")


def test_upload_and_download_file(tmp_path: Path):
    src = tmp_path / "sample.txt"
    src.write_text("churn mlops")
    dest = tmp_path / "downloaded.txt"

    with patch("churn.storage.get_s3_client") as mock_get_client, \
         patch("churn.storage.ensure_bucket", return_value=True):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        uri = storage.upload_file(src, "churn-reports", "sample.txt")
        assert uri == "s3://churn-reports/sample.txt"
        mock_client.upload_file.assert_called_once_with(str(src), "churn-reports", "sample.txt")

        res = storage.download_file("churn-reports", "sample.txt", dest)
        assert res == dest
        mock_client.download_file.assert_called_once_with("churn-reports", "sample.txt", str(dest))


def test_sync_functions_safe_when_minio_offline():
    with patch("churn.storage.is_minio_available", return_value=False):
        assert storage.sync_data_to_minio() == []
        assert storage.sync_reports_to_minio() == []
        assert storage.sync_artifacts_to_minio() == []
        all_res = storage.sync_all_to_minio()
        assert all_res["data"] == []
        assert all_res["reports"] == []
        assert all_res["artifacts"] == []
