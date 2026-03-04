from google.cloud import storage
from google.oauth2 import service_account


def upload_to_gcs(
    local_file_path: str,
    bucket_name: str,
    destination_blob_name: str,
    credentials_file_path: str,
) -> None:
    """
    Upload local file to google cloud storage bucket.

    Can be used to upload hdf5 file at the end of each run. Requires a service account
    key stored in a local JSON file.
    """
    try:
        # Authenticate using the JSON key file.
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file_path
        )
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)

        # Create a "blob" (the GCS equivalent of an S3 Object).
        blob = bucket.blob(destination_blob_name)

        print(f"Uploading: {local_file_path}")
        print(f"       to: gs://{bucket_name}/{destination_blob_name}")
        blob.upload_from_filename(local_file_path)

    except Exception as e:
        print(f"Error while uploading to google cloud storage: {e}")
