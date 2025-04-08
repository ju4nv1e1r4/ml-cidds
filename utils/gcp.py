from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CloudStorageOps:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def load_from_bucket(self, file_path: str):
        """
        Downloads a file from the bucket and returns the bytes
        Gets the name of the file to be downloaded from the bucket in Google Cloud Storage.
        """
        blob = self.bucket.blob(file_path)
        
        try:
            data = blob.download_as_bytes()
            logging("Data is loaded successfully")
            return data
        except Exception as e:
            logging.error(f"Error downloading file: {e}")
            return None

    def list_from_bucket(self):
        """List all files in the bucket"""
        blobs = self.storage_client.list_blobs(self.bucket_name)
        for blob in blobs:
            print(blob.name)

    def delete_from_bucket(self, file_path):
        """
        Delete a specific file from the bucket
        Gets the name of the file to be deleted from the bucket in Google Cloud Storage.
        """
        my_bucket = self.storage_client.bucket(self.bucket_name)
        blob = my_bucket.blob(file_path)
        generation_match_precondition = None

        blob.reload()
        generation_match_precondition = blob.generation
        blob.delete(if_generation_match=generation_match_precondition)

        return logging.info(f"File deleted: '{file_path}'.")
    
    def upload_to_bucket(self, source_file_name, destination_file_name):
        """
        Upload a file to a Google Cloud Storage bucket
        This function must have:
        - Name of the source file (local)
        - Name of the file in the destination (GCS Bucket)
        """
        my_bucket = self.storage_client.bucket(self.bucket_name)
        blob = my_bucket.blob(destination_file_name)

        blob.upload_from_filename(source_file_name)

        return logging.info(
            f"File {source_file_name} uploaded to {destination_file_name}."
        )
