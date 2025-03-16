"""Service for handling file storage in Azure."""

import os
from io import BytesIO
from typing import Any, Optional

from app.config import settings
from fastapi import UploadFile


class StorageService:
    """Service for handling file storage."""

    def __init__(self) -> None:
        """Initialize storage service."""
        self.use_azure = getattr(settings, "USE_AZURE_STORAGE", False)
        self.container_name = "puzzle-images"

        if self.use_azure:
            try:
                from azure.storage.blob import BlobServiceClient

                self.blob_service_client = BlobServiceClient.from_connection_string(
                    settings.AZURE_STORAGE_CONNECTION_STRING
                )
                self.container_client = self.blob_service_client.get_container_client(
                    self.container_name
                )
            except (ImportError, AttributeError) as e:
                print(f"Azure Storage not configured properly: {e}")
                self.use_azure = False

    async def save_file(self, file: UploadFile, file_name: str) -> str:
        """Save file to storage and return file path."""
        if self.use_azure:
            try:
                from azure.storage.blob import ContentSettings

                blob_client = self.container_client.get_blob_client(file_name)
                file_contents = await file.read()

                content_settings = ContentSettings(
                    content_type=file.content_type)
                blob_client.upload_blob(
                    file_contents,
                    overwrite=True,
                    content_settings=content_settings
                )
                return blob_client.url
            except Exception as e:
                print(f"Error saving file to Azure: {e}")
                # Fall back to local storage
                await file.seek(0)  # Reset file position after reading

        # Local file storage
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    async def get_file(self, file_name: str) -> Optional[BytesIO]:
        """Get file from storage."""
        if self.use_azure:
            try:
                blob_client = self.container_client.get_blob_client(file_name)
                if not blob_client.exists():
                    return None

                stream = BytesIO()
                blob_data = blob_client.download_blob()
                blob_data.readinto(stream)
                stream.seek(0)
                return stream
            except Exception as e:
                print(f"Error getting file from Azure: {e}")
                # Fall back to local if Azure fails

        # Local file storage
        file_path = os.path.join(settings.UPLOAD_DIR, file_name)
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            stream = BytesIO(f.read())
        return stream
