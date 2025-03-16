"""Azure blob storage stubs."""

from typing import Any, Optional, BinaryIO


class BlobServiceClient:
    def __init__(self, account_url: str, credential: Any = None) -> None: ...

    @classmethod
    def from_connection_string(
        cls, conn_str: str, credential: Any = None) -> "BlobServiceClient": ...

    def get_container_client(self, container: str) -> "ContainerClient": ...


class ContainerClient:
    def __init__(self, account_url: str, container_name: str,
                 credential: Any = None) -> None: ...

    def get_blob_client(self, blob: str) -> "BlobClient": ...


class BlobClient:
    def __init__(self, account_url: str, container_name: str,
                 blob_name: str, credential: Any = None) -> None: ...

    @property
    def url(self) -> str: ...
    def exists(self) -> bool: ...
    def upload_blob(self, data: Any, **kwargs: Any) -> Any: ...
    def download_blob(self, **kwargs: Any) -> "StorageStreamDownloader": ...


class StorageStreamDownloader:
    def __init__(self) -> None: ...
    def readinto(self, buffer: BinaryIO) -> int: ...


class ContentSettings:
    def __init__(
        self,
        content_type: Optional[str] = None,
        content_encoding: Optional[str] = None,
        content_language: Optional[str] = None,
        content_disposition: Optional[str] = None,
        cache_control: Optional[str] = None,
        content_md5: Optional[bytes] = None,
    ) -> None: ...
