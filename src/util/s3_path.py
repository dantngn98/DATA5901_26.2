# standard
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import urlparse
from typing import Self


@dataclass(frozen=True, slots=True)
class S3Path:
    bucket: str
    key: PurePosixPath = field(default_factory=PurePosixPath)
    uri: str = field(init=False)  # calculated from bucket and key

    def __post_init__(self):
        key_str = self.key.as_posix()
        uri = (
            f"s3://{self.bucket}" if key_str in ("", ".")
            else f"s3://{self.bucket}/{key_str}"
        )
        object.__setattr__(self, "uri", uri)

    @classmethod
    def from_uri(cls, uri: str) -> Self:
        # URI format: s3://{bucket}/{key}
        # Example: s3://my-bucket/path/to/file.txt

        parsed = urlparse(uri)

        if parsed.scheme != "s3":
            raise ValueError(f"expected s3:// URI but got: '{uri}'")
        if not parsed.netloc:
            raise ValueError(f"missing bucket (perhaps you forgot double '/' after s3): {uri}")

        return cls(
            bucket=parsed.netloc,
            key=PurePosixPath(parsed.path.lstrip("/")),
        )

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def suffix(self) -> str:
        return self.key.suffix

    @property
    def stem(self) -> str:
        return self.key.stem

    @property
    def parent(self) -> Self:
        # not robust to .. tricks but sufficient for internal utility

        if self.key in (PurePosixPath(), PurePosixPath(".")):
            return self

        return S3Path(
            bucket=self.bucket,
            key=self.key.parent,
        )

    def joinpath(self, *parts: str) -> Self:
        return S3Path(
            bucket=self.bucket,
            key=self.key.joinpath(*parts),
        )

    def __truediv__(self, other: str) -> Self:
        if not isinstance(other, str):
            return NotImplemented

        return self.joinpath(other)
    
    def __fspath__(self):
        return self.uri
