from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EXPECTED_MIND_FILES = {
    "train_news": Path("MINDsmall_train/news.tsv"),
    "train_behaviors": Path("MINDsmall_train/behaviors.tsv"),
    "dev_news": Path("MINDsmall_dev/news.tsv"),
    "dev_behaviors": Path("MINDsmall_dev/behaviors.tsv"),
}

DataProtocol = Literal["official_train_dev", "train_only_chronological"]
DEFAULT_DATA_PROTOCOL: DataProtocol = "official_train_dev"
DATA_PROTOCOLS: tuple[DataProtocol, ...] = (
    "official_train_dev",
    "train_only_chronological",
)
PROTOCOL_REQUIRED_FILE_KEYS: dict[DataProtocol, tuple[str, ...]] = {
    "official_train_dev": (
        "train_news",
        "train_behaviors",
        "dev_news",
        "dev_behaviors",
    ),
    "train_only_chronological": (
        "train_news",
        "train_behaviors",
    ),
}


@dataclass(frozen=True)
class LayoutValidation:
    data_dir: Path
    protocol: DataProtocol
    existing_files: dict[str, Path]
    missing_files: dict[str, Path]

    @property
    def is_valid(self) -> bool:
        return not self.missing_files


def validate_mind_layout(
    data_dir: Path,
    protocol: DataProtocol = DEFAULT_DATA_PROTOCOL,
) -> LayoutValidation:
    data_dir = data_dir.expanduser()
    existing_files: dict[str, Path] = {}
    missing_files: dict[str, Path] = {}

    for name in required_file_keys(protocol):
        relative_path = EXPECTED_MIND_FILES[name]
        full_path = data_dir / relative_path
        if full_path.exists():
            existing_files[name] = full_path
        else:
            missing_files[name] = full_path

    return LayoutValidation(
        data_dir=data_dir,
        protocol=protocol,
        existing_files=existing_files,
        missing_files=missing_files,
    )


def format_layout_validation(result: LayoutValidation) -> str:
    lines = [
        f"MIND-small layout validation for {result.data_dir}",
        f"Protocol: {result.protocol}",
    ]
    for name in required_file_keys(result.protocol):
        relative_path = EXPECTED_MIND_FILES[name]
        full_path = result.data_dir / relative_path
        status = "OK" if name in result.existing_files else "MISSING"
        lines.append(f"{status}: {full_path}")

    if result.is_valid:
        lines.append("Layout valid: all files required by the selected protocol are present.")
    else:
        lines.append(
            "Layout invalid: place the files required by the selected protocol under "
            "the paths above."
        )

    return "\n".join(lines)


def require_valid_mind_layout(
    data_dir: Path,
    protocol: DataProtocol = DEFAULT_DATA_PROTOCOL,
) -> LayoutValidation:
    result = validate_mind_layout(data_dir, protocol)
    if not result.is_valid:
        missing = "\n".join(f"- {path}" for path in result.missing_files.values())
        raise FileNotFoundError(
            f"Missing MIND-small source files required by protocol {protocol!r}:\n{missing}"
        )
    return result


def required_file_keys(protocol: DataProtocol) -> tuple[str, ...]:
    try:
        return PROTOCOL_REQUIRED_FILE_KEYS[protocol]
    except KeyError as exc:
        choices = ", ".join(DATA_PROTOCOLS)
        raise ValueError(f"Unknown data protocol {protocol!r}; expected one of: {choices}") from exc
