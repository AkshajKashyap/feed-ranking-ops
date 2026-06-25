from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED_MIND_FILES = {
    "train_news": Path("MINDsmall_train/news.tsv"),
    "train_behaviors": Path("MINDsmall_train/behaviors.tsv"),
    "dev_news": Path("MINDsmall_dev/news.tsv"),
    "dev_behaviors": Path("MINDsmall_dev/behaviors.tsv"),
}


@dataclass(frozen=True)
class LayoutValidation:
    data_dir: Path
    existing_files: dict[str, Path]
    missing_files: dict[str, Path]

    @property
    def is_valid(self) -> bool:
        return not self.missing_files


def validate_mind_layout(data_dir: Path) -> LayoutValidation:
    data_dir = data_dir.expanduser()
    existing_files: dict[str, Path] = {}
    missing_files: dict[str, Path] = {}

    for name, relative_path in EXPECTED_MIND_FILES.items():
        full_path = data_dir / relative_path
        if full_path.exists():
            existing_files[name] = full_path
        else:
            missing_files[name] = full_path

    return LayoutValidation(
        data_dir=data_dir,
        existing_files=existing_files,
        missing_files=missing_files,
    )


def format_layout_validation(result: LayoutValidation) -> str:
    lines = [f"MIND-small layout validation for {result.data_dir}"]
    for name, relative_path in EXPECTED_MIND_FILES.items():
        full_path = result.data_dir / relative_path
        status = "OK" if name in result.existing_files else "MISSING"
        lines.append(f"{status}: {full_path}")

    if result.is_valid:
        lines.append("Layout valid: all expected MIND-small files are present.")
    else:
        lines.append("Layout invalid: place the missing files under the paths above.")

    return "\n".join(lines)


def require_valid_mind_layout(data_dir: Path) -> LayoutValidation:
    result = validate_mind_layout(data_dir)
    if not result.is_valid:
        missing = "\n".join(f"- {path}" for path in result.missing_files.values())
        raise FileNotFoundError(f"Missing required MIND-small source files:\n{missing}")
    return result
