from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every BRISC2025 file against manifest.csv."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "brisc2025",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(data_root: Path) -> int:
    manifest_path = data_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "sha256"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        relative_path = Path(str(row.relative_path).replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"unsafe path: {row.relative_path}")
            continue
        file_path = data_root / relative_path
        if not file_path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual = sha256_file(file_path)
        if actual != row.sha256:
            failures.append(f"checksum mismatch: {relative_path}")

    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... and {len(failures) - 20} more"
        raise RuntimeError(f"Dataset verification failed:\n{preview}{suffix}")

    print(f"Verified {len(manifest)} dataset files under {data_root}")
    return len(manifest)


def main() -> None:
    args = parse_args()
    verify_dataset(args.data_root)


if __name__ == "__main__":
    main()
