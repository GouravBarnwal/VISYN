import hashlib
import json
from pathlib import Path

from src.production_config import ARTIFACTS_ROOT


PRODUCTION_ROOT = ARTIFACTS_ROOT / "production"
MANIFEST_PATH = (
    PRODUCTION_ROOT / "production_artifact_manifest.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Production manifest not found: {MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def resolve_artifact(relative_path: str) -> Path:
    return ARTIFACTS_ROOT / Path(relative_path)


def validate_artifact(
    artifact: dict,
) -> tuple[bool, str]:
    path = resolve_artifact(artifact["path"])

    if not path.exists():
        return False, f"MISSING: {path}"

    actual_size = path.stat().st_size

    if actual_size != artifact["size_bytes"]:
        return (
            False,
            f"SIZE MISMATCH: {path}",
        )

    actual_hash = sha256_file(path)

    if actual_hash != artifact["sha256"]:
        return (
            False,
            f"HASH MISMATCH: {path}",
        )

    return True, str(path)


def main():
    manifest = load_manifest()

    print("=" * 72)
    print("VISIONFORGE — PRODUCTION ARTIFACT INTEGRITY")
    print("=" * 72)

    failures = []
    checked = 0

    # Reference-bank manifest
    artifact = manifest["reference_split"]["manifest"]

    ok, message = validate_artifact(artifact)

    checked += 1

    if ok:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        failures.append(message)

    # Threshold artifact
    artifact = manifest["thresholds"]["artifact"]

    ok, message = validate_artifact(artifact)

    checked += 1

    if ok:
        print(f"PASS  {message}")
    else:
        print(f"FAIL  {message}")
        failures.append(message)

    # Reference banks
    for category, banks in manifest[
        "reference_banks"
    ].items():

        for layer in ("L4", "L8"):
            artifact = banks[layer]["artifact"]

            ok, message = validate_artifact(artifact)

            checked += 1

            if ok:
                print(
                    f"PASS  {category:12s} "
                    f"{layer}"
                )
            else:
                print(
                    f"FAIL  {category:12s} "
                    f"{layer} — {message}"
                )
                failures.append(message)

    print()
    print(f"Artifacts checked: {checked}")

    if failures:
        print(
            f"Integrity failures: {len(failures)}"
        )
        print()
        print(
            "PRODUCTION ARTIFACT INTEGRITY: FAIL"
        )
        raise SystemExit(1)

    print("Integrity failures: 0")
    print()
    print(
        "PRODUCTION ARTIFACT INTEGRITY: PASS"
    )


if __name__ == "__main__":
    main()