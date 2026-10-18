"""Exercise the release version/build path without uploading to PyPI."""

import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "99.88.77"


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def evaluate() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    for requirement in (
        "workflow_dispatch:",
        "RELEASE_VERSION: ${{ inputs.version }}",
        'test "$RELEASE_REF" = refs/heads/main',
        'uv version --package mem-alive --frozen "$RELEASE_VERSION"',
        "uv build --package mem-alive --no-sources",
        "needs: build",
        "environment: pypi",
        "id-token: write",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 # release/v1",
    ):
        assert requirement in workflow, f"Release workflow is missing: {requirement}"

    with tempfile.TemporaryDirectory() as temporary_dir:
        checkout = Path(temporary_dir)
        shutil.copy2(ROOT / "pyproject.toml", checkout / "pyproject.toml")
        package = checkout / "packages/mem-alive"
        shutil.copytree(ROOT / "packages/mem-alive", package)
        metadata = package / "pyproject.toml"
        original = metadata.read_bytes()

        invalid = run(
            "uv", "version", "--package", "mem-alive", "--frozen", "not-a-version", cwd=checkout
        )
        assert invalid.returncode != 0, "Invalid release version was accepted"
        assert metadata.read_bytes() == original, "Invalid version modified package metadata"

        selected = run("uv", "version", "--package", "mem-alive", "--frozen", VERSION, cwd=checkout)
        assert selected.returncode == 0, selected.stderr
        assert tomllib.loads(metadata.read_text())["project"]["version"] == VERSION

        build = run("uv", "build", "--package", "mem-alive", "--no-sources", cwd=checkout)
        assert build.returncode == 0, build.stderr
        distributions = checkout / "dist"
        wheel, = distributions.glob("*.whl")
        source, = distributions.glob("*.tar.gz")

        with zipfile.ZipFile(wheel) as archive:
            (metadata_name,) = (
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            wheel_metadata = BytesParser().parsebytes(archive.read(metadata_name))
            assert wheel_metadata["Name"] == "mem-alive"
            assert wheel_metadata["Version"] == VERSION
            source_file = package / "src/mem_alive/core/memory.py"
            assert archive.read("mem_alive/core/memory.py") == source_file.read_bytes()

        with tarfile.open(source, "r:gz") as archive:
            (project_file,) = (
                member for member in archive.getmembers() if member.name.endswith("/pyproject.toml")
            )
            project_content = archive.extractfile(project_file)
            assert project_content is not None
            source_metadata = tomllib.loads(project_content.read().decode())
            assert source_metadata["project"]["version"] == VERSION

    print(f"PASS: invalid version rejected; both distributions contain mem-alive {VERSION}")


if __name__ == "__main__":
    os.environ.setdefault("UV_CACHE_DIR", "/tmp/mem-alive-uv-cache")
    evaluate()
