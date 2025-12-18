import sys
from pathlib import Path


def pytest_configure() -> None:
    # Repo layout uses the repository root as the `loka_light` package directory.
    # When running tests from inside that directory, we need its parent on sys.path
    # so `import loka_light` resolves correctly.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root.parent))
