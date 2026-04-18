#!/usr/bin/env bash
# Build and publish landrecords-card-reader to PyPI.
#
# Requires:
#   - build + twine installed (auto-installed if missing)
#   - PyPI credentials via ~/.pypirc or TWINE_USERNAME / TWINE_PASSWORD env vars
#     (use __token__ as username with a PyPI API token as password)
#
# Usage:
#   ./publish.sh            # publish to PyPI
#   ./publish.sh --test     # publish to TestPyPI

set -euo pipefail

cd "$(dirname "$0")"

REPO_ARGS=()
if [[ "${1:-}" == "--test" ]]; then
    REPO_ARGS=(--repository testpypi)
    echo "Target: TestPyPI"
else
    echo "Target: PyPI"
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree has uncommitted changes. Commit or stash first." >&2
    exit 1
fi

VERSION=$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/version = "(.*)"/\1/')
echo "Publishing version: ${VERSION}"

python -m pip install --quiet --upgrade build twine

rm -rf dist build ./*.egg-info
python -m build

python -m twine check dist/*

python -m twine upload "${REPO_ARGS[@]}" dist/*

echo "Done. Tag this release with:  git tag v${VERSION} && git push origin v${VERSION}"
