"""Assert that the files just uploaded are actually on PyPI, and are ours.

`twine upload --skip-existing` exits 0 whether it uploaded everything or skipped
everything: "already exists" is a warning, not an error. So the upload step's exit code
says nothing about whether the release is complete, and a publish job that only checks
it is a verification step that verifies nothing.

This compares the local `dist/` against the `/simple/` index pip actually resolves
against, by **sha256**, not by filename. A name match alone would pass if PyPI held a
different build under the same name.

Usage: verify_pypi_release.py <project> <dist-dir>
"""

import hashlib
import json
import pathlib
import sys
import time
import urllib.request

ATTEMPTS = 12
SLEEP_SECONDS = 10


def digest(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def published(project):
    """{filename: sha256} from the index pip resolves against."""
    request = urllib.request.Request(
        f"https://pypi.org/simple/{project}/",
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
    return {f["filename"]: f.get("hashes", {}).get("sha256") for f in body["files"]}


def main(project, dist):
    local = {p.name: digest(p) for p in sorted(pathlib.Path(dist).iterdir()) if p.is_file()}
    if not local:
        print(f"nothing in {dist}: refusing to report a successful publish", file=sys.stderr)
        return 1
    print(f"{len(local)} local files to account for")

    missing, mismatched = list(local), []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            remote = published(project)
        except Exception as exc:                      # index briefly unavailable
            print(f"  attempt {attempt}: index unreadable ({exc})")
            time.sleep(SLEEP_SECONDS)
            continue

        missing = [name for name in local if name not in remote]
        mismatched = [
            name
            for name, sha in local.items()
            if name in remote and remote[name] is not None and remote[name] != sha
        ]
        if not missing and not mismatched:
            print(f"all {len(local)} files present on PyPI with matching sha256:")
            for name in local:
                print(f"  {name}")
            return 0
        print(f"  attempt {attempt}: missing {len(missing)}, mismatched {len(mismatched)}")
        time.sleep(SLEEP_SECONDS)

    if missing:
        print(f"NOT on PyPI after upload: {missing}", file=sys.stderr)
    if mismatched:
        print(f"on PyPI but a different build: {mismatched}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
