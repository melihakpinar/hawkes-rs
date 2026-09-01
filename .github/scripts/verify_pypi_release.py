"""Assert that the files just uploaded are actually on PyPI, and are ours.

`twine upload --skip-existing` exits 0 whether it uploaded everything or skipped
everything: "already exists" is a warning, not an error. So the upload step's exit code
says nothing about whether the release is complete, and a publish job that only checks
it is a verification step that verifies nothing.

**Presence is checked for every file; sha256 only for the files this run uploaded.**
Wheels are not bit-reproducible — rebuilding 0.1.1 produced five wheels differing from
the published ones by one to three bytes, while the sdist was byte-identical — so a file
that `--skip-existing` skipped will legitimately differ from the copy already on PyPI.
Comparing it would be a false positive, and was one. Which files this run is responsible
for is decided by snapshotting the index *before* the upload: anything absent then is
this run's to account for by digest.

Modes:

  verify_pypi_release.py <project> --snapshot <path>
      Record the filenames PyPI already has. Run before the upload.

  verify_pypi_release.py <project> <dist-dir> [--uploaded-since <path>]
      Assert. Without the snapshot every local file is treated as this run's, which is
      the strict reading and the right default for a release that uploads everything.

Usage: verify_pypi_release.py <project> <dist-dir|--snapshot path> [--uploaded-since path]
"""

import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ATTEMPTS = 12
SLEEP_SECONDS = 10


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def snapshot(project, path):
    """Filenames PyPI has right now. A project with no releases yet is not an error."""
    try:
        names = sorted(published(project))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        names = []
    pathlib.Path(path).write_text(json.dumps(names, indent=2) + "\n")
    print(f"PyPI already has {len(names)} files for {project}")
    for name in names:
        print(f"  {name}")
    return 0


def main(argv):
    project = argv[1]

    if argv[2] == "--snapshot":
        return snapshot(project, argv[3])

    dist = argv[2]
    before = set()
    if "--uploaded-since" in argv:
        path = pathlib.Path(argv[argv.index("--uploaded-since") + 1])
        before = set(json.loads(path.read_text()))

    local = {p.name: digest(p) for p in sorted(pathlib.Path(dist).iterdir()) if p.is_file()}
    if not local:
        print(f"nothing in {dist}: refusing to report a successful publish", file=sys.stderr)
        return 1

    # What this run put there, and therefore what its bytes must match.
    ours = sorted(name for name in local if name not in before)
    skipped = sorted(name for name in local if name in before)
    print(f"{len(local)} local files: {len(ours)} uploaded by this run, {len(skipped)} already present")
    if skipped:
        print("  already present, checked for presence only (wheels are not bit-reproducible):")
        for name in skipped:
            print(f"    {name}")

    missing, mismatched = list(local), []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            remote = published(project)
        except Exception as exc:
            print(f"  attempt {attempt}: index unreadable ({exc})")
            time.sleep(SLEEP_SECONDS)
            continue

        missing = [name for name in local if name not in remote]
        mismatched = [
            name
            for name in ours
            if name in remote and remote[name] is not None and remote[name] != local[name]
        ]
        if not missing and not mismatched:
            print(f"all {len(local)} files present on PyPI; {len(ours)} verified by sha256")
            for name in local:
                print(f"  {name}")
            return 0
        print(f"  attempt {attempt}: missing {len(missing)}, mismatched {len(mismatched)}")
        time.sleep(SLEEP_SECONDS)

    if missing:
        print(f"NOT on PyPI after upload: {missing}", file=sys.stderr)
    if mismatched:
        print(f"uploaded by this run but a different build is on PyPI: {mismatched}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
