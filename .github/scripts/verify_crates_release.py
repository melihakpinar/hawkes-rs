"""Assert that a version is live on crates.io, and is the version the manifest names.

The counterpart to verify_pypi_release.py, for the same reason: a publish step's exit
code is not evidence that the release landed. `cargo publish` has no --skip-existing, so
the job also has to decide whether there is anything to publish at all, and this script
answers both questions.

Two modes:

  verify_crates_release.py <crate> <version>                assert, retrying while the
                                                            index propagates
  verify_crates_release.py <crate> <version> --check-only   one query, quiet; exit 0 if
                                                            the version is already there

`--check-only` is the idempotency guard. Re-pushing a tag must not fail the job, and
`cargo publish` errors on a version that already exists, so the workflow asks first.

Usage: verify_crates_release.py <crate> <version> [--check-only]
"""

import json
import sys
import time
import urllib.error
import urllib.request

ATTEMPTS = 12
SLEEP_SECONDS = 10
# crates.io rejects requests without a descriptive User-Agent.
HEADERS = {"User-Agent": "hawkes-rs release verification (github.com/melihakpinar/hawkes-rs)"}


def crate(name):
    request = urllib.request.Request(f"https://crates.io/api/v1/crates/{name}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def live(body, version):
    """(present, yanked) for `version` in the API response."""
    for entry in body.get("versions", []):
        if entry["num"] == version:
            return True, entry["yanked"]
    return False, False


def main(argv):
    name, version = argv[1], argv[2]
    check_only = "--check-only" in argv[3:]

    if check_only:
        try:
            present, yanked = live(crate(name), version)
        except Exception:
            # Unreachable index is not evidence the version exists; say "not there" and
            # let the publish step try. A duplicate publish fails loudly, which is safe.
            return 1
        return 0 if present and not yanked else 1

    # Initialised so the diagnosis below is defined even when every attempt raised —
    # a crate that does not exist at all returns 404 on each try.
    present, yanked, highest = False, False, None

    for attempt in range(1, ATTEMPTS + 1):
        try:
            body = crate(name)
        except Exception as exc:
            print(f"  attempt {attempt}: crates.io unreadable ({exc})")
            time.sleep(SLEEP_SECONDS)
            continue

        present, yanked = live(body, version)
        highest = body["crate"]["max_version"]
        if present and not yanked and highest == version:
            print(f"crates.io has {name} {version}, not yanked, and max_version matches")
            return 0
        print(
            f"  attempt {attempt}: present={present} yanked={yanked} "
            f"max_version={highest!r} wanted={version!r}"
        )
        time.sleep(SLEEP_SECONDS)

    if not present:
        print(f"NOT on crates.io after publish: {name} {version}", file=sys.stderr)
    elif yanked:
        print(f"on crates.io but yanked: {name} {version}", file=sys.stderr)
    else:
        print(
            f"max_version is {highest!r}, not the {version} this build published",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
