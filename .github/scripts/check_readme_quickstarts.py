"""Fails if a README quickstart block differs from the committed example it claims to be.

The README says its two examples are the files CI runs. This is what makes that
sentence checkable: each fenced block must equal the corresponding file with only its
module-level header (the `//!` lines, or the docstring) removed.

Usage: check_readme_quickstarts.py [repo root]
"""

import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
readme = (root / "README.md").read_text()


def body_of_rust(path: pathlib.Path) -> str:
    lines = path.read_text().splitlines()
    while lines and (lines[0].startswith("//!") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip() + "\n"


def body_of_python(path: pathlib.Path) -> str:
    text = path.read_text()
    text = re.sub(r'\A"""[\s\S]*?"""\n+', "", text)
    return text.strip() + "\n"


def block(language: str) -> str:
    match = re.search(rf"```{language}\n([\s\S]*?)```", readme)
    if not match:
        sys.exit(f"README has no ```{language} block")
    return match.group(1).strip() + "\n"


failures = []
for language, path, body in [
    ("python", root / "hawkes-python/examples/quickstart.py", body_of_python),
    ("rust", root / "hawkes/examples/quickstart.rs", body_of_rust),
]:
    if block(language) != body(path):
        failures.append(f"README ```{language} block differs from {path}")
if failures:
    sys.exit("\n".join(failures))
print("README quickstart blocks match the committed examples")
