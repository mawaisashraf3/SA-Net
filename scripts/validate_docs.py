"""Check local Markdown targets and every shipped YAML config, without network access."""

import re
from pathlib import Path
from urllib.parse import unquote

from sanet.config import load_config


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    for markdown in [
        root / "README.md",
        *(root / "docs").rglob("*.md"),
        *(root / "results").rglob("*.md"),
        root / "data/README.md",
    ]:
        for target in re.findall(r"\]\(([^)]+)\)", markdown.read_text()):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            target = unquote(target.split(' "', 1)[0].strip("<>"))
            if not (markdown.parent / target).exists():
                errors.append(f"{markdown.relative_to(root)} -> {target}")
    configs = list((root / "configs").glob("*.yaml"))
    for path in configs:
        load_config(path)
    if errors:
        raise SystemExit("Broken local links:\n" + "\n".join(errors))
    print(f"Local documentation targets and {len(configs)} configurations verified")


if __name__ == "__main__":
    main()
