"""CLI entrypoint for SkyDreamer smoke checks."""

from __future__ import annotations

import json

from skydreamer.train.smoke import run_smoke_checks


def main() -> None:
    results = run_smoke_checks()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
