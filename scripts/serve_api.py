"""Convenience launcher for the FastAPI service: python scripts/serve_api.py [--port 8000]

Equivalent to: uvicorn kona_ebm.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("kona_ebm.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
