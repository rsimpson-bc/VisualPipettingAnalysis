"""
Entry point for running the PA server:

    python -m pa --instrument-config C:/path/to/instrument_config.json [--host 127.0.0.1] [--port 8765]

This module:
  1. Parses CLI args
  2. Loads Tier 1 (instrument config) into memory — once, at startup
  3. Starts the uvicorn server

Piomek2 should launch this process at startup and terminate it on close.
"""

import argparse
import uvicorn
from pa.state import session_state


def main():
    parser = argparse.ArgumentParser(description="PipetteAnalysis server")
    parser.add_argument(
        "--instrument-config",
        required=True,
        metavar="PATH",
        help="Absolute path to instrument_config.json (Tier 1).",
    )
    parser.add_argument(
        "--analysis-config",
        required=True,
        metavar="PATH",
        help="Absolute path to analysis_config.json (named pipelines + params).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # Load Tier 1 + analysis config once — before accepting any requests.
    session_state.startup(args.instrument_config, args.analysis_config)
    print(f"[PA] Instrument config loaded from: {args.instrument_config}")
    print(f"[PA] Analysis config loaded from:   {args.analysis_config}")
    print(f"[PA] Server starting on {args.host}:{args.port}")

    uvicorn.run(
        "pa.api.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
