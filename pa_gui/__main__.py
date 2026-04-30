"""
Entry point:  python -m pa_gui [--instrument-config PATH]
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from pa_gui.app import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser(description="PA GUI — Pipette Analysis")
    parser.add_argument(
        "--instrument-config",
        default=None,
        metavar="PATH",
        help="Path to instrument_config.json to pre-load.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("PA GUI")
    window = MainWindow(instrument_config_path=args.instrument_config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
