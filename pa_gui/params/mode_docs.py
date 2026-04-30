"""
Mode documentation viewer.

Loads markdown files from ``docs/modes/<ModeName>.md`` and renders them with
``QTextBrowser.setMarkdown``. Images referenced relatively (e.g.
``img/foo.png``) resolve against the markdown file's directory.

Used by the Pipeline Parameter Editor's "How it works" tab.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QTextBrowser

# Workspace root: …/pa_gui/params/mode_docs.py → up two = repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "docs", "modes"))


def docs_dir() -> str:
    """Return the absolute path of the docs/modes directory."""
    return _DOCS_DIR


def load_mode_doc(mode_name: str) -> Optional[str]:
    """Return the markdown text for *mode_name*, or None if no file exists."""
    if not mode_name:
        return None
    path = os.path.join(_DOCS_DIR, f"{mode_name}.md")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


class ModeDocsView(QTextBrowser):
    """Read-only markdown viewer for mode docs.

    - Resolves relative image paths against ``docs/modes/``.
    - External links open in the default browser.
    - Shows a friendly placeholder when no doc exists for the selected mode.
    """

    _PLACEHOLDER = (
        "_No documentation available for this mode yet._\n\n"
        "Add a markdown file at `docs/modes/<ModeName>.md` to populate this "
        "tab. See `docs/modes/README.md` for the conventions."
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setReadOnly(True)
        # Image resolution — QTextBrowser looks up resources relative to
        # ``setSearchPaths`` *and* the source URL. Setting the source to a
        # file:// URL inside docs_dir lets relative ``img/foo.png`` references
        # resolve correctly regardless of which mode is selected.
        self.setSearchPaths([_DOCS_DIR])

    def show_mode(self, mode_name: str) -> None:
        """Load and render the doc for *mode_name*."""
        md = load_mode_doc(mode_name)
        # Anchor the document at the docs directory so relative image paths
        # resolve correctly.  Setting source clears the document, so we set
        # source first and then setMarkdown.
        anchor = QUrl.fromLocalFile(os.path.join(_DOCS_DIR, "_anchor.md"))
        self.setSource(anchor)
        if md is None:
            self.setMarkdown(f"# {mode_name or 'Unknown mode'}\n\n{self._PLACEHOLDER}")
        else:
            self.setMarkdown(md)
