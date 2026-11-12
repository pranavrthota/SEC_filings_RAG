"""Document loader abstraction.

A loader's contract is narrow on purpose: given a path, return a `Document`
whose blocks preserve reading order, page numbers, section headings, and - the
part general-purpose RAG gets wrong - tables as tables.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from ..models import Document

# Item headings in a 10-K / 10-Q, used to tag blocks with the section an
# analyst would cite ("Item 7. MD&A").
_ITEM_RE = re.compile(
    r"^\s*(item\s+\d+[A-Za-z]?\.?)\s*[-–—:.]?\s*(.{0,140})$", re.IGNORECASE
)
_PART_RE = re.compile(r"^\s*(part\s+[IVX]+)\b(.{0,60})$", re.IGNORECASE)
_UPPER_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,.&'()/-]{6,80}$")


class DocumentLoader(ABC):
    """Turns a file into a `Document`."""

    #: Lower-case file suffixes this loader claims.
    suffixes: tuple[str, ...] = ()

    def can_load(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.suffixes

    @abstractmethod
    def load(self, path: str | Path, **metadata: Any) -> Document:
        """Parse `path` into a Document. `metadata` overrides detected fields."""

    def load_many(self, paths: Iterable[str | Path], **metadata: Any) -> list[Document]:
        return [self.load(p, **metadata) for p in paths]

    # -- shared helpers ---------------------------------------------------- #

    @staticmethod
    def detect_heading(line: str) -> str | None:
        """Return a canonical section label if `line` looks like a heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > 160:
            return None

        m = _ITEM_RE.match(stripped)
        if m:
            item = m.group(1).rstrip(".").title().replace("Item", "Item")
            rest = m.group(2).strip(" .:-–—")
            return f"{item}. {rest}".strip(". ") if rest else item

        m = _PART_RE.match(stripped)
        if m:
            return m.group(1).upper().replace("PART", "Part")

        if _UPPER_HEADING_RE.match(stripped) and not stripped.endswith("."):
            return stripped.title()
        return None

    @staticmethod
    def make_doc_id(path: str | Path, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        stem = Path(path).stem
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-").lower() or "doc"


class LoaderRegistry:
    """Dispatches a path to the first loader that claims its suffix.

    This is the seam that makes the ingestion layer pluggable: dropping in an
    `unstructured`- or LlamaParse-backed loader is a `register()` call, and no
    downstream component changes.
    """

    def __init__(self, loaders: Iterable[DocumentLoader] = ()) -> None:
        self._loaders: list[DocumentLoader] = list(loaders)

    def register(self, loader: DocumentLoader, *, prepend: bool = True) -> "LoaderRegistry":
        """Add a loader. Prepended by default so overrides win."""
        if prepend:
            self._loaders.insert(0, loader)
        else:
            self._loaders.append(loader)
        return self

    @property
    def loaders(self) -> list[DocumentLoader]:
        return list(self._loaders)

    def loader_for(self, path: str | Path) -> DocumentLoader:
        for loader in self._loaders:
            if loader.can_load(path):
                return loader
        raise ValueError(
            f"No registered loader handles {Path(path).suffix!r}. "
            f"Registered: {[type(l).__name__ for l in self._loaders]}"
        )

    def load(self, path: str | Path, **metadata: Any) -> Document:
        return self.loader_for(path).load(path, **metadata)

    @classmethod
    def default(cls) -> "LoaderRegistry":
        """The stock registry: PDF, SEC HTML, and plain text."""
        from .html_loader import SecHtmlLoader
        from .pdf_loader import PdfPlumberLoader
        from .text_loader import TextLoader

        return cls([PdfPlumberLoader(), SecHtmlLoader(), TextLoader()])
