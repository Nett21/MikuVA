"""Narzędzia PDF — odczyt tekstu z dokumentów w dozwolonych katalogach (Faza 8).

Oba narzędzia są SAFE: czytają plik i nic w nim nie zmieniają. Ograniczenia:

* **tylko dozwolone katalogi** — ta sama bramka co w ``tools/filesystem.py``
  (``host/paths.Workspace``), więc PDF spod ``~/.ssh`` nie istnieje,
* **limit stron** (``PDF_MAX_PAGES``) i limit znaków — dokument na 400 stron nie
  wypełni okna kontekstu modelu,
* **treść jest oznaczona jako niezaufana**: PDF pisał ktoś inny niż użytkownik
  tej rozmowy, więc po jego odczycie kolejne wywołanie o ryzyku MEDIUM+ wymaga
  potwierdzenia (bariera przeciw wstrzykiwaniu polecenia w treść dokumentu).

Biblioteka do czytania PDF-ów jest **opcjonalna**. Bez niej narzędzia są po prostu
niedostępne i model ich nie widzi — nie ma stanu „jest, ale się wywala". Obsługujemy
``pypdf`` oraz starszą nazwę ``PyPDF2``, bo w dystrybucjach bywa jedna albo druga.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import Field

from config import Settings, get_settings
from host.paths import PathNotAllowedError, Workspace
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# Kolejność prób importu: nowa nazwa pakietu, potem stara.
_READER_MODULES: Final[tuple[str, ...]] = ("pypdf", "PyPDF2")

# Ile znaków tekstu zwracamy najwyżej — niezależnie od liczby stron.
_MAX_TEXT_CHARS: Final[int] = 20_000


def reader_backend() -> str:
    """Nazwa dostępnej biblioteki PDF albo pusty łańcuch."""
    import importlib.util

    for module in _READER_MODULES:
        try:
            if importlib.util.find_spec(module) is not None:
                return module
        except (ImportError, ValueError):  # pragma: no cover - zależne od instalacji
            continue
    return ""


def _load_reader() -> Any:
    """Klasa ``PdfReader`` z dostępnej biblioteki."""
    import importlib

    for module_name in _READER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        reader = getattr(module, "PdfReader", None)
        if reader is not None:
            return reader
    raise ToolError(
        "brak biblioteki do czytania PDF-ów — zainstaluj pypdf (pip install pypdf)"
    )


class PdfReadArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)
    pages: int = Field(default=0, ge=0, le=500)
    start_page: int = Field(default=1, ge=1, le=5_000)


class PdfSearchArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)
    query: str = Field(min_length=1, max_length=200)
    pages: int = Field(default=0, ge=0, le=500)


class _PdfTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza: dozwolony katalog + opcjonalna biblioteka."""

    def __init__(
        self,
        spec: ToolSpec,
        workspace: Workspace,
        *,
        max_pages: int = 20,
        reader: Any | None = None,
    ) -> None:
        super().__init__(spec)
        self._workspace = workspace
        self._max_pages = max(1, int(max_pages))
        self._reader = reader

    def available(self) -> tuple[bool, str]:
        if self._reader is not None:
            return True, ""
        backend = reader_backend()
        if not backend:
            return False, (
                "brak biblioteki do czytania PDF-ów (pip install pypdf) — narzędzia PDF "
                "są niedostępne"
            )
        return True, ""

    def _resolve(self, raw: str) -> Path:
        try:
            path = self._workspace.resolve(raw, must_exist=True, must_be_file=True)
        except PathNotAllowedError as exc:
            raise ToolError(exc.message) from exc
        if path.suffix.lower() != ".pdf":
            raise ToolError(f"'{self._workspace.label(path)}' nie ma rozszerzenia .pdf")
        return path

    def _pages(self, path: Path, limit: int, *, start: int = 1) -> tuple[list[str], int]:
        """Tekst kolejnych stron. Zwraca listę stron i łączną liczbę stron w pliku."""
        reader_class = self._reader if self._reader is not None else _load_reader()
        try:
            reader = reader_class(str(path))
            pages = list(getattr(reader, "pages", []))
        except Exception as exc:
            raise ToolError(
                f"nie udało się otworzyć PDF-a '{self._workspace.label(path)}': {exc}"
            ) from exc

        total = len(pages)
        first = max(0, start - 1)
        wanted = min(limit or self._max_pages, self._max_pages)
        texts: list[str] = []
        for page in pages[first : first + wanted]:
            try:
                texts.append(str(page.extract_text() or ""))
            except Exception as exc:  # pragma: no cover - uszkodzona strona
                logger.debug("Nie udało się odczytać strony PDF: %s", exc)
                texts.append("")
        return texts, total


class PdfReadTool(_PdfTool[PdfReadArgs]):
    """``pdf.read`` — tekst z dokumentu."""

    async def run(self, args: PdfReadArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path)
        texts, total = self._pages(path, args.pages, start=args.start_page)
        joined = "\n\n".join(
            f"[str. {args.start_page + index}]\n{text.strip()}"
            for index, text in enumerate(texts)
            if text.strip()
        )
        truncated = len(joined) > _MAX_TEXT_CHARS
        if truncated:
            joined = joined[:_MAX_TEXT_CHARS].rstrip() + "\n[...] tekst obcięty"

        label = self._workspace.label(path)
        if not joined.strip():
            raise ToolError(
                f"'{label}' nie zawiera tekstu do odczytu (może być skanem — wtedy "
                "potrzebne byłoby OCR, którego asystent nie ma)"
            )
        return ToolResult.success(
            {
                "path": label,
                "pages_total": total,
                "pages_read": len(texts),
                "start_page": args.start_page,
                "truncated": truncated,
                "text": joined,
            },
            display=f"{label}: {len(texts)} z {total} stron, {len(joined)} znaków",
            untrusted=True,
        )


class PdfSearchTool(_PdfTool[PdfSearchArgs]):
    """``pdf.search`` — na których stronach występuje fraza."""

    async def run(self, args: PdfSearchArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path)
        texts, total = self._pages(path, args.pages)
        needle = args.query.casefold()
        matches: list[dict[str, Any]] = []
        for index, text in enumerate(texts, start=1):
            lowered = text.casefold()
            position = lowered.find(needle)
            if position < 0:
                continue
            start = max(0, position - 80)
            matches.append(
                {
                    "page": index,
                    "excerpt": " ".join(text[start : position + 160].split()),
                }
            )

        label = self._workspace.label(path)
        return ToolResult.success(
            {
                "path": label,
                "query": args.query,
                "pages_scanned": len(texts),
                "pages_total": total,
                "count": len(matches),
                "matches": matches,
            },
            display=f"'{args.query}' w {label}: {len(matches)} stron z {len(texts)} przeszukanych",
            untrusted=bool(matches),
        )


def build_pdf_tools(
    settings: Settings | None = None,
    *,
    workspace: Workspace | None = None,
    reader: Any | None = None,
) -> Sequence[Tool[Any]]:
    """Narzędzia PDF. ``reader`` podstawia atrapę biblioteki w testach."""
    active = settings or get_settings()
    area = workspace or Workspace.from_settings(active)
    max_pages = active.pdf_max_pages

    return (
        PdfReadTool(
            ToolSpec(
                name="pdf.read",
                description=(
                    "Extract text from a PDF file in an allowed directory. Use start_page and "
                    "pages to walk through long documents."
                ),
                summary="odczyt tekstu z PDF-a",
                args_model=PdfReadArgs,
                risk=RiskLevel.SAFE,
                timeout_s=30.0,
            ),
            area,
            max_pages=max_pages,
            reader=reader,
        ),
        PdfSearchTool(
            ToolSpec(
                name="pdf.search",
                description="Find which pages of a PDF contain a phrase, with short excerpts.",
                summary="szukanie frazy w PDF-ie",
                args_model=PdfSearchArgs,
                risk=RiskLevel.SAFE,
                timeout_s=30.0,
            ),
            area,
            max_pages=max_pages,
            reader=reader,
        ),
    )


__all__ = [
    "PdfReadArgs",
    "PdfReadTool",
    "PdfSearchArgs",
    "PdfSearchTool",
    "build_pdf_tools",
    "reader_backend",
]
