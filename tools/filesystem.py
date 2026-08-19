"""Narzędzia plikowe — wyłącznie w skonfigurowanych katalogach (Faza 8).

Poziomy ryzyka i co za nimi stoi:

============ ================================== ===============================
narzędzie    poziom                              dlaczego
============ ================================== ===============================
``fs.roots`` SAFE                                mówi tylko, gdzie wolno pracować
``fs.list``  SAFE                                odczyt katalogu
``fs.read``  SAFE                                odczyt pliku (limit rozmiaru)
``fs.search``SAFE                                szukanie po nazwie i treści
``fs.mkdir`` MEDIUM                              tworzy katalog we własnym obszarze
``fs.write`` MEDIUM → **HIGH przy nadpisaniu**   nowy plik to co innego niż
                                                 zniszczenie istniejącego
``fs.move``  HIGH                                przenosi dane, trudno cofnąć
``fs.delete``HIGH → **CRITICAL dla katalogu**    usunięcie jest nieodwracalne
============ ================================== ===============================

Cztery rzeczy, których nie da się obejść żadnym argumentem:

1. **obszar** — wszystko przechodzi przez :class:`host.paths.Workspace`, więc
   ``..``, ``~`` i dowiązania symboliczne prowadzące na zewnątrz są odrzucane,
2. **usunięcie dozwolonego katalogu** (czyli obszaru roboczego jako całości) jest
   odmawiane zawsze — nawet z potwierdzeniem i nawet z ``recursive=true``,
3. **usunięcie katalogu wymaga jawnego ``recursive=true``** i jest CRITICAL:
   potwierdzenie pokazuje, ile plików zniknie, jeszcze przed decyzją,
4. **limity** — rozmiar odczytu i zapisu, liczba wpisów, liczba usuwanych plików;
   wszystkie z konfiguracji, żadnego „bez ograniczeń".
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field

from config import Settings, get_settings
from i18n import t
from host.paths import (
    PathNotAllowedError,
    Workspace,
    count_tree,
    entry_info,
    looks_binary,
    read_text_limited,
    sorted_entries,
)
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# Ile plików przeglądamy przy szukaniu, zanim się poddamy. Bez tego limitu
# ``fs.search`` w dużym katalogu potrafi chodzić minutami.
_MAX_SCAN_FILES: Final[int] = 5_000


class _PathArgs(ToolArgs):
    path: str = Field(default=".", min_length=1, max_length=1_000)


class ListArgs(_PathArgs):
    limit: int = Field(default=50, ge=1, le=500)


class ReadArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)
    max_bytes: int | None = Field(default=None, ge=100, le=5_000_000)


class SearchArgs(ToolArgs):
    query: str = Field(min_length=1, max_length=200)
    path: str = Field(default=".", max_length=1_000)
    in_content: bool = False
    limit: int = Field(default=20, ge=1, le=200)


class WriteArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)
    content: str = Field(default="", max_length=1_000_000)
    mode: Literal["create", "overwrite", "append"] = "create"


class MkdirArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)


class MoveArgs(ToolArgs):
    source: str = Field(min_length=1, max_length=1_000)
    destination: str = Field(min_length=1, max_length=1_000)


class DeleteArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)
    recursive: bool = False


class _FilesystemTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Wspólna baza: każde narzędzie plikowe pracuje w jednym obszarze."""

    def __init__(self, spec: ToolSpec, workspace: Workspace) -> None:
        super().__init__(spec)
        self._workspace = workspace

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def available(self) -> tuple[bool, str]:
        if not self._workspace.roots:  # pragma: no cover - konfiguracja zawsze daje katalog
            return False, "nie skonfigurowano żadnego dozwolonego katalogu"
        return True, ""

    def _resolve(self, raw: str, **checks: bool) -> Path:
        try:
            return self._workspace.resolve(raw, **checks)
        except PathNotAllowedError as exc:
            # Odmowa ścieżki to komunikat dla modelu, nie awaria programu.
            raise ToolError(exc.message) from exc


# --------------------------------------------------------------------------- #
# SAFE: odczyt
# --------------------------------------------------------------------------- #


class RootsTool(_FilesystemTool[ToolArgs]):
    """``fs.roots`` — gdzie w ogóle wolno pracować."""

    async def run(self, args: ToolArgs, ctx: ToolContext) -> ToolResult:
        roots = [
            {"path": str(root), "exists": root.is_dir(), "name": root.name}
            for root in self._workspace.roots
        ]
        display = ", ".join(str(root) for root in self._workspace.roots)
        return ToolResult.success(
            {
                "roots": roots,
                "max_read_bytes": self._workspace.max_read_bytes,
                "max_write_bytes": self._workspace.max_write_bytes,
                "note": (
                    "Only these directories are reachable. Paths outside them, including "
                    "'..' and symlinks, are refused."
                ),
            },
            display=f"dozwolone katalogi: {display}",
        )


class ListTool(_FilesystemTool[ListArgs]):
    """``fs.list`` — zawartość katalogu."""

    async def run(self, args: ListArgs, ctx: ToolContext) -> ToolResult:
        directory = self._resolve(args.path, must_exist=True, must_be_dir=True)
        limit = min(args.limit, self._workspace.max_entries)
        try:
            entries = sorted_entries(directory, limit=limit)
        except PathNotAllowedError as exc:
            raise ToolError(exc.message) from exc

        items = [entry_info(entry, workspace=self._workspace) for entry in entries]
        label = self._workspace.label(directory)
        return ToolResult.success(
            {"path": label, "count": len(items), "entries": items},
            display=t("fs.listing", path=label, count=len(items)),
        )


class ReadTool(_FilesystemTool[ReadArgs]):
    """``fs.read`` — treść pliku tekstowego."""

    async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path, must_exist=True, must_be_file=True)
        if looks_binary(path):
            label = self._workspace.label(path)
            raise ToolError(t("fs.binary", path=label))
        limit = min(
            args.max_bytes or self._workspace.max_read_bytes,
            self._workspace.max_read_bytes,
        )
        try:
            text, truncated = read_text_limited(path, limit)
        except PathNotAllowedError as exc:
            raise ToolError(exc.message) from exc

        label = self._workspace.label(path)
        return ToolResult.success(
            {
                "path": label,
                "bytes": path.stat().st_size if path.exists() else 0,
                "truncated": truncated,
                "content": text,
            },
            display=f"{label}: {len(text)} znaków" + (" (obcięte)" if truncated else ""),
            # Treść pliku pisał ktoś inny niż użytkownik w tej rozmowie — dla
            # routera to dane niezaufane, więc kolejne wywołanie o wyższym ryzyku
            # będzie wymagało potwierdzenia.
            untrusted=True,
        )


class SearchTool(_FilesystemTool[SearchArgs]):
    """``fs.search`` — szukanie po nazwie, opcjonalnie po treści."""

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        directory = self._resolve(args.path or ".", must_exist=True, must_be_dir=True)
        needle = args.query.casefold()
        limit = min(args.limit, self._workspace.max_entries)
        hits: list[dict[str, Any]] = []
        scanned = 0

        for current, dirnames, filenames in os.walk(directory):
            # Katalogi ukryte pomijamy: „.git" i „.cache" to nie treść użytkownika.
            dirnames[:] = [name for name in sorted(dirnames) if not name.startswith(".")]
            for filename in sorted(filenames):
                scanned += 1
                if scanned > _MAX_SCAN_FILES or len(hits) >= limit:
                    break
                candidate = Path(current) / filename
                matched_name = needle in filename.casefold()
                line_number = 0
                if not matched_name and args.in_content:
                    line_number = self._first_match(candidate, needle)
                if not matched_name and line_number == 0:
                    continue
                info = entry_info(candidate, workspace=self._workspace)
                if line_number:
                    info["line"] = line_number
                hits.append(info)
            if scanned > _MAX_SCAN_FILES or len(hits) >= limit:
                break

        return ToolResult.success(
            {"query": args.query, "count": len(hits), "matches": hits, "scanned": scanned},
            display=t(
                "fs.search_hits",
                query=args.query,
                count=len(hits),
                path=self._workspace.label(directory),
            ),
            untrusted=bool(hits) and args.in_content,
        )

    def _first_match(self, path: Path, needle: str) -> int:
        """Numer pierwszej linii z frazą albo 0. Pliki binarne pomijamy."""
        if looks_binary(path):
            return 0
        try:
            text, _ = read_text_limited(path, self._workspace.max_read_bytes)
        except PathNotAllowedError:
            return 0
        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line.casefold():
                return number
        return 0


# --------------------------------------------------------------------------- #
# MEDIUM: tworzenie
# --------------------------------------------------------------------------- #


class MkdirTool(_FilesystemTool[MkdirArgs]):
    """``fs.mkdir`` — nowy katalog w obszarze roboczym."""

    async def run(self, args: MkdirArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path)
        if path.exists():
            return ToolResult.success(
                {"path": self._workspace.label(path), "created": False},
                display=t("fs.dir_exists", path=self._workspace.label(path)),
            )
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(t("fs.mkdir_failed", error=exc)) from exc
        return ToolResult.success(
            {"path": self._workspace.label(path), "created": True},
            display=t("fs.dir_created", path=self._workspace.label(path)),
        )

    async def preview(self, args: MkdirArgs, ctx: ToolContext) -> str:
        return f"utworzy katalog {args.path}"


class WriteTool(_FilesystemTool[WriteArgs]):
    """``fs.write`` — zapis pliku. Nadpisanie istniejącego to już HIGH."""

    def dynamic_risk(self, args: WriteArgs) -> RiskLevel:
        """Nowy plik = MEDIUM. Nadpisanie albo dopisanie do istniejącego = HIGH.

        Powód jest praktyczny: utworzenie pliku da się cofnąć jego usunięciem,
        a nadpisanie niszczy treść, której nikt nie zdążył zobaczyć.
        """
        try:
            path = self._workspace.resolve(args.path)
        except PathNotAllowedError:
            # Ścieżkę odrzuci bramka SCHEMA/NORMALIZE — tu wybieramy ostrożniej.
            return RiskLevel.HIGH
        if path.exists() and args.mode != "create":
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM

    def confirmation(
        self, args: WriteArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        try:
            path = self._workspace.resolve(args.path)
        except PathNotAllowedError:
            return super().confirmation(args, language=language)
        polish = language == "pl"
        label = self._workspace.label(path)
        existing = path.stat().st_size if path.exists() else 0
        summary = (
            f"Nadpisze plik {label} ({existing} B → {len(args.content.encode('utf-8'))} B)"
            if polish
            else f"Overwrite file {label} ({existing} B → {len(args.content.encode('utf-8'))} B)"
        )
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=self.effective_risk(args),
            summary=summary,
            details=[
                f"{'pełna ścieżka' if polish else 'full path'}: {path}",
                f"{'tryb' if polish else 'mode'}: {args.mode}",
            ],
            preview=args.content[:400],
            language=language,
        )

    async def run(self, args: WriteArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path)
        data = args.content.encode("utf-8")
        if len(data) > self._workspace.max_write_bytes:
            raise ToolError(
                t("fs.too_large", size=len(data), limit=self._workspace.max_write_bytes)
            )
        if path.is_dir():
            raise ToolError(t("fs.is_a_directory", path=self._workspace.label(path)))
        if args.mode == "create" and path.exists():
            raise ToolError(
                t("fs.file_exists", path=self._workspace.label(path))
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == "append":
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(args.content)
            else:
                # Zapis przez plik tymczasowy i podmianę: przerwanie w połowie nie
                # zostawia pliku uciętego w środku.
                temporary = path.with_name(path.name + ".miku-tmp")
                temporary.write_text(args.content, encoding="utf-8", newline="\n")
                os.replace(temporary, path)
        except OSError as exc:
            raise ToolError(t("fs.write_failed", error=exc)) from exc

        label = self._workspace.label(path)
        return ToolResult.success(
            {"path": label, "bytes": len(data), "mode": args.mode},
            display=f"zapisano {label} ({len(data)} B, {args.mode})",
        )

    async def preview(self, args: WriteArgs, ctx: ToolContext) -> str:
        return f"zapisze {len(args.content.encode('utf-8'))} B do {args.path} (tryb {args.mode})"


# --------------------------------------------------------------------------- #
# HIGH / CRITICAL: przenoszenie i usuwanie
# --------------------------------------------------------------------------- #


class MoveTool(_FilesystemTool[MoveArgs]):
    """``fs.move`` — przenieś albo zmień nazwę. Zawsze wymaga potwierdzenia."""

    def normalize(self, args: MoveArgs) -> MoveArgs:
        """Przeniesienia dozwolonego katalogu odmawiamy przed pytaniem o zgodę."""
        source = self._resolve(args.source)
        if self._workspace.is_root(source):
            raise ValueError(t("fs.no_move_root"))
        return args

    def confirmation(self, args: MoveArgs, *, language: str = "en") -> ConfirmationRequest | None:
        polish = language == "pl"
        summary = (
            f"Przeniesie {args.source} → {args.destination}"
            if polish
            else f"Move {args.source} → {args.destination}"
        )
        details: list[str] = []
        try:
            source = self._workspace.resolve(args.source)
            destination = self._workspace.resolve(args.destination)
            details = [f"{source}", f"→ {destination}"]
            if destination.exists():
                details.append(
                    "cel istnieje i zostanie zastąpiony"
                    if polish
                    else "target exists and will be replaced"
                )
        except PathNotAllowedError:
            pass
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=self.effective_risk(args),
            summary=summary,
            details=details,
            language=language,
        )

    async def run(self, args: MoveArgs, ctx: ToolContext) -> ToolResult:
        source = self._resolve(args.source, must_exist=True)
        destination = self._resolve(args.destination)
        if self._workspace.is_root(source):
            raise ToolError(t("fs.no_move_root"))
        if destination.exists() and destination.is_dir() and not source.is_dir():
            destination = destination / source.name

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except (OSError, shutil.Error) as exc:
            raise ToolError(t("fs.move_failed", error=exc)) from exc

        return ToolResult.success(
            {
                "source": self._workspace.label(source),
                "destination": self._workspace.label(destination),
            },
            display=(
                f"przeniesiono {self._workspace.label(source)} → "
                f"{self._workspace.label(destination)}"
            ),
        )

    async def preview(self, args: MoveArgs, ctx: ToolContext) -> str:
        return f"przeniesie {args.source} do {args.destination}"


class DeleteTool(_FilesystemTool[DeleteArgs]):
    """``fs.delete`` — usunięcie. Katalog rekurencyjnie to poziom CRITICAL."""

    def normalize(self, args: DeleteArgs) -> DeleteArgs:
        """Odmowy, które muszą zapaść PRZED pytaniem o zgodę.

        Bramka NORMALIZE jest przed CONFIRM, i to jest tu istotne: nie ma sensu
        pytać użytkownika „usunąć cały obszar roboczy?", jeśli odpowiedź i tak
        będzie odrzucona. Zauważone na żywo — użytkownik dostawał modal, po którym
        narzędzie mówiło „nie usuwam".
        """
        path = self._resolve(args.path)
        if self._workspace.is_root(path):
            raise ValueError(t("fs.no_delete_root", path=self._workspace.label(path)))
        if path.is_dir() and not path.is_symlink() and not args.recursive:
            raise ValueError(t("fs.need_recursive", path=self._workspace.label(path)))
        return args

    def dynamic_risk(self, args: DeleteArgs) -> RiskLevel:
        """Plik = HIGH. Katalog z zawartością = CRITICAL.

        Rozróżnienie nie jest kosmetyczne: usunięcie jednego pliku to jedna
        pomyłka, usunięcie katalogu to pomyłka razy liczba plików w drzewie.
        """
        try:
            path = self._workspace.resolve(args.path)
        except PathNotAllowedError:
            return RiskLevel.CRITICAL
        if path.is_dir() and not path.is_symlink():
            return RiskLevel.CRITICAL
        return RiskLevel.HIGH

    def confirmation(
        self, args: DeleteArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        polish = language == "pl"
        try:
            path = self._workspace.resolve(args.path)
        except PathNotAllowedError:
            return super().confirmation(args, language=language)

        label = self._workspace.label(path)
        details: list[str] = [f"{'pełna ścieżka' if polish else 'full path'}: {path}"]
        preview: str | None = None
        if path.is_dir() and not path.is_symlink():
            files, directories = count_tree(path, limit=self._workspace.max_delete_entries + 1)
            summary = (
                f"USUNIE KATALOG {label} wraz z zawartością: {files} plików, "
                f"{directories} podkatalogów"
                if polish
                else f"DELETE DIRECTORY {label} with contents: {files} files, "
                f"{directories} subdirectories"
            )
            details.append(
                "operacja jest nieodwracalna" if polish else "this cannot be undone"
            )
            names = [entry.name for entry in sorted_entries(path, limit=10)]
            preview = "\n".join(names)
        else:
            size = path.stat().st_size if path.exists() else 0
            summary = (
                f"Usunie plik {label} ({size} B)" if polish else f"Delete file {label} ({size} B)"
            )

        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=self.effective_risk(args),
            summary=summary,
            details=details,
            preview=preview,
            language=language,
        )

    async def run(self, args: DeleteArgs, ctx: ToolContext) -> ToolResult:
        path = self._resolve(args.path, must_exist=True)

        # Blokada nie do obejścia: obszar roboczy jako całość nie ginie.
        if self._workspace.is_root(path):
            raise ToolError(t("fs.no_delete_root", path=self._workspace.label(path)))

        if path.is_dir() and not path.is_symlink():
            if not args.recursive:
                raise ToolError(t("fs.need_recursive", path=self._workspace.label(path)))
            files, directories = count_tree(path, limit=self._workspace.max_delete_entries + 1)
            if files + directories > self._workspace.max_delete_entries:
                raise ToolError(
                    t(
                        "fs.too_many_entries",
                        limit=self._workspace.max_delete_entries,
                        files=files,
                        directories=directories,
                    )
                )
            try:
                shutil.rmtree(path)
            except OSError as exc:
                raise ToolError(t("fs.rmtree_failed", error=exc)) from exc
            return ToolResult.success(
                {
                    "path": self._workspace.label(path),
                    "kind": "dir",
                    "files": files,
                    "directories": directories,
                },
                display=t(
                    "fs.dir_deleted",
                    path=self._workspace.label(path),
                    files=files,
                    directories=directories,
                ),
            )

        try:
            size = path.stat().st_size
            path.unlink()
        except OSError as exc:
            raise ToolError(t("fs.unlink_failed", error=exc)) from exc
        return ToolResult.success(
            {"path": self._workspace.label(path), "kind": "file", "bytes": size},
            display=t("fs.file_deleted", path=self._workspace.label(path)),
        )

    async def preview(self, args: DeleteArgs, ctx: ToolContext) -> str:
        try:
            path = self._workspace.resolve(args.path, must_exist=True)
        except PathNotAllowedError as exc:
            return exc.message
        if path.is_dir():
            files, directories = count_tree(path, limit=self._workspace.max_delete_entries + 1)
            return (
                f"usunęłoby katalog {self._workspace.label(path)}: {files} plików, "
                f"{directories} podkatalogów"
            )
        return f"usunęłoby plik {self._workspace.label(path)}"


# --------------------------------------------------------------------------- #
# Rejestracja
# --------------------------------------------------------------------------- #


def build_filesystem_tools(
    settings: Settings | None = None, *, workspace: Workspace | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia plikowe działające w podanym (albo skonfigurowanym) obszarze."""
    active = settings or get_settings()
    area = workspace or Workspace.from_settings(active)

    return (
        RootsTool(
            ToolSpec(
                name="fs.roots",
                description=(
                    "List the directories the file tools may access. Call this first when "
                    "you are unsure whether a path is reachable."
                ),
                summary=t("spec.fs_roots"),
                args_model=ToolArgs,
                risk=RiskLevel.SAFE,
                timeout_s=5.0,
            ),
            area,
        ),
        ListTool(
            ToolSpec(
                name="fs.list",
                description=(
                    "List files and directories inside an allowed directory. Paths are "
                    "relative to that directory: use '.' for the directory itself, "
                    "'notes' for a subdirectory. Call fs.roots first if unsure."
                ),
                summary=t("spec.fs_list"),
                args_model=ListArgs,
                risk=RiskLevel.SAFE,
                timeout_s=10.0,
            ),
            area,
        ),
        ReadTool(
            ToolSpec(
                name="fs.read",
                description=(
                    "Read a text file from an allowed directory. The path is relative to "
                    "that directory, e.g. 'plan.txt' or 'notes/bike.md'. Binary files are "
                    "refused and long files are truncated."
                ),
                summary="odczyt pliku tekstowego",
                args_model=ReadArgs,
                risk=RiskLevel.SAFE,
                timeout_s=10.0,
            ),
            area,
        ),
        SearchTool(
            ToolSpec(
                name="fs.search",
                description=(
                    "Find files by name, or by content when in_content=true, inside an "
                    "allowed directory."
                ),
                summary=t("spec.fs_search"),
                args_model=SearchArgs,
                risk=RiskLevel.SAFE,
                timeout_s=20.0,
            ),
            area,
        ),
        MkdirTool(
            ToolSpec(
                name="fs.mkdir",
                description="Create a directory inside an allowed directory.",
                summary="nowy katalog",
                args_model=MkdirArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=10.0,
            ),
            area,
        ),
        WriteTool(
            ToolSpec(
                name="fs.write",
                description=(
                    "Write a text file inside an allowed directory. mode='create' fails if "
                    "the file exists; 'overwrite' and 'append' need the user's confirmation."
                ),
                summary="zapis pliku (nadpisanie wymaga zgody)",
                args_model=WriteArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=15.0,
                idempotent=False,
            ),
            area,
        ),
        MoveTool(
            ToolSpec(
                name="fs.move",
                description=(
                    "Move or rename a file or directory. Both paths must be inside allowed "
                    "directories. Requires the user's confirmation."
                ),
                summary="przenoszenie i zmiana nazwy",
                args_model=MoveArgs,
                risk=RiskLevel.HIGH,
                timeout_s=30.0,
                idempotent=False,
            ),
            area,
        ),
        DeleteTool(
            ToolSpec(
                name="fs.delete",
                description=(
                    "Delete a file, or a directory when recursive=true. Irreversible: always "
                    "requires the user's confirmation, and never removes an allowed root itself."
                ),
                summary="usuwanie (nieodwracalne, wymaga zgody)",
                args_model=DeleteArgs,
                risk=RiskLevel.HIGH,
                timeout_s=30.0,
                idempotent=False,
            ),
            area,
        ),
    )


__all__ = [
    "DeleteArgs",
    "DeleteTool",
    "ListArgs",
    "ListTool",
    "MkdirArgs",
    "MkdirTool",
    "MoveArgs",
    "MoveTool",
    "ReadArgs",
    "ReadTool",
    "RootsTool",
    "SearchArgs",
    "SearchTool",
    "WriteArgs",
    "WriteTool",
    "build_filesystem_tools",
]
