"""Narzędzia notatek — pamięć asystenta, nie dysk użytkownika (Faza 8).

============== ======== ======================================================
narzędzie      poziom   uzasadnienie
============== ======== ======================================================
``notes.search``SAFE    odczyt własnych notatek
``notes.read`` SAFE     odczyt jednej notatki
``notes.create``MEDIUM  zapis we WŁASNYCH danych asystenta
``notes.append``MEDIUM  dopisanie akapitu (treść zostaje, nic nie ginie)
``notes.delete``HIGH    usunięcie jest nieodwracalne, wymaga zgody
============== ======== ======================================================

Notatki mieszkają w bazie SQLite z Fazy 5, a nie w plikach — i to jest celowe:

* są wyszukiwalne razem z rozmowami (FTS5) i po znaczeniu (wektory z Fazy 6),
* nie wymagają dostępu do dysku użytkownika, więc narzędzia notatek działają też
  wtedy, gdy narzędzia plikowe nie mają ani jednego dozwolonego katalogu,
* usunięcie notatki zabiera też jej wektor — inaczej „zapomnij" zostawiałoby ślad.

Wszystko idzie przez :class:`brain.memory.ConversationMemory`, nie wprost do bazy:
to ona pilnuje, żeby treść i wektor nie rozjechały się przy dopisywaniu.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from config import Settings
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from i18n import t

logger = logging.getLogger(__name__)


class NotesSearchArgs(ToolArgs):
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=10, ge=1, le=50)


class NotesReadArgs(ToolArgs):
    note_id: int = Field(ge=1)


class NotesCreateArgs(ToolArgs):
    body: str = Field(min_length=1, max_length=20_000)
    title: str = Field(default="", max_length=200)
    tags: list[str] = Field(default_factory=list, max_length=10)


class NotesAppendArgs(ToolArgs):
    note_id: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=20_000)


class NotesDeleteArgs(ToolArgs):
    note_id: int = Field(ge=1)


class _NotesTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi notatek: wszystkie potrzebują działającej pamięci trwałej."""

    def __init__(self, spec: ToolSpec, memory: Any | None) -> None:
        super().__init__(spec)
        self._memory = memory

    def available(self) -> tuple[bool, str]:
        if self._memory is None:
            return False, "pamięć asystenta nie jest dostępna (MEMORY_ENABLED=false)"
        if not getattr(self._memory, "persistent", False):
            return False, f"pamięć trwała jest niedostępna ({getattr(self._memory, 'error', '')})"
        return True, ""

    @property
    def memory(self) -> Any:
        if self._memory is None:  # pragma: no cover - available() wyklucza tę ścieżkę
            raise ToolError(t("notes.no_memory"))
        return self._memory

    def _note_or_error(self, note_id: int) -> Any:
        note = self.memory.note(note_id)
        if note is None:
            raise ToolError(f"nie ma notatki o numerze {note_id}")
        return note


class NotesSearchTool(_NotesTool[NotesSearchArgs]):
    """``notes.search`` — notatki pasujące do frazy (albo ostatnie)."""

    async def run(self, args: NotesSearchArgs, ctx: ToolContext) -> ToolResult:
        query = args.query.strip()
        if query:
            hits = self.memory.search_notes(query, limit=args.limit)
            items = [
                {
                    "id": getattr(hit, "source_id", 0),
                    "title": getattr(hit, "title", ""),
                    "preview": getattr(hit, "preview", ""),
                }
                for hit in hits
            ]
        else:
            notes = self.memory.notes(limit=args.limit)
            items = [
                {"id": note.id, "title": note.title, "preview": note.preview} for note in notes
            ]
        return ToolResult.success(
            {"query": query, "count": len(items), "notes": items},
            display=f"{len(items)} notatek" + (f" dla '{query}'" if query else ""),
        )


class NotesReadTool(_NotesTool[NotesReadArgs]):
    """``notes.read`` — pełna treść notatki."""

    async def run(self, args: NotesReadArgs, ctx: ToolContext) -> ToolResult:
        note = self._note_or_error(args.note_id)
        return ToolResult.success(
            {
                "id": note.id,
                "title": note.title,
                "body": note.body,
                "tags": list(note.tags),
                "created": note.created_at.isoformat(timespec="seconds"),
            },
            display=f"notatka {note.id}: {note.title or note.preview}",
        )


class NotesCreateTool(_NotesTool[NotesCreateArgs]):
    """``notes.create`` — nowa notatka we własnych danych asystenta."""

    async def run(self, args: NotesCreateArgs, ctx: ToolContext) -> ToolResult:
        note = self.memory.add_note(args.body, title=args.title, tags=args.tags)
        if note is None:
            raise ToolError(t("notes.save_failed", error=getattr(self.memory, "error", "")))
        return ToolResult.success(
            {"id": note.id, "title": note.title, "chars": len(args.body)},
            display=t("notes.saved", id=note.id, title=note.title or note.preview),
        )

    async def preview(self, args: NotesCreateArgs, ctx: ToolContext) -> str:
        return t("notes.preview", title=args.title or args.body[:40], chars=len(args.body))


class NotesAppendTool(_NotesTool[NotesAppendArgs]):
    """``notes.append`` — dopisanie akapitu do istniejącej notatki."""

    async def run(self, args: NotesAppendArgs, ctx: ToolContext) -> ToolResult:
        self._note_or_error(args.note_id)
        note = self.memory.append_note(args.note_id, args.text)
        if note is None:
            raise ToolError(t("notes.append_failed", id=args.note_id))
        return ToolResult.success(
            {"id": note.id, "chars": len(note.body)},
            display=t("notes.appended", id=note.id, chars=len(note.body)),
        )

    async def preview(self, args: NotesAppendArgs, ctx: ToolContext) -> str:
        return f"dopisałoby {len(args.text)} znaków do notatki {args.note_id}"


class NotesDeleteTool(_NotesTool[NotesDeleteArgs]):
    """``notes.delete`` — usunięcie notatki razem z jej wektorem."""

    def confirmation(
        self, args: NotesDeleteArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        polish = language == "pl"
        note = None
        if self._memory is not None and getattr(self._memory, "persistent", False):
            note = self._memory.note(args.note_id)
        label = getattr(note, "title", "") or getattr(note, "preview", "") or str(args.note_id)
        summary = (
            f"Usunie notatkę {args.note_id}: {label}"
            if polish
            else f"Delete note {args.note_id}: {label}"
        )
        details = [
            ("operacja jest nieodwracalna" if polish else "this cannot be undone"),
        ]
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=RiskLevel.HIGH,
            summary=summary,
            details=details,
            preview=getattr(note, "body", "")[:300] if note is not None else None,
            language=language,
        )

    async def run(self, args: NotesDeleteArgs, ctx: ToolContext) -> ToolResult:
        note = self._note_or_error(args.note_id)
        label = note.title or note.preview
        if not self.memory.delete_note(args.note_id):
            raise ToolError(t("notes.delete_failed", id=args.note_id))
        return ToolResult.success(
            {"id": args.note_id, "title": note.title},
            display=t("notes.deleted", id=args.note_id, title=label),
        )

    async def preview(self, args: NotesDeleteArgs, ctx: ToolContext) -> str:
        return f"usunęłoby notatkę {args.note_id}"


def build_notes_tools(
    settings: Settings | None = None, *, memory: Any | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia notatek. Bez pamięci trwałej są niedostępne (model ich nie widzi)."""
    del settings  # notatki nie mają własnych ustawień — limity są w argumentach
    return (
        NotesSearchTool(
            ToolSpec(
                name="notes.search",
                description=(
                    "Search the assistant's own notes by keywords, or list the most recent "
                    "ones when query is empty."
                ),
                summary="szukanie w notatkach",
                args_model=NotesSearchArgs,
                risk=RiskLevel.SAFE,
                timeout_s=10.0,
            ),
            memory,
        ),
        NotesReadTool(
            ToolSpec(
                name="notes.read",
                description="Read one note by its id (use notes.search to find the id).",
                summary="odczyt notatki",
                args_model=NotesReadArgs,
                risk=RiskLevel.SAFE,
                timeout_s=10.0,
            ),
            memory,
        ),
        NotesCreateTool(
            ToolSpec(
                name="notes.create",
                description=(
                    "Store a new note in the assistant's memory. Use it when the user asks to "
                    "write something down."
                ),
                summary="nowa notatka",
                args_model=NotesCreateArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=10.0,
                idempotent=False,
            ),
            memory,
        ),
        NotesAppendTool(
            ToolSpec(
                name="notes.append",
                description="Append a paragraph to an existing note; nothing is overwritten.",
                summary="dopisanie do notatki",
                args_model=NotesAppendArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=10.0,
                idempotent=False,
            ),
            memory,
        ),
        NotesDeleteTool(
            ToolSpec(
                name="notes.delete",
                description=(
                    "Delete a note permanently, together with its semantic index entry. "
                    "Requires the user's confirmation."
                ),
                summary=t("spec.notes_delete"),
                args_model=NotesDeleteArgs,
                risk=RiskLevel.HIGH,
                timeout_s=10.0,
                idempotent=False,
            ),
            memory,
        ),
    )


__all__ = [
    "NotesAppendArgs",
    "NotesAppendTool",
    "NotesCreateArgs",
    "NotesCreateTool",
    "NotesDeleteArgs",
    "NotesDeleteTool",
    "NotesReadArgs",
    "NotesReadTool",
    "NotesSearchArgs",
    "NotesSearchTool",
    "build_notes_tools",
]
