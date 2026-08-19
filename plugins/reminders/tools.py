"""Narzędzia przypomnień: zaplanuj, pokaż, odwołaj (Faza 11).

=================== ======== =================================================
narzędzie           poziom   uzasadnienie
=================== ======== =================================================
``reminders.add``   SAFE     samo zaplanowanie niczego nie zmienia w świecie
``reminders.add``   MEDIUM   …ale jeśli ma po terminie URUCHOMIĆ inną akcję,
                             to jest już zmiana i ryzyko rośnie (eskalacja
                             z argumentów, patrz ``dynamic_risk``)
``reminders.list``  SAFE     odczyt własnych danych asystenta
``reminders.cancel``MEDIUM   kasuje plan użytkownika — odwracalne, ale zmienia
=================== ======== =================================================

Rozpoznawaniem „za 20 minut" z mowy zajmuje się model — on tłumaczy to na
``in_minutes=20``. Narzędzie przyjmuje wyłącznie postać jednoznaczną, bo
zgadywanie intencji z tekstu w kodzie narzędzia kończy się budzeniem o złej
porze. Godzina („07:00") jest liczona w strefie czasowej TEJ maszyny i wypada
najbliższego dnia, w którym jeszcze nie minęła — czyli „obudź mnie o 7"
powiedziane wieczorem znaczy jutro rano.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from pydantic import Field, model_validator

from plugins.reminders.storage import Reminder, ReminderError, ReminderStore
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from i18n import t

logger = logging.getLogger(__name__)

# Sama godzina („7:00", „07:30”) — najczęstsza postać przy budziku.
_TIME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$")
# Data z godziną, bez strefy („2026-08-20 07:00"). Traktowana jako czas lokalny.
_LOCAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{1,2}:\d{2})$"
)

# Górna granica planowania. Rok wystarcza na wszystko, co ma sens dla asystenta
# domowego, a odcina „przypomnij mi za 900000 minut" wynikające z pomyłki modelu.
MAX_HORIZON_DAYS: Final[int] = 365


def _local_now(now: datetime) -> datetime:
    """Ten sam moment w strefie czasowej maszyny."""
    return now.astimezone()


def resolve_due(*, now: datetime, in_minutes: int | None = None, at: str = "") -> datetime:
    """Zamień argumenty modelu na konkretny moment w UTC.

    Zawsze jedno z dwojga: liczba minut albo godzina. Podanie obu naraz jest
    błędem argumentów, a nie okazją do zgadywania, co użytkownik miał na myśli.
    """
    if in_minutes is not None and at.strip():
        raise ToolError("podaj albo in_minutes, albo at — nie oba naraz")

    if in_minutes is not None:
        if in_minutes <= 0:
            raise ToolError(t("rem.minutes_positive"))
        return (now + timedelta(minutes=in_minutes)).astimezone(UTC)

    text = at.strip()
    if not text:
        raise ToolError(t("rem.need_time"))

    local_now = _local_now(now)

    match = _TIME_PATTERN.match(text)
    if match is not None:
        hour, minute = int(match.group("hour")), int(match.group("minute"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ToolError(f"godzina {text!r} nie istnieje")
        target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= local_now:
            # Godzina, która dziś już minęła, znaczy „jutro o tej porze".
            target += timedelta(days=1)
        return target.astimezone(UTC)

    match = _LOCAL_PATTERN.match(text)
    if match is not None:
        try:
            naive = datetime.fromisoformat(f"{match.group('date')} {match.group('time')}")
        except ValueError as exc:
            raise ToolError(f"nie rozumiem terminu {text!r}") from exc
        # Data bez strefy = czas lokalny użytkownika, nie UTC. Odwrotne
        # założenie przesuwałoby budzik o strefę czasową.
        return naive.replace(tzinfo=local_now.tzinfo).astimezone(UTC)

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ToolError(
            t("rem.bad_time", value=repr(text))
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_now.tzinfo)
    return parsed.astimezone(UTC)


class ReminderAddArgs(ToolArgs):
    text: str = Field(min_length=1, max_length=500)
    in_minutes: int | None = Field(default=None, ge=1, le=MAX_HORIZON_DAYS * 24 * 60)
    at: str = Field(default="", max_length=40)
    # Nazwa narzędzia do wykonania po terminie. Puste = samo przypomnienie.
    # Wypełnione = przypomnienie ma coś ZROBIĆ, więc ryzyko rośnie do MEDIUM.
    action: str = Field(default="", max_length=60)

    @model_validator(mode="after")
    def _one_of(self) -> ReminderAddArgs:
        if self.in_minutes is None and not self.at.strip():
            raise ValueError("podaj in_minutes albo at")
        return self


class ReminderListArgs(ToolArgs):
    limit: int = Field(default=20, ge=1, le=100)
    # „active" (domyślnie) albo „all" — historia bywa potrzebna do pytania
    # „co mi dzwoniło rano?".
    scope: str = Field(default="active", max_length=10)


class ReminderCancelArgs(ToolArgs):
    reminder_id: int = Field(ge=1)


class _ReminderTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi przypomnień: wszystkie potrzebują działającego magazynu."""

    def __init__(self, spec: ToolSpec, store: ReminderStore | None) -> None:
        super().__init__(spec)
        self._store = store

    def available(self) -> tuple[bool, str]:
        if self._store is None:
            return False, "przypomnienia wymagają pamięci trwałej (MEMORY_ENABLED=true)"
        return True, ""

    def store(self) -> ReminderStore:
        if self._store is None:  # pragma: no cover - bramka ENABLED sprawdza to wcześniej
            raise ToolError(t("rem.no_storage"))
        return self._store


class ReminderAddTool(_ReminderTool[ReminderAddArgs]):
    """Zaplanuj przypomnienie."""

    def __init__(self, spec: ToolSpec, store: ReminderStore | None, *, max_active: int) -> None:
        super().__init__(spec, store)
        self._max_active = max_active

    def dynamic_risk(self, args: ReminderAddArgs) -> RiskLevel:
        """Samo zaplanowanie to SAFE; zaplanowanie AKCJI to już MEDIUM.

        Różnica jest realna: „przypomnij mi o praniu" nic nie robi poza
        odezwaniem się, a „za godzinę wyłącz światło w salonie" wykona po
        terminie czynność, której użytkownik w tamtej chwili może nie pilnować.
        """
        return RiskLevel.MEDIUM if args.action.strip() else RiskLevel.SAFE

    async def run(self, args: ReminderAddArgs, ctx: ToolContext) -> ToolResult:
        store = self.store()
        due = resolve_due(now=ctx.now(), in_minutes=args.in_minutes, at=args.at)

        horizon = ctx.now() + timedelta(days=MAX_HORIZON_DAYS)
        if due > horizon:
            raise ToolError(t("rem.too_far", days=MAX_HORIZON_DAYS))

        try:
            if store.count_active() >= self._max_active:
                raise ToolError(
                    t("rem.too_many", limit=self._max_active)
                )
            saved = store.add(args.text, due, action=args.action, source="model", now=ctx.now())
        except ReminderError as exc:
            raise ToolError(str(exc)) from exc

        local = saved.local_due()
        return ToolResult.success(
            {
                "id": saved.id,
                "text": saved.text,
                "due_at": saved.due_at.isoformat(),
                "due_local": local.isoformat(),
                "action": saved.action,
            },
            display=t("rem.scheduled", text=saved.text, when=local.strftime("%Y-%m-%d %H:%M")),
        )

    def confirmation(
        self, args: ReminderAddArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        if not args.action.strip():
            return None  # SAFE — router i tak nie zapyta
        summary = (
            f"Zaplanuje wykonanie {args.action} po terminie"
            if language == "pl"
            else f"Will schedule {args.action} to run at the given time"
        )
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=RiskLevel.MEDIUM,
            summary=summary,
            details=[args.text, args.at or f"za {args.in_minutes} min"],
            language=language,
        )


class ReminderListTool(_ReminderTool[ReminderListArgs]):
    """Pokaż zaplanowane przypomnienia."""

    async def run(self, args: ReminderListArgs, ctx: ToolContext) -> ToolResult:
        store = self.store()
        wanted_all = args.scope.strip().lower() in ("all", "wszystkie", "history", "historia")
        items = store.recent(limit=args.limit) if wanted_all else store.active(limit=args.limit)
        now = ctx.now()

        if not items:
            return ToolResult.success({"reminders": []}, display=t("rem.none"))

        return ToolResult.success(
            {
                "reminders": [
                    {
                        "id": item.id,
                        "text": item.text,
                        "due_local": item.local_due().isoformat(),
                        "state": item.state,
                    }
                    for item in items
                ]
            },
            display="\n".join(item.describe(now=now) for item in items),
        )


class ReminderCancelTool(_ReminderTool[ReminderCancelArgs]):
    """Odwołaj zaplanowane przypomnienie."""

    async def run(self, args: ReminderCancelArgs, ctx: ToolContext) -> ToolResult:
        store = self.store()
        cancelled: Reminder | None = store.cancel(args.reminder_id)
        if cancelled is None:
            raise ToolError(
                t("rem.not_found", id=args.reminder_id)
            )
        return ToolResult.success(
            {"id": cancelled.id, "text": cancelled.text},
            display=t("rem.cancelled", description=cancelled.describe()),
        )


def build_reminder_tools(store: ReminderStore | None, *, max_active: int = 100) -> list[Tool[Any]]:
    """Narzędzia pluginu przypomnień. Opisy są po angielsku — czyta je model."""
    return [
        ReminderAddTool(
            ToolSpec(
                name="reminders.add",
                description=(
                    "Schedule a reminder. Give either in_minutes (relative, e.g. 20) or "
                    "at (absolute: 'HH:MM' for the next occurrence of that time, or "
                    "'YYYY-MM-DD HH:MM'). Times are the user's local time. Set action only "
                    "when the reminder should trigger another tool."
                ),
                args_model=ReminderAddArgs,
                risk=RiskLevel.SAFE,
                summary="Zaplanuj przypomnienie albo budzik.",
            ),
            store,
            max_active=max_active,
        ),
        ReminderListTool(
            ToolSpec(
                name="reminders.list",
                description=(
                    "List reminders. scope='active' (default) shows what is still "
                    "scheduled, scope='all' includes ones that already fired."
                ),
                args_model=ReminderListArgs,
                risk=RiskLevel.SAFE,
                summary=t("spec.rem_list"),
            ),
            store,
        ),
        ReminderCancelTool(
            ToolSpec(
                name="reminders.cancel",
                description="Cancel a scheduled reminder by its id (see reminders.list).",
                args_model=ReminderCancelArgs,
                risk=RiskLevel.MEDIUM,
                summary=t("spec.rem_cancel"),
            ),
            store,
        ),
    ]


__all__ = [
    "MAX_HORIZON_DAYS",
    "ReminderAddArgs",
    "ReminderAddTool",
    "ReminderCancelArgs",
    "ReminderCancelTool",
    "ReminderListArgs",
    "ReminderListTool",
    "build_reminder_tools",
    "resolve_due",
]
