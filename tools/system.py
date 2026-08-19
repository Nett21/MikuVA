"""Narzędzia o zerowym ryzyku: informacje, które można tylko przeczytać (Faza 7).

Na razie jedno: ``time.now``. Wystarczy, żeby przejść cały przepływ — model pyta,
router waliduje, narzędzie odpowiada, model formułuje zdanie — a nic w systemie
się przy tym nie zmienia. Odczyt czasu nie ma efektów ubocznych i nie sięga do
sieci, więc jest SAFE i nie wymaga potwierdzenia.

Dwie rzeczy są tu świadomie zrobione „na piechotę", i to nie z lenistwa:

* **nazwy dni i miesięcy są wpisane w kod**, a nie brane z ``locale``. Formatowanie
  przez ``%A``/``%B`` dałoby wynik zależny od ustawień regionalnych maszyny — ten
  sam asystent mówiłby „Tuesday" na komputerze bez polskich locale i „wtorek" na
  innym. Język odpowiedzi ma zależeć od języka rozmowy, nie od systemu;
* **strefa czasowa to tylko „local" albo „utc"**, bez nazw IANA („Europe/Warsaw").
  ``zoneinfo`` czyta bazę stref z systemu, a Windows jej nie ma — trzeba by
  dokładać pakiet ``tzdata``. Strefa lokalna maszyny jest dostępna wszędzie i
  odpowiada na pytanie „która godzina?", które faktycznie zadaje użytkownik.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

from pydantic import Field

from config import Settings
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import (
    BaseTool,
    Tool,
    ToolArgs,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    make_tool,
)

# Nazwy własne, nie zależne od locale maszyny (patrz docstring modułu).
_WEEKDAYS_PL: Final[tuple[str, ...]] = (
    "poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela",
)
_WEEKDAYS_EN: Final[tuple[str, ...]] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
)
_MONTHS_PL: Final[tuple[str, ...]] = (
    "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
)
_MONTHS_EN: Final[tuple[str, ...]] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


class TimeNowArgs(ToolArgs):
    """Argumenty ``time.now``."""

    zone: Literal["local", "utc"] = "local"
    include_date: bool = True


def _zone_label(moment: datetime) -> str:
    """Nazwa strefy albo przesunięcie względem UTC.

    ``tzname()`` bywa puste (zależy od systemu i biblioteki C), więc jest tylko
    pierwszym wyborem — drugim jest przesunięcie, które da się policzyć wszędzie.
    """
    name = moment.tzname()
    if name:
        return name
    offset = moment.utcoffset()
    if offset is None:
        return "UTC"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def format_moment(moment: datetime, *, language: str, include_date: bool) -> str:
    """Data i godzina zdaniem, które da się przeczytać na głos."""
    polish = language == "pl"
    clock = moment.strftime("%H:%M")
    if not include_date:
        return f"{clock} ({_zone_label(moment)})"

    weekday = (_WEEKDAYS_PL if polish else _WEEKDAYS_EN)[moment.weekday()]
    month = (_MONTHS_PL if polish else _MONTHS_EN)[moment.month - 1]
    return f"{weekday}, {moment.day} {month} {moment.year}, {clock} ({_zone_label(moment)})"


def time_now(args: TimeNowArgs, ctx: ToolContext) -> ToolResult:
    """Podaj aktualną datę i godzinę.

    Czas bierzemy z ``ctx.now`` — zegara wstrzykniętego przez router. To nie jest
    nadmierna ostrożność: dzięki temu testy sprawdzają formatowanie na ustalonej
    chwili, a nie na tym, co akurat pokazuje zegar maszyny CI.
    """
    moment = ctx.now()
    if moment.tzinfo is None:  # pragma: no cover - zegar bez strefy to błąd wywołującego
        moment = moment.replace(tzinfo=UTC)
    local = moment.astimezone() if args.zone == "local" else moment.astimezone(UTC)

    display = format_moment(local, language=ctx.language, include_date=args.include_date)
    offset = local.utcoffset() or timedelta(0)
    return ToolResult.success(
        {
            "iso": local.isoformat(timespec="seconds"),
            "date": local.date().isoformat(),
            "time": local.strftime("%H:%M"),
            "weekday": (_WEEKDAYS_PL if ctx.language == "pl" else _WEEKDAYS_EN)[local.weekday()],
            "zone": _zone_label(local),
            "utc_offset_minutes": int(offset.total_seconds() // 60),
        },
        display=display,
    )


def build_time_tool() -> Tool[Any]:
    return make_tool(
        name="time.now",
        description=(
            "Return the current local date and time of the user's computer. "
            "Use it whenever the answer depends on what time or day it is now. "
            "Set zone='utc' for UTC instead of local time."
        ),
        summary="aktualna data i godzina (tylko odczyt)",
        args_model=TimeNowArgs,
        risk=RiskLevel.SAFE,
        function=time_now,
        timeout_s=5.0,
        requires_network=False,
        idempotent=True,
    )


# --------------------------------------------------------------------------- #
# Informacje o systemie (SAFE) — Faza 8
# --------------------------------------------------------------------------- #


class SystemInfoArgs(ToolArgs):
    """``system.info`` nie potrzebuje argumentów."""


def system_info(args: SystemInfoArgs, ctx: ToolContext) -> ToolResult:
    """Opis maszyny: system, procesor, sesja graficzna, dozwolone katalogi.

    Wszystko pochodzi z detekcji w ``config.py`` i z warstwy ``host/`` — ten kod
    nie pyta samodzielnie o ``sys.platform``. Świadomie NIE podajemy nazwy
    użytkownika ani nazwy komputera: model ich nie potrzebuje, a trafiałyby do
    historii rozmowy i do promptu.
    """
    from config import detect_platform
    from host.apps import session_label
    from host.paths import Workspace
    from host.privileges import account_label
    from host.processes import describe_backend as processes_backend
    from host.shell import system_shell

    info = detect_platform()
    workspace = Workspace.from_settings(ctx.settings)
    shell = system_shell(info)
    data = {
        "os": info.os_label,
        "os_family": str(info.os_family),
        "machine": info.machine,
        "cpu_count": info.cpu_count,
        "python": info.python_version,
        "session": session_label(info),
        "account": account_label(info),
        "shell": shell.name if shell is not None else "",
        "processes_backend": processes_backend(info),
        "allowed_directories": [str(root) for root in workspace.roots],
        "wsl": info.is_wsl,
    }
    display = (
        f"{info.os_label} ({info.machine}), {info.cpu_count} rdzeni, "
        f"sesja: {session_label(info)}, {account_label(info)}"
    )
    return ToolResult.success(data, display=display)


def build_system_info_tool() -> Tool[Any]:
    return make_tool(
        name="system.info",
        description=(
            "Describe this computer: operating system, CPU, graphical session, shell and "
            "which directories the file tools may use. Read-only."
        ),
        summary="informacje o systemie (tylko odczyt)",
        args_model=SystemInfoArgs,
        risk=RiskLevel.SAFE,
        function=system_info,
        timeout_s=10.0,
    )


# --------------------------------------------------------------------------- #
# Usługi użytkownika (SAFE / HIGH) — Faza 8
# --------------------------------------------------------------------------- #


class ServiceListArgs(ToolArgs):
    limit: int = Field(default=20, ge=1, le=100)


class ServiceStatusArgs(ToolArgs):
    unit: str = Field(min_length=1, max_length=120)


class ServiceControlArgs(ToolArgs):
    unit: str = Field(min_length=1, max_length=120)
    action: Literal["start", "stop", "restart"]


class _ServiceTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi usług: dostępne tylko tam, gdzie da się ich dotknąć bez roota."""

    def __init__(self, spec: ToolSpec, *, runner: Any | None = None) -> None:
        super().__init__(spec)
        self._runner = runner

    def available(self) -> tuple[bool, str]:
        from host.services import available as services_available

        return services_available()


class ServiceListTool(_ServiceTool[ServiceListArgs]):
    """``service.list`` — usługi użytkownika i ich stan."""

    async def run(self, args: ServiceListArgs, ctx: ToolContext) -> ToolResult:
        from host.services import ServiceRefusedError, list_services

        try:
            services = list_services(limit=args.limit, runner=self._runner)
        except ServiceRefusedError as exc:
            raise ToolError(exc.message) from exc
        active = [item.unit for item in services if item.active == "active"]
        return ToolResult.success(
            {"count": len(services), "services": [item.to_dict() for item in services]},
            display=f"{len(services)} usług użytkownika, aktywnych: {len(active)}",
        )


class ServiceStatusTool(_ServiceTool[ServiceStatusArgs]):
    """``service.status`` — stan jednej usługi użytkownika."""

    async def run(self, args: ServiceStatusArgs, ctx: ToolContext) -> ToolResult:
        from host.services import ServiceRefusedError, service_status

        try:
            service = service_status(args.unit, runner=self._runner)
        except ServiceRefusedError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success(
            service.to_dict(),
            display=f"{service.unit}: {service.active} ({service.sub})",
        )


class ServiceControlTool(_ServiceTool[ServiceControlArgs]):
    """``service.control`` — start/stop/restart usługi UŻYTKOWNIKA. Wymaga zgody."""

    def confirmation(
        self, args: ServiceControlArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        polish = language == "pl"
        summary = (
            f"Wykona '{args.action}' na usłudze użytkownika {args.unit}"
            if polish
            else f"Run '{args.action}' on user service {args.unit}"
        )
        details = [
            "systemctl --user " + f"{args.action} {args.unit}",
            (
                "usługi systemowe i sudo są niedostępne"
                if polish
                else "system-wide services and sudo are unavailable"
            ),
        ]
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=RiskLevel.HIGH,
            summary=summary,
            details=details,
            language=language,
        )

    async def run(self, args: ServiceControlArgs, ctx: ToolContext) -> ToolResult:
        from host.services import ServiceRefusedError, control_service

        try:
            note = control_service(args.unit, args.action, runner=self._runner)
        except ServiceRefusedError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success(
            {"unit": args.unit, "action": args.action}, display=note
        )

    async def preview(self, args: ServiceControlArgs, ctx: ToolContext) -> str:
        return f"wykonałoby systemctl --user {args.action} {args.unit}"


def build_service_tools(*, runner: Any | None = None) -> Sequence[Tool[Any]]:
    """Narzędzia usług użytkownika (``runner`` podstawia atrapę w testach)."""
    return (
        ServiceListTool(
            ToolSpec(
                name="service.list",
                description=(
                    "List the user's own background services and whether they are running "
                    "(systemctl --user). Read-only."
                ),
                summary="usługi użytkownika (tylko odczyt)",
                args_model=ServiceListArgs,
                risk=RiskLevel.SAFE,
                timeout_s=20.0,
            ),
            runner=runner,
        ),
        ServiceStatusTool(
            ToolSpec(
                name="service.status",
                description="Check the state of one user service by unit name. Read-only.",
                summary="stan usługi użytkownika",
                args_model=ServiceStatusArgs,
                risk=RiskLevel.SAFE,
                timeout_s=20.0,
            ),
            runner=runner,
        ),
        ServiceControlTool(
            ToolSpec(
                name="service.control",
                description=(
                    "Start, stop or restart one of the user's own services. Never system-wide "
                    "services, never with sudo. Requires the user's confirmation."
                ),
                summary="start/stop/restart usługi użytkownika (wymaga zgody)",
                args_model=ServiceControlArgs,
                risk=RiskLevel.HIGH,
                timeout_s=30.0,
                idempotent=False,
            ),
            runner=runner,
        ),
    )


def build_system_tools(
    settings: Settings | None = None, *, service_runner: Any | None = None
) -> Sequence[Tool[Any]]:
    """Narzędzia informacyjne i usług: czas, opis maszyny, usługi użytkownika."""
    del settings  # limity tych narzędzi są w argumentach, nie w ustawieniach
    return (
        build_time_tool(),
        build_system_info_tool(),
        *build_service_tools(runner=service_runner),
    )


__all__ = [
    "ServiceControlArgs",
    "ServiceControlTool",
    "ServiceListArgs",
    "ServiceListTool",
    "ServiceStatusArgs",
    "ServiceStatusTool",
    "SystemInfoArgs",
    "TimeNowArgs",
    "build_service_tools",
    "build_system_info_tool",
    "build_system_tools",
    "build_time_tool",
    "format_moment",
    "system_info",
    "time_now",
]
