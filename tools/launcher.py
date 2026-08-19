"""Narzędzia uruchamiania: aplikacje, adresy, procesy (Faza 8).

============== ======== ======================================================
narzędzie      poziom   uzasadnienie poziomu
============== ======== ======================================================
``app.list``   SAFE     odczyt listy zainstalowanych aplikacji
``process.list``SAFE    odczyt listy procesów (nazwa, właściciel, pamięć)
``app.launch`` MEDIUM   otwiera okno programu; nic nie niszczy, ale coś robi
``open.url``   MEDIUM   ruch sieciowy w przeglądarce użytkownika
``open.path``  MEDIUM   otwiera plik z dozwolonego katalogu skojarzonym programem
``process.kill``HIGH    zamknięcie programu może oznaczać utratę niezapisanej pracy
============== ======== ======================================================

Dlaczego ``app.launch`` to MEDIUM, a nie SAFE: uruchomienie programu nie zmienia
danych, ale wprowadza w system nowy proces, którego asystent już nie kontroluje.
Dlaczego nie HIGH: uruchamiamy **wyłącznie** aplikację z listy zainstalowanych
w systemie, nigdy dowolnej ścieżki ani polecenia — to jest „kliknij ikonę", a nie
„wykonaj cokolwiek". Do drugiego służy ``shell.run`` i ma poziom CRITICAL.

``open.url`` przyjmuje tylko schematy z ``LAUNCHER_ALLOWED_SCHEMES``
(domyślnie ``http``, ``https``, ``mailto``). ``file://`` świadomie nie jest
domyślnie: pliki otwiera ``open.path``, który pilnuje dozwolonych katalogów.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from config import Settings, get_settings
from host.apps import (
    Application,
    LaunchError,
    allowed_schemes,
    find_application,
    has_graphical_session,
    launch_application,
    list_applications,
    open_target,
    session_label,
    url_scheme,
)
from host.paths import PathNotAllowedError, Workspace
from host.privileges import refuse_if_privileged
from host.processes import (
    ProcessRefusedError,
    find_process,
    list_processes,
    terminate_process,
)
from host.processes import (
    available as processes_available,
)
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec
from i18n import t

logger = logging.getLogger(__name__)


class AppListArgs(ToolArgs):
    query: str = Field(default="", max_length=100)
    limit: int = Field(default=30, ge=1, le=200)


class AppLaunchArgs(ToolArgs):
    name: str = Field(min_length=1, max_length=120)


class OpenUrlArgs(ToolArgs):
    url: str = Field(min_length=3, max_length=2_000)


class OpenPathArgs(ToolArgs):
    path: str = Field(min_length=1, max_length=1_000)


class ProcessListArgs(ToolArgs):
    query: str = Field(default="", max_length=100)
    limit: int = Field(default=20, ge=1, le=200)


class ProcessKillArgs(ToolArgs):
    pid: int = Field(ge=2, le=4_194_304)
    force: bool = False


# --------------------------------------------------------------------------- #
# Aplikacje
# --------------------------------------------------------------------------- #


class _GraphicalTool[ArgsT: ToolArgs](BaseTool[ArgsT]):
    """Baza narzędzi wymagających sesji graficznej."""

    def available(self) -> tuple[bool, str]:
        if not has_graphical_session():
            return False, f"brak sesji graficznej ({session_label()})"
        return True, ""


class AppListTool(_GraphicalTool[AppListArgs]):
    """``app.list`` — co da się uruchomić na tej maszynie."""

    async def run(self, args: AppListArgs, ctx: ToolContext) -> ToolResult:
        applications = list_applications(limit=args.limit, query=args.query)
        names = [application.name for application in applications]
        return ToolResult.success(
            {
                "count": len(applications),
                "applications": [application.to_dict() for application in applications],
            },
            display=(
                f"{len(applications)} aplikacji: {', '.join(names[:10])}"
                + ("…" if len(names) > 10 else "")
            ),
        )


class AppLaunchTool(_GraphicalTool[AppLaunchArgs]):
    """``app.launch`` — uruchom aplikację z listy zainstalowanych."""

    def __init__(
        self, spec: ToolSpec, *, runner: Any | None = None, opener: Any | None = None
    ) -> None:
        super().__init__(spec)
        self._runner = runner
        self._opener = opener

    def _find(self, name: str) -> Application:
        application = find_application(name)
        if application is None:
            available_names = [item.name for item in list_applications(limit=15)]
            raise ToolError(
                t(
                    "app.not_found",
                    name=name,
                    examples=", ".join(available_names) or t("common.none"),
                )
            )
        return application

    async def run(self, args: AppLaunchArgs, ctx: ToolContext) -> ToolResult:
        application = self._find(args.name)
        try:
            note = launch_application(application, runner=self._runner, opener=self._opener)
        except LaunchError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success(
            {"application": application.name, "source": application.source},
            display=note,
        )

    async def preview(self, args: AppLaunchArgs, ctx: ToolContext) -> str:
        return f"uruchomiłoby aplikację '{args.name}'"


class OpenUrlTool(_GraphicalTool[OpenUrlArgs]):
    """``open.url`` — otwórz adres w domyślnej przeglądarce."""

    def __init__(
        self,
        spec: ToolSpec,
        *,
        settings: Settings | None = None,
        runner: Any | None = None,
        opener: Any | None = None,
    ) -> None:
        super().__init__(spec)
        self._settings = settings or get_settings()
        self._runner = runner
        self._opener = opener

    @property
    def schemes(self) -> tuple[str, ...]:
        return allowed_schemes(self._settings.launcher_allowed_schemes)

    async def run(self, args: OpenUrlArgs, ctx: ToolContext) -> ToolResult:
        scheme = url_scheme(args.url)
        if not scheme:
            raise ToolError(
                f"'{args.url}' nie ma schematu (http://, https://) — nie otwieram adresu bez niego"
            )
        if scheme not in self.schemes:
            raise ToolError(
                f"schemat '{scheme}' nie jest dozwolony (LAUNCHER_ALLOWED_SCHEMES: "
                f"{', '.join(self.schemes)})"
            )
        try:
            note = open_target(args.url, runner=self._runner, opener=self._opener)
        except LaunchError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success({"url": args.url, "scheme": scheme}, display=note)

    async def preview(self, args: OpenUrlArgs, ctx: ToolContext) -> str:
        return f"otworzyłoby adres {args.url}"


class OpenPathTool(_GraphicalTool[OpenPathArgs]):
    """``open.path`` — otwórz plik z dozwolonego katalogu skojarzonym programem."""

    def __init__(
        self,
        spec: ToolSpec,
        workspace: Workspace,
        *,
        runner: Any | None = None,
        opener: Any | None = None,
    ) -> None:
        super().__init__(spec)
        self._workspace = workspace
        self._runner = runner
        self._opener = opener

    def available(self) -> tuple[bool, str]:
        usable, reason = super().available()
        if not usable:
            return usable, reason
        if not self._workspace.roots:  # pragma: no cover - konfiguracja daje katalog
            return False, "nie skonfigurowano dozwolonych katalogów"
        return True, ""

    async def run(self, args: OpenPathArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = self._workspace.resolve(args.path, must_exist=True)
        except PathNotAllowedError as exc:
            raise ToolError(exc.message) from exc
        try:
            note = open_target(str(path), runner=self._runner, opener=self._opener)
        except LaunchError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success(
            {"path": self._workspace.label(path)},
            display=note,
        )

    async def preview(self, args: OpenPathArgs, ctx: ToolContext) -> str:
        return f"otworzyłoby {args.path} programem domyślnym"


# --------------------------------------------------------------------------- #
# Procesy
# --------------------------------------------------------------------------- #


class ProcessListTool(BaseTool[ProcessListArgs]):
    """``process.list`` — co działa na tej maszynie."""

    def available(self) -> tuple[bool, str]:
        return processes_available()

    async def run(self, args: ProcessListArgs, ctx: ToolContext) -> ToolResult:
        try:
            processes = list_processes(limit=args.limit, query=args.query)
        except ProcessRefusedError as exc:
            raise ToolError(exc.message) from exc
        top = ", ".join(f"{item.name} ({item.pid})" for item in processes[:5])
        return ToolResult.success(
            {
                "count": len(processes),
                "processes": [item.to_dict() for item in processes],
            },
            display=t("app.process_summary", count=len(processes), largest=top),
        )


class ProcessKillTool(BaseTool[ProcessKillArgs]):
    """``process.kill`` — zamknij program. Zawsze wymaga potwierdzenia."""

    def __init__(self, spec: ToolSpec, *, killer: Any | None = None) -> None:
        super().__init__(spec)
        self._killer = killer

    def available(self) -> tuple[bool, str]:
        usable, reason = processes_available()
        if not usable:
            return usable, reason
        # Na koncie administratora zamknięcie „czegokolwiek" przestaje być
        # ograniczone do własnych programów — wtedy narzędzia nie ma.
        refusal = refuse_if_privileged("Zamykanie procesów", strict=False)
        if refusal:
            return False, refusal
        return True, ""

    def confirmation(
        self, args: ProcessKillArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        polish = language == "pl"
        details: list[str] = []
        name = ""
        try:
            process = find_process(args.pid)
        except ProcessRefusedError:  # pragma: no cover - brak backendu wyklucza narzędzie
            process = None
        if process is not None:
            name = process.name
            details = [
                f"PID {process.pid}",
                f"{'właściciel' if polish else 'owner'}: {process.username or '?'}",
                f"{'pamięć' if polish else 'memory'}: {process.memory_mb:.0f} MB",
            ]
        summary = (
            f"Zamknie program {name or '?'} (PID {args.pid})"
            + (" — WYMUSZONE, bez zapisu danych" if args.force and polish else "")
            if polish
            else f"Close program {name or '?'} (PID {args.pid})"
            + (" — FORCED, no chance to save" if args.force else "")
        )
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=self.effective_risk(args),
            summary=summary,
            details=details,
            language=language,
        )

    async def run(self, args: ProcessKillArgs, ctx: ToolContext) -> ToolResult:
        try:
            note = terminate_process(args.pid, force=args.force, killer=self._killer)
        except ProcessRefusedError as exc:
            raise ToolError(exc.message) from exc
        return ToolResult.success(
            {"pid": args.pid, "forced": args.force}, display=note
        )

    async def preview(self, args: ProcessKillArgs, ctx: ToolContext) -> str:
        return f"zamknęłoby proces {args.pid}" + (" (wymuszone)" if args.force else "")


# --------------------------------------------------------------------------- #
# Rejestracja
# --------------------------------------------------------------------------- #


def build_launcher_tools(
    settings: Settings | None = None,
    *,
    workspace: Workspace | None = None,
    runner: Any | None = None,
    opener: Any | None = None,
    killer: Any | None = None,
) -> Sequence[Tool[Any]]:
    """Narzędzia uruchamiania. Atrapy (``runner``/``opener``/``killer``) są dla testów."""
    active = settings or get_settings()
    area = workspace or Workspace.from_settings(active)

    return (
        AppListTool(
            ToolSpec(
                name="app.list",
                description=(
                    "List applications installed on this computer. Use it before app.launch "
                    "to check the exact name."
                ),
                summary="lista zainstalowanych aplikacji",
                args_model=AppListArgs,
                risk=RiskLevel.SAFE,
                timeout_s=15.0,
            )
        ),
        AppLaunchTool(
            ToolSpec(
                name="app.launch",
                description=(
                    "Start an installed application by name. Only applications present in the "
                    "system application list can be started — never an arbitrary path."
                ),
                summary="uruchomienie zainstalowanej aplikacji",
                args_model=AppLaunchArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=15.0,
                idempotent=False,
            ),
            runner=runner,
            opener=opener,
        ),
        OpenUrlTool(
            ToolSpec(
                name="open.url",
                description=(
                    "Open a web address in the user's default browser. Only http, https and "
                    "mailto schemes are allowed by default."
                ),
                summary=t("spec.open_url"),
                args_model=OpenUrlArgs,
                risk=RiskLevel.MEDIUM,
                requires_network=True,
                timeout_s=15.0,
                idempotent=False,
            ),
            settings=active,
            runner=runner,
            opener=opener,
        ),
        OpenPathTool(
            ToolSpec(
                name="open.path",
                description=(
                    "Open a file from an allowed directory with the program the system "
                    "associates with it."
                ),
                summary=t("spec.open_path"),
                args_model=OpenPathArgs,
                risk=RiskLevel.MEDIUM,
                timeout_s=15.0,
                idempotent=False,
            ),
            area,
            runner=runner,
            opener=opener,
        ),
        ProcessListTool(
            ToolSpec(
                name="process.list",
                description=(
                    "List running processes with their PID, owner and memory usage. "
                    "Read-only."
                ),
                summary=t("spec.proc_list"),
                args_model=ProcessListArgs,
                risk=RiskLevel.SAFE,
                timeout_s=15.0,
            )
        ),
        ProcessKillTool(
            ToolSpec(
                name="process.kill",
                description=(
                    "Ask a process to close (or force it with force=true). Only processes "
                    "owned by the current user; system processes are refused. Requires the "
                    "user's confirmation."
                ),
                summary=t("spec.proc_kill"),
                args_model=ProcessKillArgs,
                risk=RiskLevel.HIGH,
                timeout_s=20.0,
                idempotent=False,
            ),
            killer=killer,
        ),
    )


__all__ = [
    "AppLaunchArgs",
    "AppLaunchTool",
    "AppListArgs",
    "AppListTool",
    "OpenPathArgs",
    "OpenPathTool",
    "OpenUrlArgs",
    "OpenUrlTool",
    "ProcessKillArgs",
    "ProcessKillTool",
    "ProcessListArgs",
    "ProcessListTool",
    "build_launcher_tools",
]
