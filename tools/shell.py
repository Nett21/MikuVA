"""Narzędzie ``shell.run`` — poziom CRITICAL, domyślnie wyłączone (Faza 8).

Osobny plik, bo to jedyne narzędzie, które uruchamia dowolny wskazany program.
Wszystkie rygory są w ``host/shell.py`` (argv bez powłoki, allowlista programów,
twarde blokady, czyste środowisko, brak podnoszenia uprawnień). Tutaj zostaje
opakowanie w kontrakt narzędzia:

* **poziom CRITICAL** — więc wymaga potwierdzenia z pełną frazą, a dodatkowo
  ``SECURITY_ALLOW_CRITICAL=true``. Bez tego drugiego narzędzie nie jest nawet
  pokazywane modelowi,
* **domyślnie niedostępne** — ``SHELL_ALLOWED_BINARIES`` jest pusta, więc na
  świeżej instalacji ``shell.run`` nie istnieje z punktu widzenia modelu,
* **nigdy na koncie root/administratora** — wtedy narzędzie jest niedostępne,
* **potwierdzenie pokazuje pełne argv i katalog roboczy**, bo to jedyne
  narzędzie, przy którym „co dokładnie się wykona" nie jest oczywiste.

Czego tu nie ma i nie będzie: pojedynczego łańcucha z poleceniem, potoków,
przekierowań i flag ``-c``/``-Command``. Uzasadnienie jest w ``host/shell.py`` —
bez kontroli nad argv nie da się utrzymać blokad w rodzaju „nigdy ``rm -rf``".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from config import Settings, get_settings
from host.paths import PathNotAllowedError, Workspace
from host.privileges import refuse_if_privileged
from host.shell import (
    CommandBlockedError,
    ShellPolicy,
    check_arguments,
    environment_names,
    resolve_binary,
    run_command,
    system_shell,
)
from security.confirm import ConfirmationRequest
from security.risk import RiskLevel
from tools.base import BaseTool, Tool, ToolArgs, ToolContext, ToolError, ToolResult, ToolSpec

logger = logging.getLogger(__name__)


class ShellRunArgs(ToolArgs):
    """Argumenty ``shell.run``.

    ``argv`` to lista: pierwszy element to NAZWA programu (nie ścieżka), reszta to
    jego argumenty. Nie ma tu pola „command" z całym poleceniem — i to jest
    najważniejsza cecha tego modelu argumentów.
    """

    argv: list[str] = Field(min_length=1, max_length=32)
    cwd: str = Field(default="", max_length=1_000)


class ShellRunTool(BaseTool[ShellRunArgs]):
    """Uruchomienie jednego programu z listy dozwolonych."""

    def __init__(
        self,
        spec: ToolSpec,
        *,
        settings: Settings | None = None,
        workspace: Workspace | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(spec)
        self._settings = settings or get_settings()
        self._workspace = workspace or Workspace.from_settings(self._settings)
        self._runner = runner

    @property
    def policy(self) -> ShellPolicy:
        return ShellPolicy.from_settings(self._settings)

    def available(self) -> tuple[bool, str]:
        policy = self.policy
        if not policy.enabled:
            return False, (
                "uruchamianie programów jest wyłączone — lista SHELL_ALLOWED_BINARIES "
                "jest pusta (to domyślny stan)"
            )
        refusal = refuse_if_privileged("Uruchamianie programów", strict=True)
        if refusal:
            return False, refusal
        return True, ""

    def _cwd(self, raw: str) -> Path | None:
        if not raw.strip():
            return self._workspace.primary if self._workspace.primary.is_dir() else None
        try:
            return self._workspace.resolve(raw, must_exist=True, must_be_dir=True)
        except PathNotAllowedError as exc:
            raise ToolError(exc.message) from exc

    def confirmation(
        self, args: ShellRunArgs, *, language: str = "en"
    ) -> ConfirmationRequest | None:
        """Pytanie pokazuje DOKŁADNIE to, co się wykona — pełne argv i katalog."""
        polish = language == "pl"
        argv = [str(item) for item in args.argv]
        summary = (
            f"URUCHOMI program: {' '.join(argv)}"
            if polish
            else f"RUN a program: {' '.join(argv)}"
        )
        details = [
            f"{'program' if polish else 'binary'}: {argv[0]}",
            f"{'argumenty' if polish else 'arguments'}: {argv[1:] or '(brak)'}",
            f"{'katalog' if polish else 'directory'}: {args.cwd or self._workspace.primary}",
            (
                "środowisko: tylko " + ", ".join(environment_names())
                if polish
                else "environment: only " + ", ".join(environment_names())
            ),
        ]
        return ConfirmationRequest.build(
            tool=self.spec.name,
            risk=RiskLevel.CRITICAL,
            summary=summary,
            details=details,
            language=language,
        )

    async def run(self, args: ShellRunArgs, ctx: ToolContext) -> ToolResult:
        policy = self.policy
        cwd = self._cwd(args.cwd)
        try:
            result = run_command(
                [str(item) for item in args.argv],
                policy,
                cwd=cwd,
                runner=self._runner,
            )
        except CommandBlockedError as exc:
            raise ToolError(exc.message) from exc

        return ToolResult.success(
            {
                "argv": list(result.argv),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "truncated": result.truncated,
                "duration_ms": result.duration_ms,
            },
            display=result.describe(),
            # Wyjście programu to treść z zewnątrz — po niej kolejne wywołanie
            # o ryzyku MEDIUM+ wymaga potwierdzenia (bariera przeciw wstrzyknięciu).
            untrusted=True,
        )

    async def preview(self, args: ShellRunArgs, ctx: ToolContext) -> str:
        argv = [str(item) for item in args.argv]
        try:
            check_arguments(argv)
            binary = resolve_binary(argv[0], self.policy)
        except CommandBlockedError as exc:
            return f"odmowa: {exc.message}"
        where = args.cwd or self._workspace.primary
        return f"uruchomiłoby {binary} z argumentami {argv[1:]} w {where}"


def build_shell_tools(
    settings: Settings | None = None,
    *,
    workspace: Workspace | None = None,
    runner: Any | None = None,
) -> Sequence[Tool[Any]]:
    """Narzędzie powłoki. Zawsze CRITICAL, domyślnie niedostępne."""
    active = settings or get_settings()
    shell = system_shell()
    hint = f" The system shell here is {shell.name}." if shell is not None else ""
    return (
        ShellRunTool(
            ToolSpec(
                name="shell.run",
                description=(
                    "Run ONE allowed program with arguments (argv list, no shell). Pipes, "
                    "redirections and inline scripts are not supported by design; the binary "
                    "must be on the configured allowlist." + hint
                ),
                summary="uruchomienie programu (CRITICAL, wymaga zgody i allowlisty)",
                args_model=ShellRunArgs,
                risk=RiskLevel.CRITICAL,
                timeout_s=min(120.0, max(1.0, active.shell_timeout_s + 5.0)),
                idempotent=False,
            ),
            settings=active,
            workspace=workspace,
            runner=runner,
        ),
    )


__all__ = ["ShellRunArgs", "ShellRunTool", "build_shell_tools"]
