"""Uruchamianie lokalnej Ollamy, żeby asystent nie wymagał drugiego terminala.

Do tej pory praca wyglądała tak: jedno okno terminala na ``ollama serve``, drugie
na asystenta. Dwa okna zajmujące pulpit po to, żeby jedno z nich tylko stało — to
nie jest wymaganie techniczne, tylko brak wygody.

Ten moduł domyka lukę: gdy model stoi **na tej maszynie** i usługa nie odpowiada,
asystent uruchamia ją sam, w tle, i mówi o tym jednym zdaniem.

Czego tu świadomie NIE ma:

* **żadnego ``sudo``** i żadnego włączania usług systemowych — asystent startuje
  proces w imieniu użytkownika, dokładnie tak, jakby ten wpisał ``ollama serve``,
* **żadnego dotykania zdalnego serwera** — gdy ``OLLAMA_HOST`` wskazuje inną
  maszynę, nie mamy tam czego uruchamiać i nawet nie próbujemy,
* **żadnego ubijania cudzych procesów** — jeśli usługa już działa (uruchomiona
  ręcznie albo przez systemd), zostawiamy ją w spokoju.

Proces startuje w **osobnej sesji** (``start_new_session``), więc zamknięcie
asystenta nie zabija modelu w połowie odpowiedzi, a Ollama zostaje dostępna dla
innych programów — tak jak usługa, którą jest.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    LOGS_DIR,
    Settings,
    detect_ollama,
    get_settings,
    is_local_host,
    subprocess_no_window_kwargs,
)
from i18n import t

logger = logging.getLogger(__name__)

# Ile czekamy, aż świeżo uruchomiona usługa zacznie odpowiadać. Ollama wstaje
# zwykle w ułamku sekundy; limit chroni przed czekaniem w nieskończoność, gdy
# proces wystartował, ale zaraz padł (np. zajęty port).
STARTUP_TIMEOUT_S: float = 12.0
POLL_INTERVAL_S: float = 0.3

# Nazwa pliku z wyjściem uruchomionej usługi — inaczej diagnostyka „dlaczego nie
# wstała" kończyłaby się na „nie wiem".
SERVER_LOG_NAME: str = "ollama-server.log"


@dataclass(frozen=True, slots=True)
class StartupResult:
    """Co się stało z próbą uruchomienia usługi."""

    running: bool
    started_by_us: bool = False
    message: str = ""
    hint: str = ""

    @property
    def ok(self) -> bool:
        return self.running


def find_binary() -> Path | None:
    """Program ``ollama`` na tej maszynie. ``None`` = nie ma czego uruchamiać.

    Szukamy wyłącznie w ``PATH`` — bez zgadywania katalogów instalacji, bo te
    różnią się między systemami i menedżerami pakietów, a zła ścieżka byłaby
    gorsza niż jej brak.
    """
    found = shutil.which("ollama")
    return Path(found) if found else None


def is_reachable(settings: Settings | None = None) -> bool:
    """Czy usługa odpowiada pod skonfigurowanym adresem?"""
    return detect_ollama(settings or get_settings()).reachable


def _log_path() -> Path:
    return LOGS_DIR / SERVER_LOG_NAME


def _spawn(binary: Path) -> subprocess.Popen[bytes] | None:
    """Uruchom ``ollama serve`` w tle. ``None`` = nie udało się wystartować."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handle = _log_path().open("ab")
    except OSError as exc:  # pragma: no cover - zależne od uprawnień
        logger.warning("Nie mogę pisać do logu usługi Ollamy: %s", exc)
        handle = subprocess.DEVNULL  # type: ignore[assignment]

    try:
        return subprocess.Popen(  # noqa: S603 - stała komenda, bez powłoki
            [str(binary), "serve"],
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            # Osobna sesja: zamknięcie asystenta (także Ctrl+C) nie ubija usługi.
            start_new_session=True,
            **subprocess_no_window_kwargs(),
        )
    except (OSError, ValueError) as exc:
        logger.warning("Nie udało się uruchomić `ollama serve`: %s", exc)
        return None


def ensure_running(
    settings: Settings | None = None, *, timeout_s: float = STARTUP_TIMEOUT_S
) -> StartupResult:
    """Zadbaj o to, żeby lokalna Ollama działała. Nigdy nie rzuca wyjątkiem.

    Zwracany opis nadaje się wprost do pokazania człowiekowi: mówi, czy usługa
    już działała, czy została uruchomiona teraz, czy nie da się jej uruchomić i
    dlaczego.
    """
    active = settings or get_settings()

    if not active.ollama_autostart:
        return StartupResult(running=is_reachable(active))

    if not is_local_host(active.ollama_host):
        # Zdalny serwer to cudza maszyna — nie nasza sprawa i nie nasze prawo.
        return StartupResult(
            running=is_reachable(active),
            message=t("ollama.remote", host=active.ollama_host),
        )

    if is_reachable(active):
        return StartupResult(running=True)

    binary = find_binary()
    if binary is None:
        return StartupResult(
            running=False,
            message=t("ollama.missing_binary"),
            hint=t("ollama.install_hint"),
        )

    logger.info("Uruchamiam `ollama serve` w tle (%s)", binary)
    process = _spawn(binary)
    if process is None:
        return StartupResult(
            running=False,
            message=t("ollama.start_failed"),
            hint=t("ollama.log_hint", path=_log_path()),
        )

    deadline = time.monotonic() + max(1.0, timeout_s)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            # Proces zdążył się zakończyć — zwykle „port zajęty" albo brak praw.
            return StartupResult(
                running=is_reachable(active),
                message=t("ollama.exited", code=process.returncode),
                hint=t("ollama.log_hint", path=_log_path()),
            )
        if is_reachable(active):
            return StartupResult(
                running=True,
                started_by_us=True,
                message=t("ollama.started"),
                hint=t("ollama.service_hint"),
            )
        time.sleep(POLL_INTERVAL_S)

    return StartupResult(
        running=False,
        message=t("ollama.timeout", seconds=int(timeout_s)),
        hint=t("ollama.log_hint", path=_log_path()),
    )


__all__ = [
    "SERVER_LOG_NAME",
    "STARTUP_TIMEOUT_S",
    "StartupResult",
    "ensure_running",
    "find_binary",
    "is_reachable",
]
