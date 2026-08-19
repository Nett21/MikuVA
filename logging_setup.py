"""Konfiguracja logowania do plików ``logs/assistant.log`` i ``logs/errors.log``.

Wydzielone z ``config.py``, żeby konfiguracja nie mieszała się z inicjalizacją
loggerów. Ścieżki pochodzą wyłącznie z ``config.py`` — ten moduł nie zna systemu
operacyjnego ani żadnej konkretnej ścieżki.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any

from config import ERROR_LOG_FILE, LOG_FILE, LOGS_DIR, Settings, get_settings

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "[ERROR] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


class SecretRedactingFilter(logging.Filter):
    """Wymazuje z KAŻDEGO wpisu logu wartości wyglądające na sekrety (Faza 9).

    Filtr na poziomie logowania, a nie tylko w naszym kodzie, bo największym
    źródłem wycieku są **biblioteki**: ``httpx`` zapisuje pełny adres żądania,
    a w adresie API siedzi klucz (``?key=…``). Zauważone w testach Fazy 9 —
    klucz trafiał do ``logs/assistant.log`` mimo tego, że nasz własny kod
    konsekwentnie go zamazywał.

    Redakcja jest wykonywana na sformatowanej treści wpisu, więc obejmuje też
    argumenty ``%s``. Import funkcji redagującej jest leniwy: ``logging_setup``
    jest konfigurowane bardzo wcześnie i nie może zależeć od warstwy sieciowej.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from host.http import redact
        except Exception:  # pragma: no cover - brak warstwy sieciowej
            return True

        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - błędne argumenty logu
            return True

        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


class _NoTracebackFormatter(logging.Formatter):
    """Formatter konsolowy bez stack trace'u — pełny ślad zostaje w pliku."""

    def formatException(self, ei: Any) -> str:  # noqa: N802 - nazwa z API logging
        return ""

    def formatStack(self, stack_info: str) -> str:  # noqa: N802 - nazwa z API logging
        return ""

    def format(self, record: logging.LogRecord) -> str:
        saved_exc_info, saved_exc_text, saved_stack = (
            record.exc_info,
            record.exc_text,
            record.stack_info,
        )
        record.exc_info, record.exc_text, record.stack_info = None, None, None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = (
                saved_exc_info,
                saved_exc_text,
                saved_stack,
            )


def _build_file_handler(
    path: Path, level: int, max_bytes: int, backup_count: int
) -> logging.Handler | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
    except OSError:
        # Katalog tylko do odczytu (np. instalacja systemowa) — logi do pliku
        # są wtedy niedostępne, ale program musi działać dalej.
        return None
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def setup_logging(
    settings: Settings | None = None,
    *,
    level_override: str | None = None,
    console: bool = False,
    force: bool = False,
) -> Path:
    """Skonfiguruj logowanie aplikacji. Zwraca katalog, w którym powstają logi.

    * ``logs/assistant.log`` — wszystko od poziomu ``LOG_LEVEL``,
    * ``logs/errors.log``    — wyłącznie ``ERROR`` i ``CRITICAL``, ze stack trace'em,
    * konsola                — domyślnie WYŁĄCZONA. Komunikaty dla użytkownika
      wypisuje ``main.py`` własnymi tagami; podwójne komunikaty i stack trace
      w terminalu byłyby szumem. ``console=True`` włącza podgląd błędów przy
      diagnostyce — nadal bez stack trace'u.
    """
    global _configured
    if _configured and not force:
        return LOGS_DIR

    active = settings or get_settings()
    level_name = (level_override or active.log_level).strip().upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover
            pass

    root.setLevel(min(level, logging.ERROR))

    main_handler = _build_file_handler(
        LOG_FILE, level, active.log_max_bytes, active.log_backup_count
    )
    if main_handler is not None:
        root.addHandler(main_handler)

    error_handler = _build_file_handler(
        ERROR_LOG_FILE, logging.ERROR, active.log_max_bytes, active.log_backup_count
    )
    if error_handler is not None:
        root.addHandler(error_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(_NoTracebackFormatter(_CONSOLE_FORMAT))
        root.addHandler(console_handler)

    if main_handler is None and error_handler is None:
        root.addHandler(logging.NullHandler())

    # Biblioteki HTTP potrafią być bardzo gadatliwe na DEBUG.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    # Sekrety nie trafiają do logu — filtr wisi na KAŻDYM uchwycie, więc obejmuje
    # też wpisy z bibliotek, nad których treścią nie mamy kontroli.
    redactor = SecretRedactingFilter()
    for handler in root.handlers:
        handler.addFilter(redactor)

    _configured = True
    return LOGS_DIR


__all__ = ["SecretRedactingFilter", "setup_logging"]
