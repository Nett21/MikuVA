"""Punkt wejścia asystenta — tryb terminalowy i diagnostyka zależności.

Uruchomienie::

    python main.py                # tryb domyślny: okno graficzne
    python main.py --terminal     # rozmowa w terminalu
    python main.py --gui          # okno graficzne (Faza 10)
    python main.py --headless     # usługa w tle: mikrofon i mowa, bez okna
                                  # i bez klawiatury (systemd --user / Harmonogram
                                  # zadań); instalacja autostartu:
                                  # python scripts/install_autostart.py
    python main.py --check-deps   # sam raport zależności, bez rozmowy
    python main.py --offline      # gwarancja pracy bez sieci (nic nie pobiera)

Program nigdy nie kończy się surowym stack trace'em — każdy przewidywalny błąd
jest zamieniany na komunikat z tagiem ``[ERROR]``, a pełny ślad ląduje w
``logs/errors.log``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# --------------------------------------------------------------------------- #
# Bootstrap: bez pydantic/pydantic-settings nie da się nawet wczytać config.py.
# Zamiast stack trace'u pokazujemy, co zrobić. Nie duplikujemy tu detekcji
# systemu — jej właścicielem jest config.py, którego w tym scenariuszu nie ma.
# --------------------------------------------------------------------------- #
try:
    from config import (
        APP_VERSION,
        ENV_FILE,
        ERROR_LOG_FILE,
        LOG_FILE,
        TAG_ERROR,
        TAG_MIC,
        TAG_SYSTEM,
        TAG_TOOL,
        TAG_USER,
        TAG_WAKE,
        ConfigError,
        DependencyReport,
        Settings,
        apply_offline_environment,
        configured_reply_language,
        describe_offline_mode,
        detect_dependencies,
        ensure_directories,
        get_settings,
        get_user_settings,
        install_instruction,
        pip_install_hint,
        reload_user_settings,
        save_user_settings,
    )
except ImportError as bootstrap_error:  # pragma: no cover - zależne od środowiska
    print(
        "[ERROR] Brakuje pakietów Pythona wymaganych do uruchomienia asystenta "
        f"({bootstrap_error}).\n"
        "        Zainstaluj zależności:\n"
        "          python -m pip install -r requirements.txt\n"
        "        Bez internetu (po przygotowaniu magazynu kół):\n"
        "          python -m pip install --no-index --find-links vendor/wheels\n"
        "                                -r requirements.txt\n"
        "        albo uruchom skrypt instalacyjny dla swojego systemu z katalogu scripts/.",
        file=sys.stderr,
    )
    raise SystemExit(3) from bootstrap_error

from brain.conversation import ConversationHistory
from brain.memory import ConversationMemory
from brain.personality import (
    build_context_message,
    build_system_prompt,
    greeting,
    is_auto_language,
    normalize_language,
    resolve_reply_language,
)
from brain.remember import MemoryCurator, detect_memory_intent
from brain.request_kind import classify as classify_request
from brain.request_kind import prompt_hint as request_prompt_hint
from brain.request_kind import user_notice as request_notice
from brain.tool_router import tool_system_rules
from brain.turn import run_turn, stream_reply
from security.confirm import AutoDenyBroker
from i18n import set_ui_language, t
from logging_setup import setup_logging

if TYPE_CHECKING:  # importy tylko dla typów — te pakiety nie są potrzebne do --check-deps
    from audio.pipeline import PipelineMessage, SpeechToTextPipeline
    from audio.tts import SpeechOutput
    from brain.llm import OllamaClient, StreamedReply
    from brain.tool_router import ToolRouter
    from tools.base import ToolContext

logger = logging.getLogger("main")

EXIT_OK = 0
EXIT_MISSING_DEPENDENCIES = 1
EXIT_CONFIG_ERROR = 2

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "/wyjscie", "/wyjście", ":q", "/q"})
_HELP_COMMANDS = frozenset({"/help", "/pomoc", "/?"})
_CLEAR_COMMANDS = frozenset({"/clear", "/nowa", "/reset"})
_RELOAD_COMMANDS = frozenset({"/reload", "/przeladuj", "/przeładuj"})
_DEPS_COMMANDS = frozenset({"/deps", "/zaleznosci", "/zależności"})
_STATUS_COMMANDS = frozenset({"/status"})
# Pamięć długoterminowa (Faza 5).
_MEMORY_COMMAND_PREFIXES = ("/pamiec", "/pamięć", "/memory")
_MIC_COMMAND_PREFIX = "/mic"
_WAKE_COMMAND_PREFIX = "/wake"
# Wyjście głosowe (Faza 4): /glos po polsku, /voice i /tts dla przyzwyczajonych.
_VOICE_COMMAND_PREFIXES = ("/glos", "/głos", "/voice", "/tts")
# Narzędzia (Faza 7).
_TOOLS_COMMANDS = frozenset({"/narzedzia", "/narzędzia", "/tools"})


def _bootstrap_ui_language(argv: Sequence[str] | None) -> None:
    """Ustaw język interfejsu, ZANIM cokolwiek zostanie wypisane.

    Kolejność: ``--ui-lang`` z wiersza poleceń → zmienna środowiskowa →
    ``UI_LANGUAGE`` z pliku ``.env``. Świadomie nie wołamy tu ``get_settings()``:
    ustawienia wczytujemy dopiero po nałożeniu flag (``--offline`` i spółka), a
    teksty pomocy ``--help`` muszą być gotowe wcześniej.
    """
    wanted = ""
    arguments = list(argv if argv is not None else sys.argv[1:])
    for index, item in enumerate(arguments):
        if item == "--ui-lang" and index + 1 < len(arguments):
            wanted = arguments[index + 1]
        elif item.startswith("--ui-lang="):
            wanted = item.split("=", 1)[1]
    if not wanted:
        wanted = os.environ.get("UI_LANGUAGE", "")
    if not wanted:
        try:
            from dotenv import dotenv_values

            wanted = str(dotenv_values(ENV_FILE).get("UI_LANGUAGE") or "")
        except Exception:  # pragma: no cover - brak dotenv nie może blokować startu
            wanted = ""
    set_ui_language(wanted)


def _configure_streams() -> None:
    """Wymuś UTF-8 na wyjściu — konsola Windows domyślnie używa strony kodowej ANSI."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - egzotyczne terminale
            pass


# --------------------------------------------------------------------------- #
# Raport zależności
# --------------------------------------------------------------------------- #


def format_report_lines(report: DependencyReport) -> list[str]:
    """Czytelna lista: element -> ok/brak -> wykryta ścieżka."""
    lines: list[str] = []
    platform_info = report.platform_info
    lines.append(t("report.system", label=platform_info.os_label, machine=platform_info.machine))
    if platform_info.is_wsl:
        lines.append(t("report.wsl"))
    lines.append(
        t(
            "report.python",
            version=platform_info.python_version,
            executable=platform_info.python_executable,
        )
    )
    lines.append(
        t(
            "report.package_manager",
            manager=platform_info.package_manager,
            script=platform_info.install_script,
        )
    )
    lines.append("")
    lines.append(t("report.dependencies"))

    width = max((len(check.name) for check in report.checks), default=10)
    for check in report.checks:
        marker = t("report.ok") if check.ok else t("report.missing")
        required = t("report.required") if check.required else t("report.optional")
        line = f"  {check.name.ljust(width)}  {marker}  [{required}]"
        if check.detail:
            line += f"  {check.detail}"
        lines.append(line)
        if check.path:
            lines.append(f"  {' ' * width}        " + t("report.path", path=check.path))
        if not check.ok and check.hint:
            lines.append(f"  {' ' * width}        → {check.hint}")
    return lines


def print_dependency_report(report: DependencyReport) -> None:
    print()
    for line in format_report_lines(report):
        print(line)
    print()

    missing_required = report.missing_required
    if missing_required:
        print(f"{TAG_ERROR} " + t("report.missing_required", count=len(missing_required)))
        for check in missing_required:
            hint = f" → {check.hint}" if check.hint else ""
            print(f"        - {check.name}: {check.detail}{hint}")
        print(
            f"{TAG_ERROR} "
            + t(
                "report.nothing_automatic",
                command=install_instruction(report.platform_info),
            )
        )
    else:
        print(f"{TAG_SYSTEM} " + t("report.all_present"))

    missing_optional = report.missing_optional
    if missing_optional:
        print(f"{TAG_SYSTEM} " + t("report.optional_missing"))
        for check in missing_optional:
            print(f"        - {check.name}: {check.detail}")
    print()


# --------------------------------------------------------------------------- #
# Wejście głosowe (Faza 2)
# --------------------------------------------------------------------------- #


class VoiceInput:
    """Opakowanie potoku mowy na potrzeby pętli terminala.

    Cała warstwa audio jest opcjonalna: importy są leniwe, a każdy błąd kończy
    się wyłączeniem trybu głosowego i komunikatem — czat tekstowy działa dalej.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: SpeechToTextPipeline | None = None
        self._enabled = False
        self._unavailable_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def status_text(self) -> str:
        if self._enabled and self._pipeline is not None:
            return t("cli.voice.mic_on", detail=self._pipeline.describe())
        if self._unavailable_reason:
            return t("cli.voice.mic_unavailable", reason=self._unavailable_reason)
        return t("cli.voice.mic_off")

    @property
    def wake_status_text(self) -> str:
        """Opis bramki frazy — działa też, zanim potok wystartuje."""
        user_settings = get_user_settings()
        phrase = user_settings.effective_wake_word
        source = (
            t("cli.voice.wake_source_file")
            if user_settings.wake_word.strip()
            else t("cli.voice.wake_source_name")
        )
        if not self._settings.wake_enabled or self._settings.wake_engine == "none":
            return t("cli.voice.wake_off", phrase=phrase, source=source)
        if self._pipeline is None:
            return t("cli.voice.wake_pending", phrase=phrase, source=source)
        if self._pipeline.wake_engine is None:
            return t("cli.voice.wake_no_gate", phrase=phrase)
        state = (
            t("cli.voice.wake_awake") if self._pipeline.is_awake else t("cli.voice.wake_waiting")
        )
        return t(
            "cli.voice.wake_active",
            phrase=self._pipeline.wake_phrase,
            source=source,
            engine=self._pipeline.wake_name,
            state=state,
        )

    def _save_wake_phrase(self, new_phrase: str) -> None:
        """Zapisz frazę w ustawieniach użytkownika i przeładuj potok."""
        if not new_phrase:
            print(f"{TAG_SYSTEM} " + t("cli.voice.give_phrase"))
            return
        try:
            saved = save_user_settings({"wake_word": new_phrase})
        except ConfigError as exc:
            print(f"{TAG_ERROR} {exc.user_message}")
            return
        print(f"{TAG_WAKE} " + t("cli.voice.new_phrase", phrase=saved.effective_wake_word))
        self._rebuild_pipeline()

    def handle_wake_command(self, argument: str) -> None:
        """Obsłuż ``/wake``, ``/wake on|off``, ``/wake teraz``, ``/wake fraza <tekst>``."""
        argument = argument.strip()

        if argument.startswith(("fraza ", "phrase ", "slowo ", "słowo ")):
            self._save_wake_phrase(argument.split(" ", 1)[1].strip())
            return

        if argument in ("off", "stop", "wyl", "wył"):
            print(f"{TAG_SYSTEM} " + t("cli.voice.gate_off"))
            self._set_wake_enabled(False)
            return

        if argument in ("on", "start", "wl", "wł"):
            print(f"{TAG_SYSTEM} " + t("cli.voice.gate_on"))
            self._set_wake_enabled(True)
            return

        if argument in ("teraz", "now", "budz", "budź"):
            if self._pipeline is None:
                print(f"{TAG_SYSTEM} " + t("cli.voice.no_voice_mode"))
                return
            self._pipeline.wake_up()
            print(f"{TAG_WAKE} " + t("cli.voice.window_open"))
            return

        if argument in ("", "status"):
            print(f"{TAG_WAKE} " + t("cli.voice.wake_status", detail=self.wake_status_text))
            print(f"{TAG_SYSTEM} " + t("cli.voice.wake_usage"))
            return

        print(f"{TAG_SYSTEM} " + t("cli.voice.wake_usage"))

    def _set_wake_enabled(self, enabled: bool) -> None:
        """Zmień tryb bramki na czas sesji (bez dotykania plików konfiguracyjnych)."""
        self._settings = self._settings.model_copy(update={"wake_enabled": enabled})
        self._rebuild_pipeline()

    def _rebuild_pipeline(self) -> None:
        """Zbuduj potok od nowa, żeby zmiana ustawień zadziałała od razu."""
        was_enabled = self._enabled
        self.close()
        if was_enabled:
            self.enable()

    def _on_event(self, message: PipelineMessage) -> None:
        from audio.pipeline import PipelineEvent

        if message.event is PipelineEvent.TRANSCRIBED:
            # Rozpoznany tekst wypisuje pętla główna jako [USER].
            logger.info("Transkrypcja: %s (%s)", message.text, message.detail)
            return
        if message.event is PipelineEvent.ERROR:
            print(f"{TAG_ERROR} {message.text}")
            if message.detail:
                print(t("cli.voice.hint", detail=message.detail))
            return

        wake_events = (
            PipelineEvent.WAITING_FOR_WAKE,
            PipelineEvent.WAKE_DETECTED,
            PipelineEvent.IGNORED,
        )
        if message.event is PipelineEvent.IGNORED:
            # Odrzucone tło to normalna praca bramki, nie zdarzenie dla
            # użytkownika — inaczej terminal zalałby się komunikatami.
            logger.info("Pominięto mowę bez frazy: %s", message.detail)
            return
        tag = TAG_WAKE if message.event in wake_events else TAG_MIC
        detail = f" ({message.detail})" if message.detail else ""
        print(f"{tag} {message.text}{detail}")

    def enable(self) -> bool:
        """Uruchom mikrofon i model. Zwraca ``False`` z komunikatem przy błędzie."""
        if self._enabled:
            return True
        try:
            from audio.pipeline import SpeechPipelineError, SpeechToTextPipeline
        except ImportError as exc:
            self._unavailable_reason = f"brak pakietów audio ({exc})"
            print(
                f"{TAG_ERROR} " + t("cli.voice.mode_unavailable", reason=self._unavailable_reason)
            )
            print(t("cli.voice.install", hint=pip_install_hint()))
            return False

        if self._pipeline is None:
            self._pipeline = SpeechToTextPipeline(self._settings, on_event=self._on_event)

        print(f"{TAG_MIC} " + t("cli.voice.preparing"))
        try:
            self._pipeline.start()
        except SpeechPipelineError as exc:
            self._unavailable_reason = exc.message
            print(f"{TAG_ERROR} {exc.user_message}")
            print(f"{TAG_SYSTEM} " + t("cli.voice.staying_text"))
            self._pipeline = None
            return False
        except Exception as exc:
            logger.exception("Nieoczekiwany błąd uruchamiania trybu głosowego")
            self._unavailable_reason = str(exc)
            print(f"{TAG_ERROR} " + t("cli.voice.start_failed", error=exc))
            print(f"{TAG_ERROR} " + t("cli.voice.details_in", path=ERROR_LOG_FILE))
            self._pipeline = None
            return False

        self._enabled = True
        return True

    def disable(self) -> None:
        self._enabled = False
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.stop()

    def close(self) -> None:
        self._enabled = False
        if self._pipeline is not None:
            with contextlib.suppress(Exception):
                self._pipeline.close()
            self._pipeline = None

    def listen(self, *, timeout_s: float | None = None, quiet: bool = False) -> str | None:
        """Nasłuchuj jednej wypowiedzi.

        ``None`` oznacza „w tym obrocie użyj klawiatury" — cisza, przerwanie
        przez Ctrl+C albo błąd rozpoznawania.

        ``timeout_s`` skraca oczekiwanie na ciszę. Usługa (``--headless``) podaje
        tu kilka sekund zamiast domyślnych trzydziestu, bo między jednym a drugim
        wywołaniem sprawdza, czy nie przyszedł SIGTERM: gdyby czekała pełne
        ``VAD_LISTEN_TIMEOUT_S``, ``systemctl --user stop`` czekałby tyle samo i
        systemd zdążyłby dobić proces SIGKILL-em. Rozpoczętej wypowiedzi limit
        nie przerywa — dotyczy wyłącznie ciszy.

        ``quiet`` wycisza komunikat o ciszy. W usłudze nasłuch wraca co kilka
        sekund, więc bez tego dziennik zapełniłby się linią „nic nie usłyszałem".
        """
        if not self._enabled or self._pipeline is None:
            return None
        try:
            transcript = self._pipeline.listen_once(timeout_s=timeout_s)
        except KeyboardInterrupt:
            print()
            print(f"{TAG_SYSTEM} " + t("cli.voice.listen_interrupted"))
            self.disable()
            return None
        except Exception as exc:
            logger.exception("Błąd nasłuchu")
            print(f"{TAG_ERROR} " + t("cli.voice.listen_error", error=exc))
            print(f"{TAG_SYSTEM} " + t("cli.voice.listen_disabled"))
            self.disable()
            return None

        if transcript is None:
            if not quiet:
                print(f"{TAG_SYSTEM} " + t("cli.voice.nothing_heard"))
            return None
        return transcript.text

    def list_devices(self) -> None:
        """Wypisz mikrofony widziane przez system (nazwy do AUDIO_INPUT_DEVICE)."""
        try:
            from audio.microphone import MicrophoneError, list_input_devices
        except ImportError as exc:
            print(f"{TAG_ERROR} " + t("cli.voice.no_audio_packages", error=exc))
            return
        try:
            devices = list_input_devices(self._settings)
        except MicrophoneError as exc:
            print(f"{TAG_ERROR} {exc.user_message}")
            return
        if not devices:
            print(f"{TAG_MIC} " + t("cli.voice.no_input_devices"))
            return
        print(f"{TAG_MIC} " + t("cli.voice.devices"))
        for device in devices:
            print(f"        - {device.describe()}")


# --------------------------------------------------------------------------- #
# Wyjście głosowe (Faza 4)
# --------------------------------------------------------------------------- #


class VoiceOutput:
    """Opakowanie syntezy mowy na potrzeby pętli terminala.

    Mowa jest wyjściem OPCJONALNYM i tak jest tu traktowana: importy są leniwe,
    a każdy brak (pakietu, głosu, głośnika) kończy się jednym komunikatem i
    dalszą pracą w trybie tekstowym. Program nigdy nie przestaje odpowiadać
    dlatego, że nie ma czym powiedzieć.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._speech: SpeechOutput | None = None
        self._enabled = False
        self._unavailable_reason: str | None = None
        self._muted = False

    # --- stan ------------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._muted

    @property
    def is_unavailable(self) -> bool:
        """Czy mowa jest niedostępna z powodu BŁĘDU (a nie wyłączenia w ustawieniach).

        Osobna właściwość, bo porównywanie ``status_text`` z „niedostępna" działa
        tylko dopóki interfejs jest po polsku — a od Fazy 10 bywa po angielsku.
        """
        return not self._enabled and bool(self._unavailable_reason)

    @property
    def status_text(self) -> str:
        if self._muted:
            return t("cli.speech.muted")
        if self._enabled and self._speech is not None:
            return t("cli.speech.on", detail=self._speech.describe())
        if self._unavailable_reason:
            return t("cli.speech.unavailable", reason=self._unavailable_reason)
        return t("cli.speech.off")

    def _on_error(self, error: Any) -> None:
        """Błąd z wątku syntezy — pokazujemy raz i schodzimy do trybu tekstowego."""
        print()
        print(f"{TAG_ERROR} {error.user_message}")
        print(
            f"{TAG_SYSTEM} Wyłączam mowę — odpowiedzi zostają tekstowe "
            "(/glos on spróbuje ponownie)."
        )
        self._enabled = False
        self._unavailable_reason = getattr(error, "message", str(error))

    # --- włączanie -------------------------------------------------------- #

    def enable(self, *, quiet: bool = False) -> bool:
        """Zbuduj silnik mowy i wyjście audio. ``False`` = zostajemy przy tekście."""
        self._muted = False
        if self._enabled and self._speech is not None:
            return True

        try:
            from audio.output import AudioOutput, AudioOutputError
            from audio.tts import SpeechOutput as SpeechOutputImpl
            from audio.tts import TTSError, create_tts_provider
        except ImportError as exc:
            self._unavailable_reason = f"brak pakietów audio ({exc})"
            if not quiet:
                print(
                    f"{TAG_ERROR} "
                    + t("cli.speech.speech_unavailable", reason=self._unavailable_reason)
                )
                print(t("cli.voice.install", hint=pip_install_hint()))
            return False

        provider = create_tts_provider(self._settings)
        if not provider.is_speaking_enabled:
            self._unavailable_reason = provider.describe()
            if not quiet:
                print(
                    f"{TAG_SYSTEM} "
                    + t("cli.speech.disabled_in_settings", detail=provider.describe())
                )
            return False

        try:
            provider.load()
        except TTSError as exc:
            self._unavailable_reason = exc.message
            if not quiet:
                print(f"{TAG_ERROR} {exc.user_message}")
                print(f"{TAG_SYSTEM} " + t("cli.speech.text_only"))
            return False
        except Exception as exc:
            logger.exception("Nieoczekiwany błąd uruchamiania mowy")
            self._unavailable_reason = str(exc)
            if not quiet:
                print(f"{TAG_ERROR} " + t("cli.speech.start_failed", error=exc))
                print(f"{TAG_ERROR} " + t("cli.voice.details_in", path=ERROR_LOG_FILE))
            return False

        try:
            sink = AudioOutput(self._settings)
        except AudioOutputError as exc:  # pragma: no cover - konstruktor nie sięga do sprzętu
            self._unavailable_reason = exc.message
            if not quiet:
                print(f"{TAG_ERROR} {exc.user_message}")
            return False

        self._speech = SpeechOutputImpl(
            provider, sink, settings=self._settings, on_error=self._on_error
        )
        self._enabled = True
        self._unavailable_reason = None
        return True

    def disable(self) -> None:
        """Wycisz na czas sesji (bez zwalniania modelu — powrót jest natychmiastowy)."""
        self._muted = True
        self.cancel()

    def close(self) -> None:
        self._enabled = False
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.close()
            self._speech = None

    # --- mówienie ---------------------------------------------------------- #

    def begin(self, language: str | None = None) -> None:
        if self.enabled and self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.begin(language)

    def feed(self, text: str) -> None:
        if self.enabled and self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.feed(text)

    def end(self) -> None:
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.end(wait=True)

    def cancel(self) -> None:
        if self._speech is not None:
            with contextlib.suppress(Exception):
                self._speech.cancel()

    # --- komendy ------------------------------------------------------------ #

    def list_voices(self) -> None:
        """Wypisz głosy widoczne na tej maszynie (nazwy do ``piper_model``)."""
        try:
            from audio.tts import iter_piper_voices
            from config import piper_voice_directories
        except ImportError as exc:
            print(f"{TAG_ERROR} " + t("cli.voice.no_audio_packages", error=exc))
            return

        voices = iter_piper_voices(self._settings)
        if not voices:
            print(f"{TAG_SYSTEM} " + t("cli.speech.no_voices"))
            for directory in piper_voice_directories(self._settings):
                print(f"        - {directory}")
            print(f"{TAG_SYSTEM} " + t("cli.speech.download_voice"))
            return

        user_settings = get_user_settings()
        current = user_settings.piper_model.strip().lower()
        print(f"{TAG_SYSTEM} " + t("cli.speech.voices"))
        for voice in voices:
            marker = t("cli.speech.selected_marker") if voice.name.lower() == current else ""
            print(f"        - {voice.describe()}{marker}")
        if not current:
            print(f"{TAG_SYSTEM} " + t("cli.speech.auto_voice_note"))

    def set_voice(self, name: str) -> None:
        """Zapisz wybrany głos w ustawieniach użytkownika i przeładuj silnik."""
        if not name:
            print(f"{TAG_SYSTEM} " + t("cli.speech.give_voice"))
            return
        try:
            from audio.tts import find_piper_voice
        except ImportError as exc:
            print(f"{TAG_ERROR} " + t("cli.voice.no_audio_packages", error=exc))
            return

        voice = find_piper_voice(name, self._settings)
        if voice is None:
            print(f"{TAG_ERROR} " + t("cli.speech.voice_not_found", name=name))
            print(f"{TAG_SYSTEM} " + t("cli.speech.list_hint"))
            return

        try:
            save_user_settings({"piper_model": voice.name})
        except ConfigError as exc:
            print(f"{TAG_ERROR} {exc.user_message}")
            return

        print(f"{TAG_SYSTEM} " + t("cli.speech.new_voice", detail=voice.describe()))
        self.close()
        self.enable()

    def say_sample(self, text: str = "") -> None:
        """Powiedz zdanie próbne — najprostszy test, czy dźwięk w ogóle wychodzi."""
        if not self._enabled and not self.enable():
            return
        self._muted = False
        user_settings = get_user_settings()
        # Próbka jest MÓWIONA, więc idzie w języku odpowiedzi, nie interfejsu.
        sample = text.strip() or t(
            "cli.speech.sample",
            _lang=_preferred_language(self._settings),
            name=user_settings.assistant_name,
        )
        print(f"{TAG_SYSTEM} " + t("cli.speech.saying", text=sample))
        if self._speech is not None:
            self._speech.speak(sample)

    def save_sample(self, target: str) -> None:
        """Zapisz zdanie próbne do pliku WAV (gdy nie ma głośnika albo do porównania głosów)."""
        if not target:
            print(f"{TAG_SYSTEM} " + t("cli.speech.give_file"))
            return
        try:
            from audio.tts import TTSError, create_tts_provider, write_wav
        except ImportError as exc:
            print(f"{TAG_ERROR} " + t("cli.voice.no_audio_packages", error=exc))
            return

        provider = create_tts_provider(self._settings)
        if not provider.is_speaking_enabled:
            print(
                f"{TAG_SYSTEM} " + t("cli.speech.disabled_in_settings", detail=provider.describe())
            )
            return

        user_settings = get_user_settings()
        sample = t(
            "cli.speech.sample",
            _lang=_preferred_language(self._settings),
            name=user_settings.assistant_name,
        )
        try:
            provider.load()
            path = write_wav(Path(target), provider.synthesize(sample))
        except TTSError as exc:
            print(f"{TAG_ERROR} {exc.user_message}")
            return
        finally:
            with contextlib.suppress(Exception):
                provider.close()
        print(f"{TAG_SYSTEM} " + t("cli.speech.sample_saved", path=path))

    def handle_command(self, argument: str) -> None:
        """Obsłuż ``/glos``, ``/glos on|off|lista|test|model <nazwa>|zapisz <plik>``."""
        argument = argument.strip()

        if argument.startswith(("model ", "glos ", "głos ", "voice ")):
            self.set_voice(argument.split(" ", 1)[1].strip())
            return

        if argument.startswith(("zapisz ", "save ", "wav ")):
            self.save_sample(argument.split(" ", 1)[1].strip())
            return

        if argument in ("lista", "list", "voices", "glosy", "głosy"):
            self.list_voices()
            return

        if argument in ("test", "proba", "próba"):
            self.say_sample()
            return

        if argument in ("off", "stop", "wyl", "wył", "cicho"):
            if self.enabled:
                self.disable()
                print(f"{TAG_SYSTEM} " + t("cli.speech.muted_now"))
            else:
                print(f"{TAG_SYSTEM} " + t("cli.speech.already_off"))
            return

        if argument in ("", "on", "start", "wl", "wł"):
            if self.enabled:
                if argument == "":  # samo /glos działa jak przełącznik
                    self.disable()
                    print(f"{TAG_SYSTEM} " + t("cli.speech.muted_now"))
                else:
                    print(f"{TAG_SYSTEM} " + t("cli.speech.already_on"))
                return
            if self.enable():
                print(f"{TAG_SYSTEM} " + t("cli.speech.state", detail=self.status_text))
            return

        if argument in ("status",):
            print(f"{TAG_SYSTEM} " + t("cli.speech.state", detail=self.status_text))
            return

        print(f"{TAG_SYSTEM} " + t("cli.speech.usage"))


def run_audio_check(settings: Settings, *, seconds: float = 6.0) -> int:
    """Zmierz tło akustyczne i zaproponuj próg VAD dla tego konkretnego sprzętu.

    Progi energetyczne zależą od mikrofonu i pomieszczenia — stała dobra na
    jednej maszynie bywa bezużyteczna na innej. Zamiast zgadywać, mierzymy.
    """
    try:
        from audio.microphone import Microphone, MicrophoneError
        from audio.resample import rms_dbfs
        from audio.vad import EnergyVAD, create_vad
    except ImportError as exc:
        print(f"{TAG_ERROR} " + t("cli.voice.no_audio_packages", error=exc))
        print(t("cli.voice.install", hint=pip_install_hint()))
        return EXIT_MISSING_DEPENDENCIES

    microphone = Microphone(settings)
    try:
        microphone.start()
    except MicrophoneError as exc:
        print(f"{TAG_ERROR} {exc.user_message}")
        return EXIT_MISSING_DEPENDENCIES

    device = microphone.device
    print()
    print(
        f"{TAG_MIC} "
        + t(
            "cli.audio.device",
            detail=device.describe() if device else t("cli.audio.system_default"),
        )
    )
    print(f"{TAG_MIC} " + t("cli.audio.measuring", seconds=f"{seconds:.0f}"))

    frames = []
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            frame = microphone.read(timeout=0.3)
            if frame is not None:
                frames.append(frame)
    except KeyboardInterrupt:
        print()
    finally:
        microphone.stop()

    if not frames:
        print(f"{TAG_ERROR} " + t("cli.audio.no_samples"))
        return EXIT_MISSING_DEPENDENCIES

    levels = sorted(rms_dbfs(frame.samples) for frame in frames)
    quiet = levels[len(levels) // 10]
    median = levels[len(levels) // 2]
    loudest = levels[-1]

    print(
        f"{TAG_MIC} " + t("cli.audio.frames", frames=len(frames), dropped=microphone.dropped_frames)
    )
    print(
        f"{TAG_MIC} "
        + t(
            "cli.audio.levels",
            quiet=f"{quiet:.1f}",
            median=f"{median:.1f}",
            peak=f"{loudest:.1f}",
        )
    )

    print(f"{TAG_MIC} " + t("cli.audio.thresholds"))
    best: float | None = None
    for threshold in (8.0, 10.0, 12.0, 14.0, 16.0, 20.0):
        detector = EnergyVAD(threshold_db=threshold, frame_ms=settings.audio_frame_ms)
        speech_frames = sum(1 for frame in frames if detector.is_speech(frame))
        share = 100.0 * speech_frames / len(frames)
        marker = ""
        if share <= 5.0 and best is None:
            best = threshold
            marker = t("cli.audio.suggested")
        print(
            t(
                "cli.audio.frames_share",
                threshold=f"{threshold:<5.0f}",
                share=f"{share:5.1f}",
                marker=marker,
            )
        )

    active_vad = create_vad(settings)
    print(f"{TAG_MIC} " + t("cli.audio.active_vad", name=active_vad.name))
    if active_vad.name == "webrtc":
        print(f"{TAG_SYSTEM} " + t("cli.audio.webrtc_note"))
    elif best is not None:
        print(f"{TAG_SYSTEM} " + t("cli.audio.write_env", value=f"{best:.0f}"))
        print(f"{TAG_SYSTEM} " + t("cli.audio.better"))
    else:
        print(f"{TAG_SYSTEM} " + t("cli.audio.too_loud"))
        print(f"{TAG_SYSTEM} " + t("cli.audio.recommended"))
    print()
    return EXIT_OK


def _handle_mic_command(command: str, voice: VoiceInput) -> None:
    """Obsłuż ``/mic``, ``/mic on``, ``/mic off``, ``/mic lista``."""
    argument = command[len(_MIC_COMMAND_PREFIX) :].strip()

    if argument in ("lista", "list", "devices", "urzadzenia", "urządzenia"):
        voice.list_devices()
        return

    if argument in ("off", "stop", "wyl", "wył"):
        if voice.enabled:
            voice.disable()
            print(f"{TAG_SYSTEM} " + t("cli.mic.off"))
        else:
            print(f"{TAG_SYSTEM} " + t("cli.mic.already_off"))
        return

    if argument in ("", "on", "start", "wl", "wł"):
        if voice.enabled:
            if argument == "":  # samo /mic działa jak przełącznik
                voice.disable()
                print(f"{TAG_SYSTEM} " + t("cli.mic.off"))
            else:
                print(f"{TAG_SYSTEM} " + t("cli.mic.already_on"))
            return
        if voice.enable():
            print(f"{TAG_SYSTEM} " + t("cli.mic.on"))
        return

    print(f"{TAG_SYSTEM} " + t("cli.mic.usage"))


# --------------------------------------------------------------------------- #
# Tryb terminalowy
# --------------------------------------------------------------------------- #


def _preferred_language(settings: Settings) -> str:
    """Język ODPOWIEDZI asystenta — patrz :func:`config.configured_reply_language`.

    Świadomie NIE jest to ``speech_language``: ten opisuje język, w którym
    użytkownik mówi, i może być listą („pl,en"). Zwracana wartość może być
    ``auto`` — wtedy (i tylko wtedy) język odpowiedzi wynika z rozpoznania
    wypowiedzi, a nie z ustawienia.
    """
    return configured_reply_language(settings)


def _print_header(settings: Settings, report: DependencyReport) -> None:
    user_settings = get_user_settings()
    language = normalize_language(_preferred_language(settings))
    print()
    print("=" * 68)
    print(t("cli.header.title", name=user_settings.assistant_name, version=APP_VERSION))
    print("=" * 68)
    print(t("cli.header.model", model=settings.ollama_model, host=settings.ollama_host))
    print(t("cli.header.mode", mode=describe_offline_mode(settings)))
    print(
        t(
            "cli.header.system",
            label=report.platform_info.os_label,
            machine=report.platform_info.machine,
        )
    )
    print(t("cli.header.gpu", detail=report.gpu.detail))
    print(t("cli.header.logs", path=LOG_FILE.parent))
    print("=" * 68)
    print(f"{TAG_SYSTEM} {greeting(user_settings, language=language)}")
    print(f"{TAG_SYSTEM} " + t("cli.header.commands"))

    microphone_check = next((check for check in report.checks if check.name == "Mikrofon"), None)
    if microphone_check is not None and microphone_check.ok:
        print(f"{TAG_SYSTEM} " + t("cli.header.mic_found"))
        if settings.wake_enabled and settings.wake_engine != "none":
            print(f"{TAG_WAKE} " + t("cli.header.wake", phrase=user_settings.effective_wake_word))
    elif microphone_check is not None:
        print(f"{TAG_SYSTEM} " + t("cli.header.no_voice", detail=microphone_check.detail))
    print()


def _print_help() -> None:
    """Lista komend w języku interfejsu.

    Nazwy komend też są tłumaczone — i to działa, bo pętla rozmowy od Fazy 1
    przyjmuje oba warianty (``/help`` i ``/pomoc``, ``/voice`` i ``/glos``).
    """
    print(f"{TAG_SYSTEM} " + t("cli.help.title"))
    for line in t("cli.help.body").splitlines():
        print(line)


def _print_status(
    settings: Settings,
    memory: ConversationMemory,
    voice: VoiceInput,
    speaker: VoiceOutput | None = None,
    router: ToolRouter | None = None,
) -> None:
    user_settings = get_user_settings()
    history = memory.history
    tag = f"{TAG_SYSTEM} "
    print(tag + t("cli.status.model", model=settings.ollama_model, host=settings.ollama_host))
    print(tag + t("cli.status.mode", mode=describe_offline_mode(settings)))
    print(
        tag
        + t(
            "cli.status.history",
            messages=len(history),
            max_messages=history.max_messages,
            chars=history.char_count,
            max_chars=history.max_chars,
        )
    )
    print(tag + t("cli.status.memory", detail=memory.status_text))
    print(tag + t("cli.status.saved", detail=memory.stats_line()))
    print(tag + t("cli.status.semantic", detail=memory.semantic_line()))
    if memory.summary:
        print(tag + t("cli.status.summary", detail=memory.summary[:120]))
    print(
        tag
        + t(
            "cli.status.tools",
            detail=router.describe() if router is not None else t("cli.status.tools_off"),
        )
    )
    if router is not None:
        print(tag + t("cli.status.audit", detail=router.audit.summary()))
    print(tag + t("cli.status.mic", detail=voice.status_text))
    print(tag + t("cli.status.wake", detail=voice.wake_status_text))
    speech_language = user_settings.speech_language
    print(
        tag
        + t(
            "cli.status.speech_language",
            detail=(
                t("cli.status.language_auto")
                if speech_language == "auto"
                else t("cli.status.language_forced", code=speech_language)
            ),
        )
    )
    print(
        tag
        + t(
            "cli.status.assistant",
            name=user_settings.assistant_name,
            tag=user_settings.log_tag,
        )
    )
    print(tag + t("cli.status.accent", color=user_settings.ui_accent_color))
    print(
        tag
        + t(
            "cli.status.speech",
            detail=(
                speaker.status_text if speaker is not None else t("cli.status.speech_not_started")
            ),
        )
    )
    print(
        tag
        + t(
            "cli.status.engine",
            engine=user_settings.voice_engine,
            voice=user_settings.piper_model or t("cli.status.auto_voice"),
            speed=f"{user_settings.voice_speed:.2f}",
            volume=f"{user_settings.voice_volume:.2f}",
            rvc=t("common.yes") if user_settings.rvc.enabled else t("common.no"),
        )
    )
    traits = user_settings.personality_traits.strip()
    print(tag + t("cli.status.traits", detail=traits or t("cli.status.none")))


def _handle_memory_command(argument: str, memory: ConversationMemory) -> None:
    """Obsłuż ``/pamiec`` i jej warianty.

    Bez argumentu: stan pamięci. Dalej: ``fakty``, ``zapamietaj klucz=wartość``,
    ``zapomnij klucz``, ``notatka <tekst>``, ``notatki``, ``szukaj <fraza>``.
    """
    command, _, rest = argument.strip().partition(" ")
    command = command.strip().lower()
    rest = rest.strip()

    if not command or command in ("stan", "status"):
        print(f"{TAG_SYSTEM} " + t("cli.mem.state", detail=memory.status_text))
        print(f"{TAG_SYSTEM} " + t("cli.mem.saved", detail=memory.stats_line()))
        print(
            f"{TAG_SYSTEM} "
            + t(
                "cli.mem.window",
                messages=len(memory.history),
                pending=memory.pending_count,
            )
        )
        print(f"{TAG_SYSTEM} " + t("cli.mem.semantic", detail=memory.semantic_line()))
        if memory.summary:
            print(f"{TAG_SYSTEM} " + t("cli.mem.summary", detail=memory.summary))
        if not memory.persistent:
            print(f"{TAG_SYSTEM} " + t("cli.mem.hint"))
        return

    if not memory.persistent:
        print(f"{TAG_ERROR} " + t("cli.mem.unavailable", detail=memory.status_text))
        return

    if command in ("fakty", "facts"):
        facts = memory.facts()
        if not facts:
            print(f"{TAG_SYSTEM} " + t("cli.mem.no_facts"))
            print(f"{TAG_SYSTEM} " + t("cli.mem.add_fact_hint"))
            return
        print(f"{TAG_SYSTEM} " + t("cli.mem.facts", count=len(facts)))
        for fact in facts:
            print(t("cli.mem.fact_line", key=fact.key, value=fact.value, source=fact.source))
        preferences = memory.preferences()
        for preference in preferences:
            print(
                t(
                    "cli.mem.preference_line",
                    key=preference.key,
                    value=preference.value,
                )
            )
        return

    if command in ("zapamietaj", "zapamiętaj", "remember"):
        key, separator, value = rest.partition("=")
        if not separator:
            key, separator, value = rest.partition(":")
        if not separator or not key.strip() or not value.strip():
            print(f"{TAG_SYSTEM} " + t("cli.mem.remember_usage"))
            return
        if memory.remember_fact(key.strip(), value.strip()):
            print(
                f"{TAG_SYSTEM} "
                + t("cli.mem.remembered", key=key.strip().lower(), value=value.strip())
            )
        else:
            print(f"{TAG_ERROR} " + t("cli.mem.remember_failed", error=memory.error))
        return

    if command in ("zapomnij", "forget"):
        if not rest:
            print(f"{TAG_SYSTEM} " + t("cli.mem.forget_usage"))
            return
        if memory.forget_fact(rest):
            print(f"{TAG_SYSTEM} " + t("cli.mem.forgotten", key=rest.strip().lower()))
        else:
            print(f"{TAG_SYSTEM} " + t("cli.mem.no_such_fact", key=rest.strip().lower()))
        return

    if command in ("notatka", "note"):
        if not rest:
            print(f"{TAG_SYSTEM} " + t("cli.mem.note_usage"))
            return
        note = memory.add_note(rest)
        if note is None:
            print(f"{TAG_ERROR} " + t("cli.mem.note_failed", error=memory.error))
        else:
            print(f"{TAG_SYSTEM} " + t("cli.mem.note_saved", id=note.id))
        return

    if command in ("notatki", "notes"):
        notes = memory.notes()
        if not notes:
            print(f"{TAG_SYSTEM} " + t("cli.mem.no_notes"))
            return
        print(f"{TAG_SYSTEM} " + t("cli.mem.notes", count=len(notes)))
        for note in notes:
            print(f"        #{note.id} {note.preview}")
        return

    if command in ("przypomnij", "skojarz", "recall"):
        if not rest:
            print(f"{TAG_SYSTEM} " + t("cli.mem.recall_usage"))
            return
        if not memory.semantic_available:
            print(
                f"{TAG_SYSTEM} " + t("cli.mem.semantic_unavailable", detail=memory.semantic_line())
            )
            print(f"{TAG_SYSTEM} " + t("cli.mem.word_search_works", phrase=rest))
            return
        hits = memory.recall(rest, limit=10)
        if not hits:
            print(f"{TAG_SYSTEM} " + t("cli.mem.nothing_recalled", phrase=rest))
            return
        print(f"{TAG_SYSTEM} " + t("cli.mem.recalled", count=len(hits)))
        for hit in hits:
            when = hit.created_at.astimezone().strftime("%Y-%m-%d") if hit.created_at else "?"
            print(f"        [{when}] {hit.score:.2f} {hit.source_table}: {hit.preview}")
        return

    if command in ("reindeks", "reindex", "przelicz"):
        if not memory.semantic_available:
            print(
                f"{TAG_SYSTEM} " + t("cli.mem.semantic_unavailable", detail=memory.semantic_line())
            )
            return
        print(f"{TAG_SYSTEM} " + t("cli.mem.reindexing"))
        count = memory.reindex(progress=lambda table, done: print(f"        {table}: {done}"))
        print(
            f"{TAG_SYSTEM} " + t("cli.mem.reindex_done", count=count, detail=memory.semantic_line())
        )
        return

    if command in ("szukaj", "search"):
        if not rest:
            print(f"{TAG_SYSTEM} " + t("cli.mem.search_usage"))
            return
        hits = memory.search(rest)
        if not hits:
            print(f"{TAG_SYSTEM} " + t("cli.mem.nothing_found", phrase=rest))
            return
        print(f"{TAG_SYSTEM} " + t("cli.mem.found", count=len(hits)))
        for hit in hits:
            when = hit.created_at.astimezone().strftime("%Y-%m-%d") if hit.created_at else "?"
            skad = t("cli.mem.note_kind") if hit.kind == "note" else t("cli.mem.chat_kind")
            kto = f" ({hit.context})" if hit.context else ""
            print(f"        [{when}] {skad}{kto}: {hit.preview}")
        return

    print(f"{TAG_SYSTEM} " + t("cli.mem.unknown_command", command=command))
    print(f"{TAG_SYSTEM} " + t("cli.mem.available_commands"))


def _say_line(speaker: VoiceOutput | None, text: str, language: str | None = None) -> None:
    """Wypowiedz jedno zdanie tą samą drogą co odpowiedzi modelu (jeśli mowa działa)."""
    if speaker is None or not speaker.enabled:
        return
    try:
        speaker.begin(language)
        speaker.feed(text)
        speaker.end()
    except Exception:  # mowa jest dodatkiem — nie może wywrócić potwierdzenia
        logger.debug("Nie udało się wypowiedzieć potwierdzenia", exc_info=True)


def _handle_memory_intent(
    loop: asyncio.AbstractEventLoop,
    curator: MemoryCurator,
    intent: Any,
    client: OllamaClient,
    memory: ConversationMemory,
    tag: str,
    language: str,
    speaker: VoiceOutput | None,
) -> bool:
    """Wykonaj „zapamiętaj/zapomnij". ``True`` = tura obsłużona, nie pytamy modelu.

    Ocena treści idzie do modelu (jedno krótkie zapytanie), ale jego brak nie
    blokuje niczego: :class:`MemoryCurator` ma wtedy własną heurystykę.
    """
    try:
        outcome = loop.run_until_complete(curator.handle(intent, client, language=language))
    except KeyboardInterrupt:
        print(f"{TAG_SYSTEM} " + t("cli.mem.save_interrupted"))
        return True
    except Exception as exc:
        logger.exception("Obsługa polecenia pamięciowego nie powiodła się")
        print(f"{TAG_ERROR} " + t("cli.mem.handle_failed", error=exc))
        return True

    print(f"{tag} {outcome.message}")
    # Potwierdzenie ląduje też w oknie rozmowy — inaczej model w następnej turze
    # nie wiedziałby, że cokolwiek zostało zapamiętane.
    memory.add_assistant(outcome.message, language=language)
    _say_line(speaker, outcome.message, language)
    return True


class _TerminalView:
    """Wypisywanie tury w terminalu — kanał wyjściowy dla :mod:`brain.turn`.

    Cała logika tury (strumień, pętla narzędziowa, budżet) mieszka od Fazy 10 w
    ``brain/turn.py``, żeby terminal i GUI prowadziły dokładnie tę samą rozmowę.
    Tutaj zostaje wyłącznie to, co odróżnia terminal: tagi i ``print``.
    """

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def on_thinking(self) -> None:
        # Modele rozumujące milczą przez pierwsze kilkanaście sekund — bez tego
        # sygnału terminal wygląda na zawieszony.
        print(f"{TAG_SYSTEM} model analizuje pytanie...", flush=True)

    def on_reply_start(self) -> None:
        print(f"{self._tag} ", end="", flush=True)

    def on_chunk(self, text: str) -> None:
        print(text, end="", flush=True)

    def on_reply_end(self, text: str) -> None:
        print()

    def on_tool(self, outcome: Any) -> None:
        print(f"{TAG_TOOL} {outcome.line_for_user()}")

    def on_notice(self, text: str) -> None:
        print(f"{TAG_SYSTEM} {text}")


async def _stream_answer(
    client: OllamaClient,
    history: ConversationHistory,
    tag: str,
    system_prompt: str,
    *,
    speaker: VoiceOutput | None = None,
    language: str | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    collect: StreamedReply | None = None,
) -> str:
    """Wypisz (i wypowiedz) odpowiedź modelu w miarę jej napływania.

    Sama mechanika strumienia siedzi w :func:`brain.turn.stream_reply` — tutaj
    zostaje wpięcie terminalowego widoku.
    """
    return await stream_reply(
        client,
        history,
        system_prompt,
        view=_TerminalView(tag),
        speaker=speaker,
        language=language,
        tools=tools,
        collect=collect,
    )


def _build_plugins(settings: Settings, memory: Any) -> Any:
    """Menedżer pluginów dla tej sesji (Faza 11). Nigdy nie rzuca wyjątkiem."""
    if not settings.plugins_enabled:
        return None
    try:
        from plugins.manager import PluginContext, PluginManager

        manager = PluginManager(
            settings,
            context=PluginContext(
                settings=settings, database=memory.database, memory=memory
            ),
        )
        manager.load()
        return manager
    except Exception as exc:  # pragma: no cover - zależne od zawartości plugins/
        logger.warning("Warstwa pluginów niedostępna: %s", exc)
        return None


def _print_plugin_notices(plugins: Any, speak: Any = None) -> None:
    """Wypisz to, co pluginy chcą powiedzieć same z siebie (np. przypomnienia).

    W terminalu sprawdzamy to przed każdym pytaniem o wejście, bo ``input()``
    blokuje wątek: przypomnienie pojawi się przy najbliższej interakcji, a nie
    co do sekundy. W GUI jest inaczej — tam pętla robocza sprawdza pluginy sama
    (patrz ``gui/runtime.py``), więc budzik odzywa się o czasie.
    """
    if plugins is None:
        return
    for notice in plugins.poll():
        print(f"{TAG_SYSTEM} {notice.text}")
        if notice.speak and speak is not None:
            with contextlib.suppress(Exception):
                speak(notice.text)


def _build_tools(
    settings: Settings,
    memory: ConversationMemory,
    *,
    plugins: Any = None,
    broker: Any = None,
) -> tuple[ToolRouter | None, ToolContext | None]:
    """Zbuduj router narzędzi i kontekst dla nich (Faza 7).

    Zwraca ``(None, None)``, gdy narzędzia są wyłączone albo gdy warstwy narzędzi
    nie da się zaimportować — rozmowa działa wtedy dokładnie jak przed Fazą 7.

    ``broker`` to kanał potwierdzeń. Bez niego router dobiera go sam
    (``default_broker``: terminal, gdy jest interaktywny; inaczej odmowa).
    Tryb bezobsługowy podaje go WPROST, żeby z kodu było widać, że w usłudze
    nikt nie zatwierdzi akcji HIGH/CRITICAL.
    """
    if not settings.tools_enabled:
        return None, None
    try:
        from brain.tool_router import build_router
        from tools.base import ToolContext
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Warstwa narzędzi jest niedostępna: %s", exc)
        print(f"{TAG_SYSTEM} " + t("cli.tools.unavailable", error=exc))
        return None, None

    try:
        router = build_router(
            settings,
            broker=broker,
            database=memory.database,
            conversation_id=memory.conversation_id,
            # Narzędzia notatek pracują na tej samej pamięci co rozmowa (Faza 8),
            # więc notatka zapisana narzędziem jest od razu wyszukiwalna po znaczeniu.
            memory=memory,
            plugins=plugins,
        )
    except Exception as exc:
        logger.exception("Nie udało się zbudować routera narzędzi")
        print(f"{TAG_ERROR} " + t("cli.tools.failed", error=exc))
        return None, None

    context = ToolContext(settings=settings, dry_run=settings.security_dry_run)
    return router, context


def _web_tools_ready(router: ToolRouter | None) -> bool:
    """Czy model ma czym sięgnąć po świeże dane (Faza 9).

    Nie wystarczy „narzędzia włączone": w trybie offline, bez internetu albo przy
    ``WEB_ENABLED=false`` narzędzia sieciowe są niewidoczne dla modelu — a wtedy
    użytkownik ma usłyszeć, że odpowiedź nie będzie bieżąca.
    """
    if router is None:
        return False
    return any(
        "." in name and name.split(".")[0] in ("web", "weather", "news", "youtube")
        for name in router.visible_names()
    )


def _print_tools(router: ToolRouter | None, settings: Settings, language: str = "pl") -> None:
    """Komenda ``/narzedzia`` — co jest zarejestrowane, z jakim ryzykiem i stanem."""
    if router is None:
        print(f"{TAG_SYSTEM} " + t("cli.tools.disabled"))
        return
    print(f"{TAG_SYSTEM} " + t("cli.tools.list", detail=router.describe()))
    # Dozwolone katalogi to pierwsza rzecz, o którą pyta użytkownik przy
    # narzędziach plikowych — pokazujemy je bez proszenia (Faza 8).
    try:
        from host.paths import Workspace

        print(
            f"{TAG_SYSTEM} "
            + t("cli.tools.files", detail=Workspace.from_settings(settings).describe())
        )
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.debug("Nie udało się opisać obszaru plików: %s", exc)
    for line in router.registry.describe(router.policy, language=language):
        print(f"        {line}")
    audit = router.audit.summary()
    print(f"{TAG_SYSTEM} " + t("cli.tools.audit", detail=audit))
    for entry in router.audit.recent(5):
        print(f"        {entry.as_line()}")


async def _answer_with_tools(
    client: OllamaClient,
    memory: ConversationMemory,
    router: ToolRouter | None,
    ctx: ToolContext | None,
    tag: str,
    system_prompt: str,
    *,
    speaker: VoiceOutput | None,
    language: str,
    context: str = "",
) -> str:
    """Pełny przepływ tury: model → narzędzia → model → odpowiedź (Faza 7).

    Sama pętla (parsowanie wywołań, budżet, ostatnie przejście bez narzędzi)
    mieszka od Fazy 10 w :func:`brain.turn.run_turn` — wspólnie dla terminala i
    GUI. Tutaj zostaje wyłącznie terminalowy widok.
    """
    return await run_turn(
        client,
        memory,
        router,
        ctx,
        system_prompt,
        view=_TerminalView(tag),
        speaker=speaker,
        language=language,
        context=context,
    )


def run_reindex(settings: Settings) -> int:
    """Policz embeddingi dla całej pamięci i zakończ (``--reindex-memory``).

    Potrzebne po zmianie ``EMBEDDING_MODEL`` i po włączeniu pamięci semantycznej
    na bazie, która powstała wcześniej. Bez tego stare wspomnienia po prostu nie
    są kojarzone — wektory z różnych modeli nigdy nie są mieszane.
    """
    memory = ConversationMemory(settings, source="reindex")
    try:
        if not memory.persistent:
            print(f"{TAG_ERROR} " + t("cli.reindex.memory_unavailable", error=memory.error))
            return EXIT_MISSING_DEPENDENCIES
        if not memory.semantic_available:
            print(
                f"{TAG_ERROR} "
                + t("cli.reindex.semantic_unavailable", detail=memory.semantic_line())
            )
            print(f"{TAG_SYSTEM} " + t("cli.reindex.details"))
            return EXIT_MISSING_DEPENDENCIES

        print(f"{TAG_SYSTEM} Model: {memory.semantic_line()}")
        print(f"{TAG_SYSTEM} " + t("cli.reindex.computing"))
        started = time.monotonic()
        count = memory.reindex(progress=lambda table, done: print(f"        {table}: {done}"))
        elapsed = time.monotonic() - started
        print(f"{TAG_SYSTEM} " + t("cli.reindex.done", count=count, seconds=f"{elapsed:.1f}"))
        print(f"{TAG_SYSTEM} {memory.semantic_line()}")
        return EXIT_OK
    finally:
        memory.close()


def run_terminal(
    settings: Settings,
    report: DependencyReport,
    *,
    start_in_voice_mode: bool = False,
    speech_enabled: bool = True,
) -> int:
    """Pętla rozmowy w terminalu (wejście z klawiatury albo z mikrofonu)."""
    try:
        from brain.llm import LLMError, OllamaClient
    except ImportError as exc:
        print(
            f"{TAG_ERROR} Brakuje pakietu wymaganego do rozmowy z modelem ({exc}).\n"
            f"        Zainstaluj zależności: {pip_install_hint(report.offline)}\n"
            f"        albo: {install_instruction(report.platform_info)}"
        )
        return EXIT_MISSING_DEPENDENCIES

    _print_header(settings, report)

    if not report.ollama.reachable:
        print(f"{TAG_ERROR} " + t("cli.run.ollama_down", host=report.ollama.host))
        print(f"{TAG_SYSTEM} " + t("cli.run.ollama_hint"))
        print()
    elif not report.ollama.model_present:
        print(f"{TAG_ERROR} " + t("cli.run.model_missing", model=settings.ollama_model))
        print()

    # Pamięć obejmuje okno rozmowy: gdy bazy nie da się otworzyć, zostaje samo
    # okno w RAM i asystent działa dokładnie jak w Fazie 1.
    memory = ConversationMemory(settings, source="terminal")
    if settings.memory_enabled and not memory.persistent:
        print(f"{TAG_SYSTEM} " + t("cli.run.memory_unavailable", error=memory.error))
        print(f"{TAG_SYSTEM} " + t("cli.run.memory_note"))
        print()

    voice = VoiceInput(settings)
    if start_in_voice_mode:
        if voice.enable():
            print(f"{TAG_SYSTEM} " + t("cli.run.voice_on"))
        else:
            print(f"{TAG_SYSTEM} " + t("cli.run.typing"))
        print()

    # Mowa jest wyjściem opcjonalnym: gdy się nie uda, mówimy o tym raz i
    # rozmowa toczy się dalej tekstem. Cicha próba (quiet=True) nie zasypuje
    # ekranu komunikatem na maszynie, która po prostu nie ma głośnika.
    speaker = VoiceOutput(settings)
    if speech_enabled and speaker.enable(quiet=True):
        print(f"{TAG_SYSTEM} " + t("cli.run.speech_state", detail=speaker.status_text))
        print()
    elif speech_enabled and speaker.is_unavailable:
        print(f"{TAG_SYSTEM} " + t("cli.run.speech_unavailable", detail=speaker.status_text))
        print(f"{TAG_SYSTEM} " + t("cli.run.speech_hint"))
        print()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = OllamaClient(settings)
    curator = MemoryCurator(memory, settings)

    # Narzędzia (Faza 7). Router jest jedyną drogą od modelu do jakiegokolwiek
    # działania; bez niego (albo gdy TOOLS_ENABLED=false) asystent tylko rozmawia.
    plugins = _build_plugins(settings, memory)
    router, tool_ctx = _build_tools(settings, memory, plugins=plugins)
    if plugins is not None:
        print(f"{TAG_SYSTEM} " + t("cli.plugins.list", detail=plugins.describe()))
    if router is not None:
        print(f"{TAG_SYSTEM} " + t("cli.tools.list", detail=router.describe()))
        if not router.broker.available:
            print(f"{TAG_SYSTEM} " + t("cli.tools.no_confirm_channel"))
        print()

    try:
        while True:
            # Przypomnienia i inne powiadomienia pluginów — przed pytaniem o wejście.
            _print_plugin_notices(plugins, speak=voice.speak if hasattr(voice, "speak") else None)
            spoken_text = voice.listen() if voice.enabled else None

            if spoken_text is None:
                try:
                    raw_input_text = input(f"{TAG_USER} ")
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    break
                user_text = raw_input_text.strip()
            else:
                user_text = spoken_text.strip()
                print(f"{TAG_USER} {user_text}")

            if not user_text:
                continue

            lowered = user_text.lower()
            if lowered in _EXIT_COMMANDS:
                break
            if lowered in _HELP_COMMANDS:
                _print_help()
                continue
            if lowered in _CLEAR_COMMANDS:
                memory.new_session()
                print(
                    f"{TAG_SYSTEM} "
                    + t(
                        "cli.run.new_chat",
                        detail=(
                            t("cli.run.facts_stay")
                            if memory.persistent
                            else t("cli.run.memory_off")
                        ),
                    )
                )
                continue
            if lowered in _RELOAD_COMMANDS:
                user_settings = reload_user_settings()
                print(
                    f"{TAG_SYSTEM} "
                    + t(
                        "cli.run.reloaded",
                        name=user_settings.assistant_name,
                        tag=user_settings.log_tag,
                    )
                )
                continue
            if lowered in _DEPS_COMMANDS:
                report = detect_dependencies(settings)
                print_dependency_report(report)
                continue
            if lowered in _STATUS_COMMANDS:
                _print_status(settings, memory, voice, speaker, router)
                continue
            if lowered in _MEMORY_COMMAND_PREFIXES or lowered.startswith(
                tuple(f"{prefix} " for prefix in _MEMORY_COMMAND_PREFIXES)
            ):
                # Wartości faktów zachowują wielkość liter — to tekst użytkownika.
                _, _, argument = user_text.partition(" ")
                _handle_memory_command(argument, memory)
                continue
            if lowered in _VOICE_COMMAND_PREFIXES or lowered.startswith(
                tuple(f"{prefix} " for prefix in _VOICE_COMMAND_PREFIXES)
            ):
                # Nazwa głosu zachowuje wielkość liter — to nazwa pliku.
                _, _, argument = user_text.partition(" ")
                speaker.handle_command(argument)
                continue
            if lowered == _MIC_COMMAND_PREFIX or lowered.startswith(f"{_MIC_COMMAND_PREFIX} "):
                _handle_mic_command(lowered, voice)
                continue
            if lowered == _WAKE_COMMAND_PREFIX or lowered.startswith(f"{_WAKE_COMMAND_PREFIX} "):
                # Fraza zachowuje oryginalną wielkość liter — to tekst użytkownika.
                voice.handle_wake_command(user_text[len(_WAKE_COMMAND_PREFIX) :])
                continue
            if lowered in _TOOLS_COMMANDS:
                _print_tools(router, settings)
                continue

            user_settings = get_user_settings()
            tag = user_settings.log_tag
            # Ustawiony język wiąże (LANGUAGE=en → także pytanie po polsku dostaje
            # odpowiedź po angielsku); rozpoznawanie wchodzi tylko przy LANGUAGE=auto.
            preferred = _preferred_language(settings)
            language = resolve_reply_language(preferred, user_text)
            lock_language = not is_auto_language(preferred)

            memory.add_user(user_text, language=language)

            # „Zapamiętaj, że…" / „Zapomnij, że…" (Faza 6). Rozpoznanie jest
            # czysto tekstowe, więc działa też przy niedostępnym modelu; model
            # ocenia jedynie, czy informacja jest trwała, czy chwilowa.
            intent = detect_memory_intent(user_text)
            if intent is not None:
                outcome = _handle_memory_intent(
                    loop, curator, intent, client, memory, tag, language, speaker
                )
                if outcome:
                    continue

            # Streszczenie tego, co wypadło z okna rozmowy. Zwykle nie ma czego
            # streszczać i wywołanie kończy się natychmiast; gdy jest — model
            # dostaje treść starszych tur zamiast ich braku.
            if memory.needs_compaction:
                print(f"{TAG_SYSTEM} " + t("cli.run.compacting"))
                try:
                    loop.run_until_complete(memory.compact(client, language=language))
                except KeyboardInterrupt:
                    print(f"{TAG_SYSTEM} " + t("cli.run.compact_interrupted"))
                except Exception:  # streszczenie jest dodatkiem, nie warunkiem rozmowy
                    logger.exception("Streszczanie rozmowy nie powiodło się")

            # Nowa tura = nowy budżet wywołań narzędzi i czysty ślad danych
            # niezaufanych (Faza 7).
            if router is not None:
                router.reset_turn(conversation_id=memory.conversation_id)

            # LOCAL czy WEB (Faza 9): czy pytanie wymaga świeżych danych. Ocena jest
            # czysto tekstowa, więc nie kosztuje tury ani nie zależy od Ollamy.
            assessment = classify_request(user_text)
            web_ready = _web_tools_ready(router)
            logger.info(
                "Rodzaj pytania: %s (%s), narzędzia sieciowe: %s",
                assessment.kind.value,
                assessment.describe(),
                "dostępne" if web_ready else "niedostępne",
            )
            notice = request_notice(assessment, language=language, web_available=web_ready)
            if notice:
                # Jedno zdanie do przeczytania i do wypowiedzenia: pytanie wymaga
                # internetu, którego teraz nie ma. Rozmowa toczy się dalej.
                print(f"{TAG_SYSTEM} {notice}")
                _say_line(speaker, notice, language)

            # Kontekst dostaje bieżące pytanie: dzięki temu do promptu trafiają
            # wspomnienia podobne ZNACZENIEM, a nie tylko fakty i streszczenia.
            # Prompt systemowy jest STAŁY między turami (patrz brain/llm.py),
            # a wszystko, co zmienne, idzie osobną wiadomością na końcu.
            system_prompt = build_system_prompt(
                user_settings,
                language=language,
                lock_language=lock_language,
                tool_rules=(
                    tool_system_rules(language) if router is not None and router.enabled else ""
                ),
            )
            turn_context = build_context_message(
                language=language,
                extra_context=memory.context_block(language, query=user_text),
                request_hint=request_prompt_hint(
                    assessment, language=language, web_available=web_ready
                ),
            )

            task = loop.create_task(
                _answer_with_tools(
                    client,
                    memory,
                    router,
                    tool_ctx.localized(language) if tool_ctx is not None else None,
                    tag,
                    system_prompt,
                    speaker=speaker,
                    language=language,
                    context=turn_context,
                )
            )
            try:
                answer = loop.run_until_complete(task)
            except KeyboardInterrupt:
                task.cancel()
                speaker.cancel()
                with contextlib.suppress(BaseException):
                    loop.run_until_complete(task)
                print()
                print(f"{TAG_SYSTEM} " + t("cli.run.generation_interrupted"))
                continue
            except LLMError as exc:
                print(f"{TAG_ERROR} {exc.user_message}")
                continue
            except Exception as exc:  # nieoczekiwany błąd — bez stack trace'u dla użytkownika
                logger.exception("Nieoczekiwany błąd podczas rozmowy")
                print(f"{TAG_ERROR} " + t("cli.run.unexpected", error=exc))
                print(f"{TAG_ERROR} " + t("cli.voice.details_in", path=ERROR_LOG_FILE))
                continue

            if answer.strip():
                memory.add_assistant(answer, language=language)
            else:
                print(f"{TAG_SYSTEM} " + t("cli.run.empty_reply"))
    finally:
        voice.close()
        speaker.close()
        memory.close()
        # Kolejność jak w asyncio.run(): zamknij klienta, dokończ generatory
        # asynchroniczne, dopiero potem zamknij pętlę. Bez shutdown_asyncgens()
        # niedokończony strumień odpowiedzi daje "Task was destroyed but it is
        # pending!" przy wyjściu z programu.
        with contextlib.suppress(Exception):
            loop.run_until_complete(client.aclose())
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_default_executor())
        with contextlib.suppress(Exception):
            loop.close()
        asyncio.set_event_loop(None)

    print(f"{TAG_SYSTEM} " + t("cli.run.goodbye"))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Tryb bezobsługowy (--headless): usługa w tle, bez klawiatury i bez okna
# --------------------------------------------------------------------------- #


def _install_stop_handlers(stop: threading.Event) -> Callable[[], None]:
    """Podłącz sygnały zamknięcia. Zwraca funkcję przywracającą poprzedni stan.

    ``systemd`` zatrzymuje usługę SIGTERM-em, a ``Ctrl+C`` w terminalu daje
    SIGINT. Windows nie zna SIGTERM-a w tym samym znaczeniu, ale zna SIGBREAK
    (``Ctrl+Break`` i zatrzymanie zadania) — dlatego lista sygnałów jest budowana
    z tego, co na tej maszynie faktycznie istnieje, a nie z założeń.

    Obsługa jest celowo minimalna: ustawia zdarzenie i wraca. Zamykanie potoku
    audio i bazy dzieje się w normalnym ``finally`` pętli, a nie w przerwaniu —
    w handlerze sygnału nie wolno robić niczego, co może się zablokować.
    """
    previous: list[tuple[int, Any]] = []

    def _on_signal(signum: int, _frame: Any) -> None:
        logger.info("Otrzymano sygnał %s — kończę pracę", signum)
        stop.set()

    for name in ("SIGTERM", "SIGINT", "SIGBREAK", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous.append((int(number), signal.signal(number, _on_signal)))
        except (ValueError, OSError, RuntimeError):  # pragma: no cover - zależne od systemu
            # `signal.signal` działa tylko w wątku głównym i tylko dla sygnałów,
            # które ten system obsługuje. Brak któregoś nie jest błędem.
            logger.debug("Nie udało się podłączyć obsługi %s", name)

    def restore() -> None:
        for number, handler in previous:
            with contextlib.suppress(ValueError, OSError, RuntimeError, TypeError):
                signal.signal(number, handler)

    return restore


def _headless_wait_for_ollama(
    client: OllamaClient,
    stop: threading.Event,
    *,
    loop: asyncio.AbstractEventLoop,
    timeout_s: float,
) -> bool:
    """Poczekaj, aż serwer modelu zacznie odpowiadać (albo aż minie limit).

    W trybie usługi asystent startuje razem z sesją użytkownika, często ZANIM
    Ollama zdąży się podnieść. Bez tego czekania pierwsze pytanie kończyłoby się
    komunikatem o braku połączenia mimo poprawnej konfiguracji.

    ``loop`` jest przekazywany, a NIE tworzony tutaj, i to nie jest szczegół
    stylu. ``httpx.AsyncClient`` wiąże pulę połączeń z pętlą, na której je
    otworzył; sprawdzenie dostępności na własnej pętli, a potem rozmowa na
    innej, kończy się błędem „Event loop is closed" przy pierwszym pytaniu.
    Do tego ``asyncio.set_event_loop(None)`` w bloku ``finally`` zdejmowałoby
    pętlę ustawioną przez wywołującego.

    Brak serwera po limicie NIE jest błędem krytycznym: usługa działa dalej i
    spróbuje jeszcze raz przy pierwszej wypowiedzi. Zamknięcie w tym czasie
    (``stop``) przerywa czekanie natychmiast.
    """
    if timeout_s <= 0:
        return False
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    while not stop.is_set():
        try:
            if loop.run_until_complete(client.is_available()):
                return True
        except Exception:  # pragma: no cover - dowolna awaria sieci = próbuj dalej
            logger.debug("Sprawdzenie dostępności modelu nie powiodło się", exc_info=True)
        if time.monotonic() >= deadline:
            return False
        # Czekamy na zdarzeniu, a nie w `sleep` — SIGTERM w tej fazie ma zamknąć
        # usługę od razu, a nie dopiero po dopełnieniu odstępu.
        stop.wait(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 2, 10.0)
    return False


class _HeadlessView:
    """Kanał wyjściowy tury dla usługi: wszystko do logu, nic do terminala.

    W usłudze `stdout` trafia do dziennika systemu (journald, Podgląd zdarzeń
    albo plik), więc strumieniowanie odpowiedzi po jednym fragmencie zalałoby go
    tysiącami linii. Zapisujemy więc jedno zdanie na turę, a użytkownik i tak
    słyszy odpowiedź na głos.
    """

    def __init__(self) -> None:
        self.answer = ""

    def on_thinking(self) -> None:
        logger.debug("Model liczy odpowiedź")

    def on_reply_start(self) -> None:
        pass

    def on_chunk(self, chunk: str) -> None:
        pass

    def on_reply_end(self, answer: str) -> None:
        self.answer = answer
        logger.info("Odpowiedź (%d znaków): %s", len(answer), _one_line(answer, limit=200))

    def on_tool(self, outcome: Any) -> None:
        logger.info("Narzędzie: %s", _one_line(outcome.line_for_user(), limit=200))

    def on_notice(self, text: str) -> None:
        logger.info("%s", text)


def _one_line(text: str, *, limit: int = 200) -> str:
    """Tekst nadający się do jednej linii dziennika."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def run_headless(
    settings: Settings,
    report: DependencyReport,
    *,
    speech_enabled: bool = True,
    stop: threading.Event | None = None,
    max_turns: int = 0,
) -> int:
    """Pętla usługi: mikrofon → model → głośnik, bez klawiatury i bez okna.

    Ten tryb istnieje po to, żeby asystenta dało się uruchomić z ``systemd
    --user`` na Linuksie i z Harmonogramu zadań na Windowsie. Różnice wobec
    :func:`run_terminal` nie są kosmetyczne:

    * **nigdy nie woła ``input()``** — w usłudze ``stdin`` jest zamknięty albo
      wskazuje ``/dev/null``. Wywołanie ``input()`` skończyłoby się ``EOFError``
      w pętli i procesem kręcącym się w kółko na 100% CPU,
    * **wejście głosowe jest WARUNKIEM startu**, a nie dodatkiem: usługa bez
      mikrofonu nie ma jak przyjąć polecenia, więc kończy się jasnym kodem
      wyjścia zamiast czekać w nieskończoność,
    * **potwierdzenia narzędzi są odrzucane automatycznie**. Nie ma komu zadać
      pytania, więc HIGH i CRITICAL nie wykonują się nigdy. To jest świadomy
      wybór: „brak odpowiedzi" znaczy „nie", nigdy „tak",
    * **SIGTERM zamyka pracę czysto** — mikrofon, głośnik i baza są zamykane w
      ``finally``, więc dziennik nie kończy się stack trace'em przy każdym
      ``systemctl stop``.

    ``stop`` pozwala zatrzymać pętlę z zewnątrz (używa tego test i obsługa
    sygnałów). ``max_turns`` > 0 ogranicza liczbę obrotów — wyłącznie dla testów.
    """
    try:
        from brain.llm import LLMError, OllamaClient
    except ImportError as exc:
        print(
            f"{TAG_ERROR} Brakuje pakietu wymaganego do rozmowy z modelem ({exc}).\n"
            f"        Zainstaluj zależności: {pip_install_hint(report.offline)}",
            file=sys.stderr,
        )
        return EXIT_MISSING_DEPENDENCIES

    stop_event = stop if stop is not None else threading.Event()
    restore_signals = _install_stop_handlers(stop_event) if stop is None else (lambda: None)

    user_settings = get_user_settings()
    print(f"{TAG_SYSTEM} " + t("cli.headless.starting", name=user_settings.assistant_name))
    logger.info(
        "Tryb bezobsługowy: model %s, mikrofon %s, mowa %s",
        settings.ollama_model,
        settings.audio_input_device or "domyślny",
        "włączona" if speech_enabled else "wyłączona",
    )

    # --- wejście głosowe: bez niego usługa nie ma sensu -------------------- #
    voice = VoiceInput(settings)
    if not voice.enable():
        print(f"{TAG_ERROR} " + t("cli.headless.no_microphone"), file=sys.stderr)
        print(f"{TAG_SYSTEM} " + t("cli.headless.no_microphone_hint"), file=sys.stderr)
        voice.close()
        restore_signals()
        return EXIT_MISSING_DEPENDENCIES

    speaker = VoiceOutput(settings)
    if speech_enabled and not speaker.enable(quiet=True):
        # Głos jest dodatkiem — usługa bez głośnika nadal wykonuje polecenia i
        # zapisuje odpowiedzi do dziennika.
        print(f"{TAG_SYSTEM} " + t("cli.headless.no_speech", detail=speaker.status_text))

    memory = ConversationMemory(settings, source="headless")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = OllamaClient(settings)
    curator = MemoryCurator(memory, settings)

    # Potwierdzenia: jawnie AutoDenyBroker zamiast `default_broker()`. Wynik jest
    # ten sam, ale zapisany wprost — czytający kod nie musi zgadywać, czy usługa
    # przypadkiem nie odziedziczy interaktywnego terminala, gdy ktoś uruchomi ją
    # z konsoli do testów.
    broker = AutoDenyBroker(reason=t("cli.headless.deny_reason"))
    plugins = _build_plugins(settings, memory)
    router, tool_ctx = _build_tools(settings, memory, plugins=plugins, broker=broker)
    if router is not None:
        logger.info("Narzędzia: %s", router.describe())
        print(f"{TAG_SYSTEM} " + t("cli.headless.confirmations_denied"))

    if _headless_wait_for_ollama(
        client, stop_event, loop=loop, timeout_s=settings.headless_ollama_wait_s
    ):
        logger.info("Serwer modelu odpowiada")
    elif not stop_event.is_set():
        print(f"{TAG_SYSTEM} " + t("cli.run.ollama_down", host=report.ollama.host))

    turns = 0
    exit_code = EXIT_OK
    try:
        if speech_enabled and speaker.enabled and settings.headless_greeting:
            _say_line(speaker, greeting(user_settings, language=_preferred_language(settings)))

        while not stop_event.is_set():
            if max_turns > 0 and turns >= max_turns:
                break
            turns += 1

            mowa = _speak_notice(speaker) if speaker.enabled else None
            _print_plugin_notices(plugins, speak=mowa)

            if not voice.enabled:
                # Nasłuch padł (odłączony mikrofon, awaria sterownika). Usługa nie
                # ma dokąd się przełączyć, więc próbuje wstać z odstępem zamiast
                # kręcić się w pętli albo umierać.
                if stop_event.wait(settings.headless_retry_s):
                    break
                if not voice.enable():
                    logger.warning("Nasłuch nadal niedostępny — spróbuję ponownie za %.0f s",
                                   settings.headless_retry_s)
                    continue
                print(f"{TAG_SYSTEM} " + t("cli.headless.microphone_back"))

            spoken = voice.listen(timeout_s=settings.headless_listen_slice_s, quiet=True)
            if stop_event.is_set():
                break
            if spoken is None:
                continue
            user_text = spoken.strip()
            if not user_text:
                continue

            print(f"{TAG_USER} {user_text}")
            tag = get_user_settings().log_tag
            preferred = _preferred_language(settings)
            language = resolve_reply_language(preferred, user_text)
            lock_language = not is_auto_language(preferred)

            memory.add_user(user_text, language=language)

            intent = detect_memory_intent(user_text)
            if intent is not None and _handle_memory_intent(
                loop, curator, intent, client, memory, tag, language, speaker
            ):
                continue

            if memory.needs_compaction:
                try:
                    loop.run_until_complete(memory.compact(client, language=language))
                except Exception:  # streszczenie jest dodatkiem, nie warunkiem rozmowy
                    logger.exception("Streszczanie rozmowy nie powiodło się")

            if router is not None:
                router.reset_turn(conversation_id=memory.conversation_id)

            assessment = classify_request(user_text)
            web_ready = _web_tools_ready(router)
            notice = request_notice(assessment, language=language, web_available=web_ready)
            if notice:
                logger.info("%s", notice)
                _say_line(speaker, notice, language)

            system_prompt = build_system_prompt(
                get_user_settings(),
                language=language,
                lock_language=lock_language,
                tool_rules=(
                    tool_system_rules(language) if router is not None and router.enabled else ""
                ),
            )
            turn_context = build_context_message(
                language=language,
                extra_context=memory.context_block(language, query=user_text),
                request_hint=request_prompt_hint(
                    assessment, language=language, web_available=web_ready
                ),
            )

            try:
                answer = loop.run_until_complete(
                    run_turn(
                        client,
                        memory,
                        router,
                        tool_ctx.localized(language) if tool_ctx is not None else None,
                        system_prompt,
                        view=_HeadlessView(),
                        speaker=speaker,
                        language=language,
                        context=turn_context,
                    )
                )
            except LLMError as exc:
                logger.error("Model niedostępny: %s", exc.message)
                _say_line(speaker, exc.message, language)
                continue
            except Exception as exc:
                logger.exception("Nieoczekiwany błąd tury")
                logger.error("Szczegóły w %s (%s)", ERROR_LOG_FILE, exc)
                continue

            if answer.strip():
                memory.add_assistant(answer, language=language)
    except KeyboardInterrupt:  # pragma: no cover - przerwanie z terminala
        logger.info("Przerwano z klawiatury")
    finally:
        restore_signals()
        voice.close()
        speaker.close()
        memory.close()
        with contextlib.suppress(Exception):
            loop.run_until_complete(client.aclose())
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_default_executor())
        with contextlib.suppress(Exception):
            loop.close()
        asyncio.set_event_loop(None)

    print(f"{TAG_SYSTEM} " + t("cli.headless.stopped"))
    logger.info("Tryb bezobsługowy zakończony po %d obrotach", turns)
    return exit_code


def _speak_notice(speaker: VoiceOutput) -> Callable[[str], None]:
    """Funkcja wypowiadająca powiadomienie pluginu (przypomnienie) w usłudze."""

    def speak(text: str) -> None:
        _say_line(speaker, text)

    return speak


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=t("cli.arg.description"),
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help=t("cli.arg.terminal"),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help=t("cli.arg.gui"),
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help=t("cli.arg.no_gui"),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=t("cli.arg.headless"),
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help=t("cli.arg.check_deps"),
    )
    parser.add_argument(
        "--audio-check",
        action="store_true",
        help=t("cli.arg.audio_check"),
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help=t("cli.arg.voice"),
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help=t("cli.arg.no_voice"),
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help=t("cli.arg.no_wake"),
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help=t("cli.arg.no_tts"),
    )
    parser.add_argument(
        "--voice-test",
        nargs="?",
        const="",
        default=None,
        metavar=t("cli.arg.metavar_text"),
        help=t("cli.arg.voice_test"),
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help=t("cli.arg.list_voices"),
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help=t("cli.arg.no_memory"),
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help=t("cli.arg.no_embeddings"),
    )
    parser.add_argument(
        "--reindex-memory",
        action="store_true",
        help=t("cli.arg.reindex_memory"),
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help=t("cli.arg.no_tools"),
    )
    parser.add_argument(
        "--dry-run-tools",
        action="store_true",
        help=t("cli.arg.dry_run_tools"),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=t("cli.arg.offline"),
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help=t("cli.arg.online"),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=t("cli.arg.log_level"),
    )
    parser.add_argument(
        "--ui-lang",
        default=None,
        metavar="KOD",
        help=t("cli.arg.ui_lang"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APP_VERSION}",
    )
    return parser


def _ensure_ollama(settings: Settings) -> None:
    """Uruchom lokalną Ollamę, jeśli nie działa. Nigdy nie blokuje startu."""
    if not settings.ollama_autostart:
        return
    try:
        from host.ollama import ensure_running
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.debug("Warstwa uruchamiania Ollamy niedostępna: %s", exc)
        return
    try:
        result = ensure_running(settings)
    except Exception as exc:  # pragma: no cover - nic tu nie może wywrócić startu
        logger.warning("Nie udało się sprawdzić usługi Ollamy: %s", exc)
        return
    if result.message:
        tag = TAG_SYSTEM if result.ok else TAG_ERROR
        print(f"{tag} {result.message}")
        if result.hint:
            print(f"{TAG_SYSTEM} {result.hint}")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_streams()
    # Język interfejsu musi być znany przed zbudowaniem parsera: `--help` też
    # jest tekstem dla człowieka.
    _bootstrap_ui_language(argv)
    args = build_parser().parse_args(argv)

    ensure_directories()

    if args.offline and args.online:
        print(f"{TAG_ERROR} " + t("cli.main.flags_conflict"))
        return EXIT_CONFIG_ERROR
    if args.gui and args.no_gui:
        print(f"{TAG_ERROR} " + t("cli.main.gui_conflict"))
        return EXIT_CONFIG_ERROR
    if args.headless and (args.gui or args.terminal):
        # --headless to usługa bez interfejsu: okno i interaktywny terminal
        # wykluczają się z nim wprost, a milczące zignorowanie jednej z flag
        # kończyłoby się usługą robiącą coś innego, niż napisano w pliku unitu.
        print(f"{TAG_ERROR} " + t("cli.main.headless_conflict"))
        return EXIT_CONFIG_ERROR
    if args.headless and args.no_voice:
        print(f"{TAG_ERROR} " + t("cli.main.headless_needs_voice"))
        return EXIT_CONFIG_ERROR
    if args.ui_lang:
        os.environ["UI_LANGUAGE"] = args.ui_lang
    # Zmienna środowiskowa ma pierwszeństwo nad .env w pydantic-settings, więc
    # flaga z wiersza poleceń wygrywa z plikiem konfiguracyjnym.
    if args.offline:
        os.environ["OFFLINE_MODE"] = "on"
    elif args.online:
        os.environ["OFFLINE_MODE"] = "off"
    if args.no_wake:
        os.environ["WAKE_ENABLED"] = "false"
    if args.no_tts:
        os.environ["TTS_ENABLED"] = "false"
    if args.no_memory:
        os.environ["MEMORY_ENABLED"] = "false"
    if args.no_embeddings:
        os.environ["EMBEDDINGS_ENABLED"] = "false"
    if args.no_tools:
        os.environ["TOOLS_ENABLED"] = "false"
    if args.dry_run_tools:
        os.environ["SECURITY_DRY_RUN"] = "true"

    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"{TAG_ERROR} {exc.user_message}")
        return EXIT_CONFIG_ERROR

    # Ustawienia znają już plik .env, więc dopiero teraz rozstrzygamy „auto" —
    # przy UI_LANGUAGE=auto interfejs idzie za językiem odpowiedzi.
    set_ui_language(settings.ui_language, reply_language=settings.language)

    # Zanim cokolwiek zaimportuje huggingface_hub: w trybie offline blokujemy
    # wszystkie próby wyjścia do sieci, zawsze — kierujemy cache modeli do
    # katalogu projektu i wypinamy proxy dla adresów lokalnych (Ollama).
    offline = apply_offline_environment(settings)

    try:
        setup_logging(settings, level_override=args.log_level)
    except Exception as exc:  # logowanie nigdy nie może zablokować startu
        print(f"{TAG_ERROR} " + t("cli.main.logging_failed", error=exc))

    logger.info(
        "Start aplikacji, wersja %s, tryb %s",
        APP_VERSION,
        "offline" if offline else "online",
    )

    # Rejestracja sprawdzeń Fazy 2 w tym samym mechanizmie co Faza 1. Import
    # jest opcjonalny: brak warstwy audio ma tylko usunąć jej pozycje z raportu.
    try:
        import audio.dependencies  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie zarejestrowano sprawdzeń audio: %s", exc)

    # To samo dla pamięci długoterminowej (Faza 5). Import NIE otwiera bazy —
    # sprawdzenie tylko ogląda katalog i możliwości biblioteki SQLite.
    try:
        import database  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie zarejestrowano sprawdzeń pamięci: %s", exc)

    # Pamięć semantyczna (Faza 6). Import NIE ładuje modelu embeddingów —
    # sprawdzenie ogląda tylko pakiety, katalog modeli i stan indeksu w bazie.
    try:
        import brain.dependencies  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie zarejestrowano sprawdzeń pamięci semantycznej: %s", exc)

    # Narzędzia (Faza 7). Import buduje rejestr, ale NIE wywołuje niczego —
    # sprawdzenie mówi tylko, co model zobaczy i czy jest komu zadać pytanie
    # o zgodę na narzędzia o wyższym ryzyku.
    try:
        import tools.dependencies  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie zarejestrowano sprawdzeń narzędzi: %s", exc)

    # Pluginy (Faza 11). Sprawdzenie ładuje moduły pluginów i pyta je o
    # dostępność — nie wywołuje ich narzędzi i nie łączy się z niczym.
    try:
        import plugins.dependencies  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od zawartości plugins/
        logger.warning("Nie zarejestrowano sprawdzeń pluginów: %s", exc)

    # Interfejs graficzny (Faza 10). Import NIE wciąga tkintera — sprawdzenie tylko
    # próbuje go zaimportować i mówi, czego brakuje: pakietu, biblioteki systemowej
    # Tk czy sesji graficznej. Na maszynie bez Tk raport nadal się liczy.
    try:
        import gui.dependencies  # noqa: F401 - import rejestruje sprawdzenia
    except Exception as exc:  # pragma: no cover - zależne od instalacji
        logger.warning("Nie zarejestrowano sprawdzeń GUI: %s", exc)

    # Ollama: uruchamiamy ją SAMI, zanim policzymy raport — inaczej użytkownik
    # musiałby trzymać drugie okno terminala tylko po to, żeby w nim stała usługa.
    _ensure_ollama(settings)

    try:
        report = detect_dependencies(settings)
    except Exception as exc:
        logger.exception("Detekcja zależności nie powiodła się")
        print(f"{TAG_ERROR} " + t("cli.main.deps_failed", error=exc))
        return EXIT_CONFIG_ERROR

    if args.check_deps:
        print_dependency_report(report)
        return EXIT_OK if report.ok else EXIT_MISSING_DEPENDENCIES

    if args.audio_check:
        return run_audio_check(settings)

    if args.reindex_memory:
        return run_reindex(settings)

    if args.list_voices:
        VoiceOutput(settings).list_voices()
        return EXIT_OK

    if args.voice_test is not None:
        speaker = VoiceOutput(settings)
        try:
            if not speaker.enable():
                print(f"{TAG_SYSTEM} " + t("cli.main.speech_report"))
                return EXIT_MISSING_DEPENDENCIES
            speaker.say_sample(args.voice_test)
        finally:
            speaker.close()
        return EXIT_OK

    if report.missing_required:
        print()
        print(f"{TAG_ERROR} " + t("cli.main.missing_required"))
        for check in report.missing_required:
            hint = f" → {check.hint}" if check.hint else ""
            print(f"        - {check.name}: {check.detail}{hint}")
        print(
            f"{TAG_ERROR} "
            + t(
                "cli.main.nothing_automatic",
                command=install_instruction(report.platform_info),
            )
        )
        print(f"{TAG_SYSTEM} " + t("cli.main.full_report"))
        print(f"{TAG_SYSTEM} " + t("cli.main.starting_anyway"))

    if args.headless:
        # Wejście głosowe jest jedynym wejściem usługi — INPUT_MODE=text nie ma
        # tu żadnego sensu, więc nie pytamy o nie ustawień.
        return run_headless(settings, report, speech_enabled=settings.tts_enabled)

    start_in_voice_mode = (args.voice or settings.input_mode == "voice") and not args.no_voice
    # Okno albo terminal: flaga wygrywa z ustawieniem, a --no-gui wygrywa ze
    # wszystkim (tak jak --no-voice przy INPUT_MODE=voice).
    use_gui = (args.gui or settings.gui_enabled) and not args.no_gui

    if use_gui:
        # Import lokalny: pakiet gui wciąga CustomTkintera dopiero w run_gui, więc
        # brak Tk nie przeszkadza w niczym innym (--check-deps, --terminal).
        from gui import run_gui, toolkit_status

        status = toolkit_status()
        if not status.ok and not args.gui:
            # Okno jest DOMYŚLNE, ale nie obowiązkowe: gdy na tej maszynie nie ma
            # Tk albo sesji graficznej, mówimy o tym jednym zdaniem i pracujemy
            # dalej w terminalu. Wprost poproszone `--gui` kończy się błędem, bo
            # wtedy użytkownik czeka na okno, a nie na rozmowę w konsoli.
            print(f"{TAG_SYSTEM} " + t("cli.gui_fallback", reason=status.detail))
            if status.hint:
                print(f"{TAG_SYSTEM} {status.hint}")
        else:
            return run_gui(
                settings,
                report,
                speech_enabled=settings.tts_enabled,
                start_in_voice_mode=start_in_voice_mode,
            )

    return run_terminal(
        settings,
        report,
        start_in_voice_mode=start_in_voice_mode,
        speech_enabled=settings.tts_enabled,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover - przerwanie przez użytkownika
        print()
        raise SystemExit(EXIT_OK) from None
