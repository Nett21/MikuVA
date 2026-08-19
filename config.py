"""Konfiguracja i detekcja środowiska asystenta (Faza 1).

Ten moduł jest JEDYNYM miejscem w projekcie, które:

* wie, na jakim systemie operacyjnym i sprzęcie działamy,
* wyznacza ścieżki na dysku,
* uruchamia procesy zewnętrzne (``nvidia-smi``) i odpytuje usługi (Ollama).

Reszta kodu korzysta wyłącznie z funkcji i modeli wystawionych tutaj i nigdy
nie sprawdza samodzielnie ``sys.platform`` ani nie skleja ścieżek ze stringów.

Dwie warstwy konfiguracji:

* ``.env``                       -> :class:`Settings`      (infrastruktura, rzadko zmieniane)
* ``config/user_settings.json``  -> :class:`UserSettings`  (ustawienia użytkownika/GUI)
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import ipaddress
import json
import logging
import os
import platform as _stdlib_platform
import re
import shutil
import socket
import subprocess  # nosec B404 - używane wyłącznie do wywołania nvidia-smi bez powłoki
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from i18n import t, translate_or_text

logger = logging.getLogger(__name__)

APP_NAME: Final[str] = "miku-assistant"
APP_VERSION: Final[str] = "0.1.0"
REQUIRED_PYTHON: Final[tuple[int, int]] = (3, 12)

# --------------------------------------------------------------------------- #
# Ścieżki — wszystko względem TEGO pliku, nigdy względem katalogu roboczego.
# Każdą można nadpisać zmienną środowiskową (przydatne przy instalacji
# systemowej, gdzie katalog programu bywa tylko do odczytu).
# --------------------------------------------------------------------------- #


def _path_from_env(variable: str) -> Path | None:
    """Zwróć ścieżkę ze zmiennej środowiskowej albo ``None``, jeśli jej nie ma."""
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return Path(os.path.expandvars(raw)).expanduser()


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
CONFIG_DIR: Final[Path] = _path_from_env("MIKU_CONFIG_DIR") or PROJECT_ROOT / "config"
LOGS_DIR: Final[Path] = _path_from_env("MIKU_LOGS_DIR") or PROJECT_ROOT / "logs"
MODELS_DIR: Final[Path] = _path_from_env("MIKU_MODELS_DIR") or PROJECT_ROOT / "models"
ENV_FILE: Final[Path] = _path_from_env("MIKU_ENV_FILE") or PROJECT_ROOT / ".env"

WHISPER_CACHE_DIR: Final[Path] = MODELS_DIR / "whisper"
# Modele słowa aktywującego (Faza 3) — trzymane w projekcie z tego samego
# powodu co modele Whispera: przenoszą się razem z katalogiem.
WAKEWORD_DIR: Final[Path] = MODELS_DIR / "wakeword"
# Głosy Piper (Faza 4): pliki .onnx wraz z .onnx.json. Katalog w projekcie jest
# tylko PIERWSZYM z przeszukiwanych miejsc — pełną listę (razem z katalogami
# systemowymi wykrytymi dla danej platformy) zwraca piper_voice_directories().
PIPER_DIR: Final[Path] = MODELS_DIR / "piper"
# Model embeddingów (Faza 6) — pamięć semantyczna. Osobny katalog od Whispera,
# bo to inny model i inny cykl życia; też w projekcie, żeby przenosił się razem
# z nim i nie lądował w ``~/.cache``.
EMBEDDINGS_DIR: Final[Path] = MODELS_DIR / "embeddings"

# Lokalny magazyn kół pip (``pip download``) — pozwala zainstalować zależności
# na maszynie bez internetu: pip install --no-index --find-links vendor/wheels
WHEELHOUSE_DIR: Final[Path] = (
    _path_from_env("MIKU_WHEELHOUSE_DIR") or PROJECT_ROOT / "vendor" / "wheels"
)

USER_SETTINGS_FILE: Final[Path] = CONFIG_DIR / "user_settings.json"
USER_SETTINGS_EXAMPLE_FILE: Final[Path] = CONFIG_DIR / "user_settings.example.json"
DEPENDENCY_STATUS_FILE: Final[Path] = CONFIG_DIR / "dependency_status.json"
# Manifest zapisywany przez scripts/prepare_offline.py — czysta diagnostyka.
OFFLINE_BUNDLE_FILE: Final[Path] = CONFIG_DIR / "offline_bundle.json"

LOG_FILE: Final[Path] = LOGS_DIR / "assistant.log"
ERROR_LOG_FILE: Final[Path] = LOGS_DIR / "errors.log"

# Pamięć długoterminowa (Faza 5). Nazwa pliku jest stała, KATALOG — nie: wyznacza
# go app_data_directory() na podstawie systemu, na którym program właśnie działa.
DEFAULT_DATABASE_NAME: Final[str] = "miku.sqlite3"
# Umowna „ścieżka" bazy trzymanej wyłącznie w pamięci procesu (sqlite3).
MEMORY_DATABASE: Final[str] = ":memory:"

# Tagi używane w terminalu. Tag odpowiedzi asystenta jest budowany dynamicznie
# z ``UserSettings.display_tag`` — patrz :attr:`UserSettings.display_tag`.
TAG_MIC: Final[str] = "[MIC]"
TAG_WAKE: Final[str] = "[WAKE]"
TAG_USER: Final[str] = "[USER]"
TAG_TOOL: Final[str] = "[TOOL]"
TAG_ERROR: Final[str] = "[ERROR]"
TAG_SYSTEM: Final[str] = "[SYSTEM]"


class ConfigError(RuntimeError):
    """Błąd konfiguracji nadający się do pokazania użytkownikowi."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


def ensure_directories() -> None:
    """Utwórz katalogi robocze. Brak uprawnień jest logowany, nie wysadza programu."""
    for directory in (CONFIG_DIR, LOGS_DIR, MODELS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - zależne od uprawnień systemu
            logger.warning("Nie udało się utworzyć katalogu %s: %s", directory, exc)


# --------------------------------------------------------------------------- #
# Warstwa 1: .env -> Settings (infrastruktura)
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Ustawienia infrastrukturalne czytane z ``.env`` i zmiennych środowiskowych."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- Tryb offline ---
    # on   = nic w programie nie sięga do internetu (twarda gwarancja),
    # off  = wolno pobierać brakujące modele,
    # auto = offline, gdy komplet zasobów jest już lokalnie; inaczej online.
    # Wartości logiczne (true/false/1/0/yes/no) są tłumaczone na on/off.
    offline_mode: Literal["auto", "on", "off"] = "auto"

    # --- LLM (Ollama) ---
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    # Czy asystent ma sam uruchomić lokalną Ollamę, gdy nie odpowiada. Dzięki
    # temu nie trzeba trzymać drugiego okna terminala tylko po to, by w nim stało
    # `ollama serve`. Nigdy nie dotyczy serwera na innej maszynie.
    ollama_autostart: bool = True
    ollama_keep_alive: str = "10m"
    ollama_num_ctx: int = Field(default=8192, ge=512, le=131_072)
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_max_tokens: int = Field(default=1024, ge=-1, le=32_768)
    ollama_connect_timeout: float = Field(default=5.0, gt=0.0, le=120.0)
    ollama_read_timeout: float = Field(default=120.0, gt=0.0, le=3600.0)

    # --- STT (Faza 2) ---
    whisper_model: str = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = "auto"
    # pusty łańcuch = automatyczne rozpoznanie języka przez Whisper
    whisper_language: str = ""
    whisper_beam_size: int = Field(default=5, ge=1, le=10)
    whisper_allow_download: bool = True
    whisper_min_duration_s: float = Field(default=0.35, ge=0.0, le=10.0)
    whisper_max_no_speech_prob: float = Field(default=0.75, ge=0.0, le=1.0)
    # Po ilu sekundach ciszy zwolnić model Whispera z pamięci (i z VRAM-u, gdy
    # liczy na GPU). Model wczytany „na wszelki wypadek" nie zużywa cykli, ale
    # trzyma kilkaset MB — na laptopie z 8 GB i jednym GPU to jest różnica
    # między działającą grą a swapem. Ponowne wczytanie kosztuje ok. 1–3 s,
    # więc próg jest liczony w minutach, nie w sekundach.
    #
    # 0 = nigdy nie zwalniaj (zachowanie sprzed tej opcji).
    whisper_idle_unload_s: float = Field(default=300.0, ge=0.0, le=86_400.0)

    # --- Wejście audio (Faza 2) ---
    input_mode: Literal["text", "voice"] = "text"
    mic_enabled: bool = True
    # Fragment nazwy urządzenia, np. "Yeti". Puste = domyślne urządzenie systemowe.
    # Świadomie NIE przyjmujemy indeksu urządzenia — ten sam indeks oznacza inny
    # sprzęt na innym komputerze.
    audio_input_device: str = ""
    audio_sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    audio_frame_ms: Literal[10, 20, 30] = 20
    audio_queue_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    audio_suppress_device_warnings: bool = True

    # --- Słowo aktywujące (Faza 3) ---
    # Sama FRAZA nie jest tutaj: mieszka w config/user_settings.json (pole
    # wake_word), bo należy do warstwy użytkownika — tak jak imię asystenta.
    wake_enabled: bool = True
    # auto = openWakeWord, jeśli jest zainstalowany i ma model dla tej frazy;
    # w przeciwnym razie detektor oparty o Whisper (dowolna fraza, bez modelu).
    wake_engine: Literal["auto", "whisper", "openwakeword", "none"] = "auto"
    # Model Whispera używany WYŁĄCZNIE do wykrywania frazy. Mały z rozmysłem:
    # ma być tani na słabym CPU, a nie dokładny. Puste = ten sam co główny.
    # Model detektora frazy. „tiny" był domyślny do czasu pomiaru na korpusie
    # 480 transkrypcji mowy syntetycznej: wykrywał frazę w 20% przypadków, bo
    # „Miku" zapisywał jako „tymiku", „micu", a przy automatycznym języku nawet
    # japońskimi znakami. „base" z podpowiedzią frazy i dopełnieniem ciszą daje
    # 95% przy zerze fałszywych pobudek, kosztem ~0,3 s na wypowiedź (CPU).
    wake_whisper_model: str = "base"
    wake_similarity: float = Field(default=0.72, ge=0.0, le=1.0)
    # Próg dla SAMEJ nazwy asystenta („miku" bez „hej"). Whisper nagminnie gubi
    # albo przekręca pierwsze słowo („tej miku", „ok, miku", „i miku"), a nazwa
    # jest tym, co naprawdę odróżnia zawołanie od rozmowy w tle. Wartość 0,8
    # wybrana pomiarowo: niżej (0,75) korpus zaczyna reagować na „mikser",
    # „mikrofon" i „Mika" — 7,9% fałszywych pobudek zamiast zera.
    wake_name_similarity: float = Field(default=0.80, ge=0.0, le=1.0)
    # Próg pewności openWakeWord (inna skala niż powyższa).
    wake_openwakeword_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Wypowiedzi dłuższe niż tyle nie są nawet sprawdzane pod kątem frazy —
    # nikt nie mówi „hej miku" przez 10 sekund, a to oszczędza CPU.
    wake_max_utterance_s: float = Field(default=4.0, ge=0.5, le=30.0)
    # Ile sekund po aktywacji można mówić bez powtarzania frazy.
    wake_window_s: float = Field(default=30.0, ge=0.0, le=600.0)

    # --- Synteza mowy (Faza 4) ---
    # Tutaj mieszka wyłącznie MECHANIKA i infrastruktura. Wybór głosu należy do
    # użytkownika i siedzi w config/user_settings.json (piper_model / piper_voices).
    tts_enabled: bool = True
    # Katalog z modelami głosów Pipera (.onnx + .onnx.json). Puste = wykryj
    # automatycznie; patrz piper_voice_directories().
    piper_voices_dir: str = ""
    # Ścieżka do binarki Pipera albo sama nazwa polecenia. Puste = szukaj w PATH
    # i w katalogach z głosami. Używane tylko wtedy, gdy nie ma pakietu `piper`.
    piper_binary: str = ""
    # Mówienie zdaniami: pierwsze zdanie idzie do głośnika, gdy model generuje
    # kolejne. Wyłączenie = synteza dopiero po całej odpowiedzi (prościej, wolniej).
    tts_stream_sentences: bool = True
    # Krótsze fragmenty czekają na dalszy ciąg — synteza dwuwyrazowych urywków
    # brzmi rwanie i kosztuje więcej niż jedno dłuższe zdanie.
    tts_min_sentence_chars: int = Field(default=24, ge=0, le=1_000)
    # Zdanie bez kropki (model potrafi „lecieć" jednym ciągiem) jest dzielone
    # twardo po tylu znakach, żeby odtwarzanie kiedykolwiek ruszyło.
    tts_max_sentence_chars: int = Field(default=320, ge=40, le=5_000)
    # Limit czasu na syntezę jednego fragmentu (proces Pipera potrafi zawisnąć).
    tts_timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    # Fragment NAZWY urządzenia wyjściowego (nie indeks!). Puste = domyślne.
    audio_output_device: str = ""
    # Ile sekund dźwięku wolno trzymać w kolejce odtwarzania.
    audio_output_queue_seconds: float = Field(default=10.0, ge=0.5, le=300.0)

    # --- VAD (Faza 2) ---
    vad_engine: Literal["auto", "webrtc", "energy"] = "auto"
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    # 14 dB wynika z pomiaru na realnym mikrofonie nagłownym: przy 8 dB tło
    # pokoju było klasyfikowane jako mowa w 88% ramek i wypowiedź nigdy się nie
    # kończyła. Kalibrację na własnym sprzęcie robi `python main.py --audio-check`.
    vad_energy_threshold_db: float = Field(default=14.0, ge=1.0, le=40.0)
    vad_min_speech_ms: int = Field(default=250, ge=0, le=5_000)
    vad_min_silence_ms: int = Field(default=700, ge=100, le=10_000)
    vad_preroll_ms: int = Field(default=300, ge=0, le=3_000)
    vad_max_utterance_s: float = Field(default=20.0, ge=1.0, le=300.0)
    vad_listen_timeout_s: float = Field(default=30.0, ge=1.0, le=600.0)

    # --- Aplikacja ---
    # Język odpowiedzi asystenta. Wpisany kod OBOWIĄZUJE: przy LANGUAGE=en pytanie
    # zadane po polsku również dostaje odpowiedź po angielsku. Polski w pełni
    # obsługiwany — LANGUAGE=pl (albo "speech_language": "pl" w user_settings.json).
    # LANGUAGE=auto oddaje decyzję rozpoznawaniu języka przy każdej wypowiedzi.
    language: str = "en"
    # Język INTERFEJSU: napisy w oknie, komunikaty w terminalu, opisy stanu. To
    # coś innego niż LANGUAGE (język odpowiedzi modelu) — ktoś może chcieć
    # angielskiego interfejsu przy polskich odpowiedziach i odwrotnie.
    # „auto" = idź za LANGUAGE. Obsługiwane: en, pl.
    ui_language: str = "en"
    # Czy `python main.py` bez argumentów ma otwierać okno zamiast terminala.
    # Domyślnie TAK: okno jest głównym interfejsem od Fazy 10, a na maszynie bez
    # Tk albo bez sesji graficznej asystent sam schodzi do terminala z jednym
    # komunikatem. Flagi --gui i --no-gui mają pierwszeństwo nad tym ustawieniem.
    gui_enabled: bool = True
    # --- Tryb bezobsługowy (--headless): usługa w tle ---
    # Ile sekund czekać przy starcie na serwer modelu. Usługa użytkownika
    # startuje razem z sesją, często ZANIM Ollama zdąży się podnieść.
    # 0 = nie czekaj wcale.
    headless_ollama_wait_s: float = Field(default=60.0, ge=0.0, le=900.0)
    # Odstęp między próbami odzyskania nasłuchu po awarii mikrofonu.
    headless_retry_s: float = Field(default=15.0, ge=1.0, le=600.0)
    # Długość jednego okna nasłuchu w usłudze. Między oknami pętla sprawdza, czy
    # nie przyszedł SIGTERM — czyli TA wartość, a nie VAD_LISTEN_TIMEOUT_S,
    # decyduje, jak szybko kończy się `systemctl --user stop`. Musi być wyraźnie
    # mniejsza niż TimeoutStopSec w pliku jednostki (domyślnie 30 s), inaczej
    # systemd dobija proces SIGKILL-em w środku sprzątania.
    #
    # Rozpoczętej wypowiedzi ten limit NIE przerywa — dotyczy tylko ciszy.
    headless_listen_slice_s: float = Field(default=5.0, ge=1.0, le=120.0)
    # Czy usługa ma się przywitać na głos przy starcie. Domyślnie NIE: usługa
    # startuje razem z logowaniem i powitanie odzywałoby się przy każdym
    # uruchomieniu komputera.
    headless_greeting: bool = False
    log_level: str = "INFO"
    log_max_bytes: int = Field(default=1_048_576, ge=10_000, le=104_857_600)
    log_backup_count: int = Field(default=3, ge=0, le=50)

    # --- Historia rozmowy ---
    history_max_messages: int = Field(default=40, ge=2, le=1000)
    history_max_chars: int = Field(default=12_000, ge=500, le=1_000_000)

    # Ile z tego okna trafia DO MODELU w jednej turze. To NIE jest to samo co
    # limity okna wyżej: okno jest większe, bo z niego powstają streszczenia
    # (Faza 5) i to ono opisuje rozmowę dla człowieka. Model dostaje ostatni
    # fragment, bo na słabszej maszynie każdy dodatkowy tysiąc tokenów promptu
    # to sekundy czekania — a starsze tury i tak wracają streszczeniem oraz
    # przypomnieniem semantycznym w bloku kontekstu.
    #
    # 0 = bez dodatkowego limitu (do modelu idzie całe okno).
    llm_history_max_messages: int = Field(default=16, ge=0, le=1000)
    llm_history_max_chars: int = Field(default=6_000, ge=0, le=1_000_000)

    # --- Pamięć długoterminowa (Faza 5) ---
    # Wyłączenie zostawia asystenta z samą pamięcią roboczą (okno rozmowy w RAM):
    # nic nie jest zapisywane na dysk i nic nie jest wczytywane przy starcie.
    memory_enabled: bool = True
    # Plik bazy SQLite. Puste = katalog danych właściwy dla TEGO systemu
    # (patrz app_data_directory()). Ścieżka względna liczy się od katalogu
    # projektu. Świadomie nie ma tu wartości domyślnej z konkretną ścieżką —
    # „/home/…", „C:\\…" ani „~/.miku" nie istnieją na każdej maszynie.
    database_path: str = ""
    # Ile sekund czekać, gdy bazę trzyma zajętą inny proces (np. druga instancja).
    database_timeout_s: float = Field(default=5.0, gt=0.0, le=120.0)
    # Kopia pliku bazy przed serią migracji. Kosztuje tyle, co rozmiar bazy.
    database_backup_before_migration: bool = True
    # Zamiast obcinania najstarszych wiadomości (Faza 1) — streszczenie ich
    # przez model. Wyłączenie wraca do zwykłego obcinania.
    memory_summary_enabled: bool = True
    # Po przekroczeniu limitu okno jest przycinane PONIŻEJ limitu, do tego
    # ułamka — inaczej streszczanie odpalałoby się przy każdej kolejnej turze.
    memory_trim_ratio: float = Field(default=0.75, ge=0.25, le=0.95)
    # Górna granica długości streszczenia (model bywa rozwlekły).
    memory_summary_max_chars: int = Field(default=1_500, ge=200, le=20_000)
    # Ile faktów i preferencji doklejać do promptu systemowego.
    memory_context_facts: int = Field(default=20, ge=0, le=200)
    # Automatyczne zapominanie starych rozmów. 0 = trzymaj bezterminowo.
    memory_retention_days: int = Field(default=0, ge=0, le=3650)

    # --- Pamięć semantyczna: embeddingi (Faza 6) ---
    # Wyłączenie zostawia pamięć z Fazy 5 (fakty, notatki, wyszukiwanie po
    # słowach) — znika tylko odnajdywanie wspomnień „po znaczeniu".
    embeddings_enabled: bool = True
    # auto = sentence-transformers, jeśli jest zainstalowany; w przeciwnym razie
    # embeddingi z Ollamy (ta i tak musi działać, więc nie dokłada zależności).
    # ŻADEN z wariantów nie wychodzi poza tę maszynę — nie ma tu i nigdy nie
    # będzie dostawcy chmurowego, bo to są prywatne rozmowy użytkownika.
    embedding_engine: Literal["auto", "sentence-transformers", "ollama", "none"] = "auto"
    # Model sentence-transformers. Domyślny jest wielojęzyczny (obsługuje polski)
    # i mały (~470 MB w pamięci, 384 wymiary) — na CPU liczy zdanie w kilka ms.
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Model embeddingów w Ollamie (wariant zapasowy): ollama pull nomic-embed-text
    embedding_ollama_model: str = "nomic-embed-text"
    # auto rozpoznaje tylko CUDA i CPU — bo tylko te dwie da się wykryć bez
    # importu PyTorcha. „mps" (układy Apple M*) trzeba wskazać wprost: na Macu
    # bez tego wpisu embeddingi policzą się na CPU, czyli poprawnie, tylko wolniej.
    embedding_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    # Czym liczyć podobieństwo wektorów. auto = FAISS, jeśli jest zainstalowany.
    # „numpy" wymuszamy tam, gdzie koło faiss-cpu jest zbudowane pod instrukcje
    # (AVX2), których dany procesor nie ma — taka biblioteka nie zgłasza błędu,
    # tylko wywala proces, więc musi być sposób ominięcia jej bez odinstalowania.
    vector_index: Literal["auto", "faiss", "numpy"] = "auto"
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    # Pobranie modelu przy pierwszym użyciu. W trybie offline i tak nic nie idzie
    # do sieci — patrz apply_offline_environment().
    embedding_allow_download: bool = True
    # Ile wspomnień doklejać do promptu i od jakiego podobieństwa (0..1).
    memory_recall_limit: int = Field(default=5, ge=0, le=50)
    memory_recall_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    # Czy indeksować wypowiedzi użytkownika (wektor i tak powstaje na potrzeby
    # wyszukiwania, więc nie kosztuje to dodatkowego liczenia).
    memory_embed_messages: bool = True
    # Informacja uznana przez model za chwilową dostaje termin ważności zamiast
    # miejsca w pamięci na zawsze. 0 = traktuj chwilowe jak trwałe.
    memory_transient_ttl_hours: int = Field(default=24, ge=0, le=8760)

    # --- Narzędzia i bezpieczeństwo (Faza 7) ---
    # Wyłączenie zostawia asystenta przy samej rozmowie: model nie dostaje listy
    # narzędzi, a router odrzuca każde wywołanie.
    tools_enabled: bool = True
    # Lista dozwolonych narzędzi po przecinku albo „*". Zawężenie działa jak
    # allowlista: co nie jest wymienione, nie zostanie wywołane.
    tools_allowed: str = "*"
    # Lista wyłączonych — ma pierwszeństwo nad TOOLS_ALLOWED.
    tools_disabled: str = ""
    # Ile wywołań narzędzi wolno w JEDNEJ turze. Zabezpieczenie przed pętlą
    # narzędzie → wynik → narzędzie, w którą modele potrafią wpaść.
    tools_max_calls_per_turn: int = Field(default=6, ge=1, le=50)
    # Twardy limit czasu jednego wywołania. Nie ma wariantu „bez limitu" —
    # zawieszone narzędzie zablokowałoby całą rozmowę.
    tool_timeout_s: float = Field(default=15.0, gt=0.0, le=600.0)
    # Ile znaków wyniku narzędzia wpuszczamy do promptu (resztę obcinamy).
    tool_result_max_chars: int = Field(default=4_000, ge=200, le=100_000)
    # Od jakiego poziomu ryzyka pytamy użytkownika o zgodę. Wartość można tylko
    # OBNIŻYĆ poniżej HIGH — HIGH i CRITICAL wymagają zgody zawsze, niezależnie
    # od tego ustawienia (patrz security/policy.py).
    security_require_confirm_from: str = "HIGH"
    # Narzędzia CRITICAL (nieodwracalne, systemowe) są domyślnie wyłączone i nie
    # są nawet pokazywane modelowi.
    security_allow_critical: bool = False
    # Jak długo żądanie potwierdzenia jest ważne. Zgoda po tym czasie nie działa.
    security_confirm_timeout_s: float = Field(default=60.0, ge=5.0, le=3600.0)
    # Log wywołań narzędzi (tabela tool_audit, tylko do dopisywania).
    security_audit_enabled: bool = True
    # Tryb próbny: narzędzia zwracają podgląd zamiast działać.
    security_dry_run: bool = False

    # --- Narzędzia systemowe (Faza 8) ---
    # Katalogi, w których narzędzia plikowe mogą cokolwiek robić. Lista rozdzielona
    # średnikami albo przecinkami; puste = TYLKO własny katalog roboczy asystenta.
    # Nic poza tymi katalogami nie jest widoczne dla żadnego narzędzia plikowego —
    # i nie da się tego obejść ścieżką z ``..`` ani dowiązaniem symbolicznym.
    fs_allowed_roots: str = ""
    # Limity odczytu i zapisu. Chronią okno kontekstu modelu i pamięć maszyny.
    fs_max_read_bytes: int = Field(default=200_000, ge=1_000, le=50_000_000)
    fs_max_write_bytes: int = Field(default=200_000, ge=100, le=50_000_000)
    fs_max_entries: int = Field(default=200, ge=1, le=10_000)
    # Ile plików wolno usunąć jednym wywołaniem (katalog rekurencyjnie).
    fs_max_delete_entries: int = Field(default=50, ge=1, le=10_000)
    # Schematy adresów, które wolno otworzyć. „file" świadomie NIE jest domyślnie:
    # otwarcie pliku to zadanie open.path, które pilnuje dozwolonych katalogów.
    launcher_allowed_schemes: str = "http,https,mailto"
    # Programy, które wolno uruchomić narzędziem shell.run. PUSTA LISTA = narzędzie
    # jest wyłączone. Nazwy, nie ścieżki — rozwiązuje je PATH z weryfikacją katalogu.
    shell_allowed_binaries: str = ""
    shell_timeout_s: float = Field(default=20.0, gt=0.0, le=300.0)
    shell_max_output_chars: int = Field(default=4_000, ge=200, le=100_000)
    # Ile stron PDF-a wolno przeczytać jednym wywołaniem.
    pdf_max_pages: int = Field(default=20, ge=1, le=500)

    # --- Pluginy (Faza 11) ---
    # Rozszerzenia w katalogu ``plugins/``. Każdy plugin rejestruje własne
    # narzędzia i przechodzi przez TEN SAM router i te same bramki co narzędzia
    # wbudowane — bycie pluginem nie daje żadnej taryfy ulgowej.
    plugins_enabled: bool = True
    # Allowlista nazw pluginów po przecinku albo „*" (wszystkie znalezione).
    plugins_allowed: str = "*"
    # Lista wyłączonych — ma pierwszeństwo nad PLUGINS_ALLOWED.
    plugins_disabled: str = ""
    # Dodatkowy katalog z pluginami spoza repozytorium (własne rozszerzenia
    # użytkownika). Puste = tylko katalog ``plugins/`` w projekcie. Ścieżka
    # względna liczy się od katalogu projektu; ``~`` jest rozwijane.
    plugins_extra_dir: str = ""

    # --- Plugin: przypomnienia (Faza 11) ---
    # Co ile sekund asystent sprawdza, czy coś już „dzwoni". Sprawdzenie to
    # jedno zapytanie do SQLite, więc częstsze niż raz na sekundę nie ma sensu.
    reminders_poll_s: float = Field(default=20.0, ge=1.0, le=3600.0)
    # Ile przypomnień wolno zaplanować. Zabezpieczenie przed modelem, który w
    # pętli planuje „za minutę" — nie przed użytkownikiem.
    reminders_max_active: int = Field(default=100, ge=1, le=10_000)
    # Jak długo trzymamy przypomnienia już zrealizowane (do „co mi dzwoniło?").
    reminders_keep_days: int = Field(default=30, ge=0, le=3650)

    # --- Plugin: Home Assistant (Faza 11) ---
    # Adres instancji, np. http://homeassistant.local:8123 albo http://192.168.1.10:8123.
    # PUSTE = plugin jest nieaktywny i model nie widzi jego narzędzi.
    home_assistant_url: str = ""
    # Long-lived access token. SecretStr: nie pokazuje się w logach, w repr
    # obiektu ustawień ani w raporcie zależności — trzeba go jawnie odpakować.
    home_assistant_token: SecretStr = SecretStr("")
    home_assistant_timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)
    # Domeny encji, których przełączenie jest traktowane jak HIGH (zgoda
    # użytkownika obowiązkowa). Zamek w drzwiach i brama garażowa to nie to samo
    # co lampka nocna, choć w API Home Assistanta wygląda tak samo.
    home_assistant_high_risk_domains: str = "lock,cover,gate,alarm_control_panel,valve,water_heater"
    # Ile encji wolno pokazać jednym wywołaniem — okno kontekstu modelu jest
    # skończone, a instalacje domowe miewają setki encji.
    home_assistant_max_entities: int = Field(default=60, ge=1, le=1_000)

    # --- Narzędzia sieciowe (Faza 9) ---
    # Wyłączenie zostawia asystenta z narzędziami lokalnymi. Tryb offline
    # (OFFLINE_MODE) i tak ma pierwszeństwo — wtedy sieci nie ma niezależnie od tego.
    web_enabled: bool = True
    web_timeout_s: float = Field(default=15.0, gt=0.0, le=120.0)
    # Limit pobieranych bajtów i limit tekstu wpuszczanego do promptu — to dwie
    # różne rzeczy: strona bywa duża, a modelowi wystarczy jej treść.
    web_max_bytes: int = Field(default=2_000_000, ge=10_000, le=50_000_000)
    web_max_chars: int = Field(default=6_000, ge=500, le=100_000)
    web_max_redirects: int = Field(default=3, ge=0, le=10)
    web_user_agent: str = ""
    # Dostęp do adresów prywatnych (localhost, 192.168.x.x) jest domyślnie
    # ZABLOKOWANY — inaczej model dałby się namówić na własną Ollamę albo na
    # metadane maszyny w chmurze. Włączać tylko dla własnej instancji SearXNG.
    web_allow_private_hosts: bool = False

    # Wyszukiwanie. „duckduckgo" nie wymaga klucza; „searxng" wymaga adresu
    # własnej instancji w SEARCH_BASE_URL.
    search_provider: Literal["duckduckgo", "searxng", "none"] = "duckduckgo"
    search_base_url: str = ""
    search_max_results: int = Field(default=5, ge=1, le=20)

    # Pogoda: Open-Meteo nie wymaga klucza API (razem z geokodowaniem nazw miast).
    weather_provider: Literal["open-meteo", "none"] = "open-meteo"
    weather_default_location: str = ""
    weather_units: Literal["metric", "imperial"] = "metric"

    # Wiadomości: lista kanałów RSS (rozdzielona przecinkami albo średnikami).
    # Puste = szukanie przez NEWS_SEARCH_URL, w którym {query} i {language} są
    # podstawiane. Domyślny adres to kanał RSS Google News — da się go podmienić
    # na dowolny inny serwis bez zmiany kodu.
    news_feeds: str = ""
    news_search_url: str = "https://news.google.com/rss/search?q={query}&hl={language}"
    news_max_items: int = Field(default=8, ge=1, le=50)

    # YouTube: wyszukiwanie wymaga klucza Data API v3 (nie ma sensownego wariantu
    # bez klucza). Transkrypcje działają bez klucza.
    youtube_max_results: int = Field(default=5, ge=1, le=25)

    # --- Klucze API (Faza 9) ---
    # SecretStr, żeby wartość nie wyciekła do logu, do repr ani do raportu
    # zależności. Wszystkie są OPCJONALNE: brak klucza wyłącza dokładnie te
    # narzędzia, które go wymagają, i nic więcej.
    youtube_api_key: SecretStr = SecretStr("")
    weather_api_key: SecretStr = SecretStr("")
    news_api_key: SecretStr = SecretStr("")
    search_api_key: SecretStr = SecretStr("")

    # --- Interfejs graficzny (Faza 10) ---
    # Uwaga: WYGLĄD okna (kolory) NIE jest tutaj — buduje go GUI z
    # ``ui_accent_color`` w config/user_settings.json, bo to preferencja
    # użytkownika, nie infrastruktura. Tu są wyłącznie rzeczy zależne od maszyny.
    #
    # Tryb jasności: „system" pyta system operacyjny (na części pulpitów Linuksa
    # nie ma jak tego odczytać — wtedy używany jest motyw ciemny).
    gui_theme_mode: Literal["system", "dark", "light"] = "system"
    # Skalowanie interfejsu. 0 = automat (DPI wykryte przez toolkit). Ręczna
    # wartość jest dla przypadków, w których automat nie ma skąd wiedzieć:
    # Wayland ze skalowaniem frakcyjnym, zdalny pulpit, ekran 4K na dokowaniu.
    gui_scaling: float = Field(default=0.0, ge=0.0, le=3.0)
    # Krój pisma okna. Puste = wybór z listy krojów FAKTYCZNIE zainstalowanych na
    # tej maszynie; nazwa nieistniejącego kroju jest ignorowana z ostrzeżeniem.
    gui_font_family: str = ""
    # Rozmiar okna „SZEROKOŚĆxWYSOKOŚĆ", np. 1280x800. Puste = dopasowanie do
    # rozdzielczości ekranu (okno nigdy nie jest większe niż ekran).
    gui_window_size: str = ""

    @field_validator("offline_mode", mode="before")
    @classmethod
    def _normalize_offline_mode(cls, value: Any) -> Any:
        """Przyjmij zarówno ``auto/on/off``, jak i zapis logiczny z ``.env``."""
        if isinstance(value, bool):
            return "on" if value else "off"
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "tak", "on", "offline"):
            return "on"
        if normalized in ("0", "false", "no", "nie", "off", "online"):
            return "off"
        return normalized or "auto"

    @field_validator("audio_frame_ms", mode="before")
    @classmethod
    def _coerce_frame_ms(cls, value: Any) -> Any:
        """Zamień tekst z ``.env`` na liczbę.

        Regresja z prawdziwej instalacji: pydantic nie konwertuje ``"20"`` na
        ``20`` dla ``Literal[10, 20, 30]``, więc plik ``.env`` skopiowany z
        ``.env.example`` wywracał start programu na walidacji. Wartości z
        plików środowiskowych ZAWSZE przychodzą jako tekst.
        """
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return value

    @field_validator("ollama_host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        host = value.strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            raise ValueError(
                t("cfg.bad_ollama_host", value=repr(value))
            )
        return host

    @field_validator("ollama_model", "whisper_model", "embedding_model")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(t("cfg.empty_value"))
        return stripped

    @field_validator("whisper_language")
    @classmethod
    def _validate_whisper_language(cls, value: str) -> str:
        return value.strip().lower()[:5]

    @field_validator("audio_input_device", "audio_output_device")
    @classmethod
    def _validate_audio_device(cls, value: str, info: ValidationInfo) -> str:
        stripped = value.strip()
        if stripped.isdigit():
            raise ValueError(
                t("cfg.device_by_name", field=(info.field_name or "device").upper())
            )
        return stripped

    @field_validator("piper_voices_dir", "piper_binary", "database_path")
    @classmethod
    def _validate_optional_location(cls, value: str) -> str:
        """Ścieżki podane przez użytkownika: tylko przycięcie białych znaków.

        Rozwinięcie ``~`` i zmiennych środowiskowych następuje dopiero w chwili
        użycia — inaczej wartość zapamiętana przy starcie przestawałaby pasować
        po zmianie ``HOME``/``USERPROFILE`` (np. w usłudze systemowej).
        """
        return value.strip()

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if level not in allowed:
            raise ValueError(t("cfg.bad_log_level", allowed=", ".join(sorted(allowed))))
        return level

    @field_validator("security_require_confirm_from")
    @classmethod
    def _validate_confirm_from(cls, value: str) -> str:
        """Przyjmij „high", „HIGH", „High". Literówka schodzi do HIGH, nie niżej.

        Świadomie nie rzucamy błędem: nierozpoznana wartość nie może zatrzymać
        asystenta, ale też nie może obniżyć rygoru — HIGH i CRITICAL i tak zawsze
        wymagają potwierdzenia.
        """
        level = str(value or "").strip().upper()
        if level not in ("SAFE", "MEDIUM", "HIGH", "CRITICAL"):
            logger.warning(
                "SECURITY_REQUIRE_CONFIRM_FROM=%r nie jest poziomem ryzyka — używam HIGH.", value
            )
            return "HIGH"
        return level

    @field_validator("gui_window_size")
    @classmethod
    def _validate_window_size(cls, value: str) -> str:
        """``"1280x800"`` albo puste. Zła wartość nie blokuje startu — okno dobiera samo.

        Rozmiar okna nie jest wart przerwania uruchomienia: literówka w ``.env``
        ma dać okno dopasowane do ekranu, a nie komunikat o błędzie konfiguracji.
        """
        text = value.strip().lower().replace(" ", "")
        if not text:
            return ""
        width, separator, height = text.partition("x")
        if separator and width.isdigit() and height.isdigit():
            return f"{int(width)}x{int(height)}"
        logger.warning(
            "GUI_WINDOW_SIZE=%r nie ma postaci SZEROKOŚĆxWYSOKOŚĆ — dopasuję okno do ekranu.",
            value,
        )
        return ""

    @field_validator("ui_language")
    @classmethod
    def _validate_ui_language(cls, value: str) -> str:
        """Kod języka interfejsu albo ``auto``. Nieznany kod schodzi do angielskiego.

        Świadomie bez błędu walidacji: literówka w ``UI_LANGUAGE`` ma dać
        angielskie napisy, a nie odmowę uruchomienia asystenta.
        """
        code = value.strip().lower()
        if code in ("", "auto"):
            return "auto"
        return code.replace("_", "-").split("-", 1)[0][:2]

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        code = value.strip().lower()
        # „auto" to nie kod języka, a tryb — nie wolno go obciąć do dwóch znaków.
        if code in ("auto", "detect", "wykryj"):
            return "auto"
        return code[:2] or "en"

    def api_url(self, path: str) -> str:
        """Zbuduj adres endpointu Ollamy (``/api/chat`` itd.)."""
        return f"{self.ollama_host}/{path.lstrip('/')}"

    @property
    def frame_samples(self) -> int:
        """Liczba próbek w jednej ramce audio przy docelowej częstotliwości."""
        return int(self.audio_sample_rate * self.audio_frame_ms / 1000)


_settings_lock = threading.RLock()
_settings_cache: Settings | None = None


def _instantiate_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigError(
            t("cfg.bad_env_values", details=_format_validation_error(exc)),
            hint=t("cfg.fix_env", path=ENV_FILE),
        ) from exc
    except Exception as exc:  # np. brak python-dotenv przy odczycie env_file
        logger.warning(
            "Nie udało się wczytać %s (%s) — używam zmiennych środowiskowych.", ENV_FILE, exc
        )
        try:
            return Settings(_env_file=None)  # type: ignore[call-arg]
        except ValidationError as inner:
            raise ConfigError(
                t("cfg.bad_config_values", details=_format_validation_error(inner)),
                hint=t("cfg.check_env_vars"),
            ) from inner


def get_settings(*, force_reload: bool = False) -> Settings:
    """Zwróć ustawienia z ``.env`` (singleton, z możliwością przeładowania)."""
    global _settings_cache
    with _settings_lock:
        if _settings_cache is None or force_reload:
            _settings_cache = _instantiate_settings()
        return _settings_cache


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        lines.append(f"  - {location}: {error.get('msg', 'nieprawidłowa wartość')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tryb offline
#
# Cały asystent liczy lokalnie (Ollama na 127.0.0.1, Whisper z models/whisper),
# więc jedyne miejsca, w których program mógłby sięgnąć do internetu, to:
#   1. pobranie modelu Whisper z HuggingFace przy pierwszym użyciu mikrofonu,
#   2. instalacja pakietów pip,
#   3. `ollama pull` wybranego modelu językowego.
# Wszystkie trzy da się wykonać ZAWCZASU (scripts/prepare_offline.py), a ten
# moduł pilnuje, żeby po przygotowaniu nic już nie próbowało wyjść do sieci —
# także biblioteki zewnętrzne, które robią to za plecami kodu (huggingface_hub
# sprawdza aktualność modelu przy każdym starcie, co bez sieci kończy się
# długim czekaniem na timeout).
# --------------------------------------------------------------------------- #


# Zmienne rozpoznawane przez huggingface_hub / transformers. Ustawiamy je,
# zanim te biblioteki zostaną zaimportowane — czytają je w czasie importu.
_OFFLINE_ENVIRONMENT: Final[dict[str, str]] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}

# Ollama stoi lokalnie, więc nigdy nie wolno jej odpytywać przez proxy —
# ustawione HTTP_PROXY na maszynie bez internetu zerwałoby połączenie z 11434.
_LOCAL_HOSTS: Final[tuple[str, ...]] = ("127.0.0.1", "localhost", "::1")


def is_local_host(url: str) -> bool:
    """Czy adres wskazuje TĘ maszynę?

    Od tego zależy obietnica prywatności: embeddingi i rozmowy wolno liczyć
    tylko lokalnie, a ``OLLAMA_HOST`` da się przestawić na dowolny adres.
    Sprawdzamy sam host z adresu — bez pytania DNS-u (na maszynie offline nie ma
    kogo pytać) i bez zakładania, że „localhost" to jedyna możliwa nazwa.
    """
    raw = url.strip()
    if not raw:
        return False
    host = urllib.parse.urlsplit(raw if "://" in raw else f"http://{raw}").hostname or ""
    lowered = host.lower()
    if lowered in _LOCAL_HOSTS or lowered.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        # Nazwa, której nie umiemy rozstrzygnąć bez DNS-u — traktujemy jako obcą.
        return False


def _short_whisper_name(repo_id: str) -> str:
    """``Systran/faster-whisper-small`` -> ``small`` (obsługa też wariantów distil)."""
    name = repo_id.split("/")[-1].lower()
    if name.startswith("faster-distil-whisper-"):
        return "distil-" + name.removeprefix("faster-distil-whisper-")
    return name.removeprefix("faster-whisper-")


def _hf_dir_to_repo_id(directory_name: str) -> str:
    """``models--Systran--faster-whisper-small`` -> ``Systran/faster-whisper-small``."""
    return directory_name.removeprefix("models--").replace("--", "/")


def _usable_snapshot(model_dir: Path) -> Path | None:
    """Migawka HF zawierająca kompletny model CTranslate2 (albo ``None``).

    Sprawdzamy realne pliki, a nie samą obecność katalogu: przerwane pobieranie
    zostawia strukturę katalogów i dowiązania do nieistniejących blobów.
    """
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None

    candidates: list[Path] = []
    reference = model_dir / "refs" / "main"
    try:
        if reference.is_file():
            revision = reference.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots / revision)
        candidates.extend(sorted(snapshots.iterdir()))
    except OSError:
        return None

    for candidate in candidates:
        try:
            if not candidate.is_dir() or not (candidate / "model.bin").is_file():
                continue
            has_tokenizer = (candidate / "tokenizer.json").is_file() or (
                candidate / "vocabulary.txt"
            ).is_file()
        except OSError:  # pragma: no cover - zależne od uprawnień
            continue
        if has_tokenizer:
            return candidate
    return None


def iter_local_whisper_models(cache_dir: Path | None = None) -> dict[str, Path]:
    """Modele Whisper dostępne bez sieci: ``{nazwa: katalog migawki}``."""
    root = cache_dir or WHISPER_CACHE_DIR
    found: dict[str, Path] = {}
    try:
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:  # pragma: no cover - zależne od uprawnień
        return found

    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        snapshot = _usable_snapshot(entry)
        if snapshot is not None:
            found[_short_whisper_name(_hf_dir_to_repo_id(entry.name))] = snapshot
    return found


def find_local_whisper_model(model: str, cache_dir: Path | None = None) -> Path | None:
    """Ścieżka do modelu Whisper, który da się załadować bez internetu.

    Kolejność: własny katalog wskazany w ``WHISPER_MODEL``, potem cache
    HuggingFace w ``models/whisper``. ``None`` oznacza „trzeba pobrać".
    """
    raw = model.strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    try:
        if candidate.is_dir() and (candidate / "model.bin").is_file():
            return candidate
    except OSError:  # pragma: no cover - zależne od uprawnień
        pass

    wanted = _short_whisper_name(raw) if "/" in raw else raw.lower()
    return iter_local_whisper_models(cache_dir).get(wanted)


# Pliki, po których poznajemy kompletny model sentence-transformers na dysku.
# Wystarczy jeden z nich: układ katalogu różni się między wersjami biblioteki,
# ale każdy działający model ma konfigurację modułów i wagi.
_ST_MARKER_FILES: Final[tuple[str, ...]] = (
    "modules.json",
    "config_sentence_transformers.json",
)
_ST_WEIGHT_FILES: Final[tuple[str, ...]] = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.onnx",
    "tf_model.h5",
)


def _is_sentence_transformer_dir(candidate: Path) -> bool:
    """Czy katalog wygląda na kompletny model sentence-transformers?

    Sprawdzamy realne pliki, nie samą obecność katalogu: przerwane pobieranie
    zostawia strukturę i dowiązania do nieistniejących blobów.
    """
    try:
        if not candidate.is_dir():
            return False
        if not any((candidate / name).is_file() for name in _ST_MARKER_FILES):
            return False
        if any((candidate / name).is_file() for name in _ST_WEIGHT_FILES):
            return True
        # Wagi bywają w podkatalogu modułu (``0_Transformer/``).
        return any(
            (child / name).is_file()
            for child in candidate.iterdir()
            if child.is_dir()
            for name in _ST_WEIGHT_FILES
        )
    except OSError:  # pragma: no cover - zależne od uprawnień
        return False


def _embedding_snapshot(model_dir: Path) -> Path | None:
    """Migawka HF z kompletnym modelem embeddingów (albo ``None``)."""
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return _embedding_dir_or_none(model_dir)

    candidates: list[Path] = []
    reference = model_dir / "refs" / "main"
    try:
        if reference.is_file():
            revision = reference.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(snapshots / revision)
        candidates.extend(sorted(snapshots.iterdir()))
    except OSError:  # pragma: no cover - zależne od uprawnień
        return None

    return next((item for item in candidates if _is_sentence_transformer_dir(item)), None)


def _embedding_dir_or_none(candidate: Path) -> Path | None:
    return candidate if _is_sentence_transformer_dir(candidate) else None


def find_local_embedding_model(model: str, cache_dir: Path | None = None) -> Path | None:
    """Ścieżka do modelu embeddingów, który da się załadować bez internetu.

    Kolejność: własny katalog wpisany w ``EMBEDDING_MODEL``, potem cache w
    ``models/embeddings`` — zarówno w układzie HuggingFace
    (``models--sentence-transformers--nazwa``), jak i w prostszym, starszym
    (``sentence-transformers_nazwa``). ``None`` oznacza „trzeba pobrać".
    """
    raw = model.strip()
    if not raw:
        return None

    explicit = _embedding_dir_or_none(Path(os.path.expandvars(raw)).expanduser())
    if explicit is not None:
        return explicit

    root = cache_dir or EMBEDDINGS_DIR
    # Nazwa bez organizacji: „sentence-transformers/xyz" i „xyz" to ten sam model.
    short = raw.split("/")[-1].lower()
    try:
        entries = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:  # pragma: no cover - zależne od uprawnień
        return None

    for entry in entries:
        if not entry.is_dir():
            continue
        # models--org--nazwa (HF hub) albo org_nazwa / nazwa (starsze wydania ST)
        name = entry.name.removeprefix("models--").replace("--", "/").replace("_", "/")
        if name.split("/")[-1].lower() != short:
            continue
        snapshot = _embedding_snapshot(entry)
        if snapshot is not None:
            return snapshot
    return None


def is_offline(settings: Settings | None = None) -> bool:
    """Czy program ma się zachowywać tak, jakby nie było internetu?"""
    active = settings or get_settings()
    if active.offline_mode == "on":
        return True
    if active.offline_mode == "off":
        return False
    # auto: skoro model mowy jest już na dysku, nie ma po co ruszać sieci.
    return find_local_whisper_model(active.whisper_model) is not None


def describe_offline_mode(settings: Settings | None = None) -> str:
    """Jednolinijkowy opis trybu dla nagłówka, ``/status`` i raportu zależności."""
    active = settings or get_settings()
    offline = is_offline(active)
    if active.offline_mode != "auto":
        return t("mode.forced_offline") if offline else t("mode.forced_online")
    if offline:
        return t("mode.auto_offline")
    return t("mode.auto_online", model=active.whisper_model, path=WHISPER_CACHE_DIR)


def _ensure_local_no_proxy() -> None:
    """Dopisz adresy lokalne do ``no_proxy`` — Ollama nigdy nie idzie przez proxy."""
    for variable in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(variable, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        missing = [host for host in _LOCAL_HOSTS if host not in entries]
        if missing:
            os.environ[variable] = ",".join(entries + missing)


def apply_offline_environment(
    settings: Settings | None = None, *, offline: bool | None = None
) -> bool:
    """Przygotuj zmienne środowiskowe pod pracę bez sieci. Zwraca stan trybu.

    Wołane RAZ, na starcie programu — zanim cokolwiek zaimportuje
    ``huggingface_hub``. Istniejące ustawienia użytkownika mają pierwszeństwo
    (``setdefault``), żeby dało się je świadomie nadpisać z zewnątrz.
    """
    active = settings or get_settings()
    resolved = is_offline(active) if offline is None else offline

    # Cache HF w katalogu projektu — także online, żeby model nie wylądował
    # w ~/.cache i żeby dało się przenieść projekt razem z modelami.
    os.environ.setdefault("HF_HUB_CACHE", str(WHISPER_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(WHISPER_CACHE_DIR))
    _ensure_local_no_proxy()

    if resolved:
        for name, value in _OFFLINE_ENVIRONMENT.items():
            os.environ.setdefault(name, value)
    return resolved


def wheelhouse_packages(directory: Path | None = None) -> int:
    """Ile pakietów leży w lokalnym magazynie kół (``vendor/wheels``)."""
    root = directory or WHEELHOUSE_DIR
    try:
        if not root.is_dir():
            return 0
        return sum(
            1
            for item in root.iterdir()
            if item.is_file() and item.suffix in (".whl", ".gz", ".zip")
        )
    except OSError:  # pragma: no cover - zależne od uprawnień
        return 0


def pip_install_hint(offline: bool | None = None, *, dev: bool = False) -> str:
    """Polecenie instalacji zależności właściwe dla trybu pracy."""
    requirements = "requirements-dev.txt" if dev else "requirements.txt"
    resolved = is_offline() if offline is None else offline
    if not resolved:
        return t("install.pip", requirements=requirements)
    if wheelhouse_packages():
        return t("install.pip_offline", wheelhouse=WHEELHOUSE_DIR, requirements=requirements)
    return t("install.pip_prepare", wheelhouse=WHEELHOUSE_DIR, requirements=requirements)


# --------------------------------------------------------------------------- #
# Detekcja platformy
# --------------------------------------------------------------------------- #


class OSFamily(StrEnum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class PackageManager(StrEnum):
    APT = "apt"
    PACMAN = "pacman"
    DNF = "dnf"
    ZYPPER = "zypper"
    APK = "apk"
    BREW = "brew"
    WINGET = "winget"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    """Wynik detekcji systemu — jedyne źródło prawdy o platformie."""

    os_family: OSFamily
    os_label: str
    os_release: str
    os_version: str
    distro_id: str | None
    distro_like: tuple[str, ...]
    machine: str
    processor: str
    cpu_count: int
    python_version: str
    python_executable: str
    package_manager: PackageManager
    install_script: str
    is_wsl: bool

    @property
    def is_windows(self) -> bool:
        return self.os_family is OSFamily.WINDOWS

    @property
    def is_linux(self) -> bool:
        return self.os_family is OSFamily.LINUX

    @property
    def is_macos(self) -> bool:
        return self.os_family is OSFamily.MACOS

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_family": str(self.os_family),
            "os_label": self.os_label,
            "os_release": self.os_release,
            "os_version": self.os_version,
            "distro_id": self.distro_id,
            "distro_like": list(self.distro_like),
            "machine": self.machine,
            "processor": self.processor,
            "cpu_count": self.cpu_count,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "package_manager": str(self.package_manager),
            "install_script": self.install_script,
            "is_wsl": self.is_wsl,
        }


def _detect_os_family() -> OSFamily:
    if sys.platform.startswith("win"):
        return OSFamily.WINDOWS
    if sys.platform.startswith("linux"):
        return OSFamily.LINUX
    if sys.platform == "darwin":
        return OSFamily.MACOS
    return OSFamily.UNKNOWN


def _read_os_release() -> dict[str, str]:
    """Sparsuj ``/etc/os-release``; pusty słownik, gdy plik nie istnieje."""
    data: dict[str, str] = {}
    for candidate in (Path("/etc/os-release"), Path("/usr/lib/os-release")):
        try:
            content = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip().strip('"').strip("'")
        if data:
            break
    return data


def _detect_package_manager(os_family: OSFamily) -> PackageManager:
    """Wykryj menedżer pakietów po obecności binarki w PATH (nie po nazwie dystrybucji)."""
    if os_family is OSFamily.WINDOWS:
        return PackageManager.WINGET if shutil.which("winget") else PackageManager.NONE
    if os_family is OSFamily.MACOS:
        return PackageManager.BREW if shutil.which("brew") else PackageManager.NONE
    if os_family is OSFamily.LINUX:
        candidates = (
            ("pacman", PackageManager.PACMAN),
            ("apt-get", PackageManager.APT),
            ("apt", PackageManager.APT),
            ("dnf", PackageManager.DNF),
            ("zypper", PackageManager.ZYPPER),
            ("apk", PackageManager.APK),
        )
        for binary, manager in candidates:
            if shutil.which(binary):
                return manager
    return PackageManager.NONE


def _install_script_for(os_family: OSFamily, manager: PackageManager) -> str:
    """Skrypt instalacyjny właściwy dla wykrytej platformy (katalog ``scripts/``)."""
    if os_family is OSFamily.WINDOWS:
        return r"scripts\install-windows.ps1"
    if os_family is OSFamily.MACOS:
        return "scripts/install-macos.sh"
    if os_family is OSFamily.LINUX:
        if manager is PackageManager.APT:
            return "scripts/install-apt.sh"
        if manager is PackageManager.PACMAN:
            return "scripts/install-pacman.sh"
        return "scripts/install-linux-generic.sh"
    return "scripts/install-linux-generic.sh"


def _detect_platform_uncached() -> PlatformInfo:
    os_family = _detect_os_family()
    os_release_data = _read_os_release() if os_family is OSFamily.LINUX else {}

    distro_id = os_release_data.get("ID") or None
    distro_like = tuple(part for part in os_release_data.get("ID_LIKE", "").split() if part)

    if os_family is OSFamily.LINUX:
        os_label = os_release_data.get("PRETTY_NAME") or "Linux"
    elif os_family is OSFamily.WINDOWS:
        os_label = f"Windows {_stdlib_platform.release()}".strip()
    elif os_family is OSFamily.MACOS:
        os_label = f"macOS {_stdlib_platform.mac_ver()[0]}".strip()
    else:
        os_label = _stdlib_platform.system() or "nieznany system"

    release = _stdlib_platform.release()
    is_wsl = os_family is OSFamily.LINUX and "microsoft" in release.lower()
    manager = _detect_package_manager(os_family)

    return PlatformInfo(
        os_family=os_family,
        os_label=os_label,
        os_release=release,
        os_version=_stdlib_platform.version(),
        distro_id=distro_id,
        distro_like=distro_like,
        machine=_stdlib_platform.machine() or "unknown",
        processor=_stdlib_platform.processor() or "unknown",
        cpu_count=os.cpu_count() or 1,
        python_version=_stdlib_platform.python_version(),
        python_executable=sys.executable or "python",
        package_manager=manager,
        install_script=_install_script_for(os_family, manager),
        is_wsl=is_wsl,
    )


_platform_lock = threading.RLock()
_platform_cache: PlatformInfo | None = None


def detect_platform(*, force_reload: bool = False) -> PlatformInfo:
    """Wykryj system operacyjny, architekturę i menedżer pakietów (wynik cache'owany)."""
    global _platform_cache
    with _platform_lock:
        if _platform_cache is None or force_reload:
            _platform_cache = _detect_platform_uncached()
        return _platform_cache


def install_instruction(platform_info: PlatformInfo | None = None) -> str:
    """Komunikat kierujący użytkownika do właściwego skryptu instalacyjnego."""
    info = platform_info or detect_platform()
    return t("install.run_script", script=info.install_script)


def _subprocess_kwargs() -> dict[str, Any]:
    """Argumenty ``subprocess`` ukrywające okno konsoli na Windowsie."""
    kwargs: dict[str, Any] = {}
    creation_flag = getattr(subprocess, "CREATE_NO_WINDOW", None)
    if creation_flag is not None:
        kwargs["creationflags"] = creation_flag
    return kwargs


def subprocess_no_window_kwargs() -> dict[str, Any]:
    """Publiczna wersja :func:`_subprocess_kwargs` dla pozostałych modułów.

    Każdy proces potomny (np. binarka Pipera z Fazy 4) ma być uruchamiany z tymi
    argumentami — inaczej na Windowsie mignęłoby czarne okno konsoli przy każdej
    wypowiedzi. Na pozostałych systemach słownik jest pusty i nic nie zmienia.
    """
    return _subprocess_kwargs()


# --------------------------------------------------------------------------- #
# Katalogi danych systemowych i lokalizacja Pipera (Faza 4)
#
# To jedyne miejsce, które wie, gdzie DANY system trzyma dane aplikacji.
# ``audio/tts.py`` dostaje gotową listę katalogów i nigdy nie skleja ścieżek
# sam ani nie sprawdza ``sys.platform``.
# --------------------------------------------------------------------------- #


def _existing_unique(paths: Sequence[Path]) -> list[Path]:
    """Usuń duplikaty zachowując kolejność (ścieżki nie muszą istnieć)."""
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve(strict=False))
        except OSError:  # pragma: no cover - egzotyczne systemy plików
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _home_directory() -> Path | None:
    """Katalog domowy albo ``None``, gdy system go nie zna.

    ``Path.home()`` rzuca ``RuntimeError``, gdy nie da się ustalić katalogu
    domowego (usługa systemowa bez ``HOME``, konto bez profilu na Windowsie).
    Brak katalogu domowego ma zawęzić listę kandydatów, a nie wywrócić program.
    """
    try:
        return Path.home()
    except (RuntimeError, OSError):  # pragma: no cover - zależne od środowiska
        return None


def home_directory() -> Path | None:
    """Katalog domowy użytkownika albo ``None``, gdy system go nie zna.

    Publiczna wersja :func:`_home_directory` dla warstwy ``host/`` (Faza 8): to
    ona buduje ścieżki systemowe, ale katalogu domowego nie zgaduje sama.
    """
    return _home_directory()


def path_from_env(variable: str) -> Path | None:
    """Ścieżka ze zmiennej środowiskowej (z rozwinięciem ``~`` i ``$VAR``).

    Publiczna wersja :func:`_path_from_env`. Warstwa ``host/`` czyta przez nią
    zmienne systemowe (``APPDATA``, ``PROGRAMDATA``, ``XDG_*``) — dzięki temu
    rozwijanie ścieżek jest w projekcie zrobione w jeden sposób.
    """
    return _path_from_env(variable)


def _platform_data_home(platform_info: PlatformInfo | None = None) -> Path | None:
    """Bazowy katalog danych użytkownika DLA TEGO systemu (bez nazwy aplikacji).

    Windows: ``%LOCALAPPDATA%`` (albo ``%APPDATA%``), macOS:
    ``~/Library/Application Support``, reszta: ``$XDG_DATA_HOME`` albo
    ``~/.local/share``. ``None`` = system nie podał żadnego z tych miejsc.
    """
    info = platform_info or detect_platform()

    if info.is_windows:
        for variable in ("LOCALAPPDATA", "APPDATA"):
            base = _path_from_env(variable)
            if base is not None:
                return base
        return None

    home = _home_directory()
    if info.is_macos:
        return home / "Library" / "Application Support" if home is not None else None

    xdg_home = _path_from_env("XDG_DATA_HOME")
    if xdg_home is not None:
        return xdg_home
    return home / ".local" / "share" if home is not None else None


def app_data_directory(platform_info: PlatformInfo | None = None) -> Path:
    """Katalog na dane ZAPISYWANE przez program (baza SQLite, indeksy).

    Kolejność:

    1. ``MIKU_DATA_DIR`` — świadomy wybór użytkownika albo instalatora,
    2. katalog danych właściwy dla systemu + :data:`APP_NAME`,
    3. ``data/`` w katalogu projektu — ostatnia deska ratunku dla maszyny, na
       której nie da się ustalić katalogu domowego (kontener, usługa, live USB).

    Funkcja niczego nie tworzy: katalog zakłada dopiero ten, kto naprawdę
    chce coś zapisać (``database/database.py``).
    """
    override = _path_from_env("MIKU_DATA_DIR")
    if override is not None:
        return override
    base = _platform_data_home(platform_info)
    if base is None:
        return PROJECT_ROOT / "data"
    return base / APP_NAME


def database_file(settings: Settings | None = None) -> Path:
    """Pełna ścieżka pliku bazy SQLite.

    ``DATABASE_PATH`` z ``.env`` ma pierwszeństwo (ścieżka względna liczy się od
    katalogu projektu, ``~`` i zmienne środowiskowe są rozwijane). Puste pole =
    plik :data:`DEFAULT_DATABASE_NAME` w katalogu z :func:`app_data_directory`.
    Wartość ``:memory:`` daje bazę wyłącznie w pamięci (przydatne w testach).
    """
    active = settings or get_settings()
    configured = active.database_path.strip()
    if not configured:
        return app_data_directory() / DEFAULT_DATABASE_NAME
    if configured == MEMORY_DATABASE:
        return Path(MEMORY_DATABASE)
    path = Path(os.path.expandvars(configured)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def user_data_directories(platform_info: PlatformInfo | None = None) -> list[Path]:
    """Katalogi, w których DANY system trzyma dane aplikacji użytkownika.

    Zwracane ścieżki są tylko kandydatami — żadna nie musi istnieć. Wszystkie
    pochodzą ze zmiennych środowiskowych systemu (``XDG_DATA_HOME``,
    ``LOCALAPPDATA``, ``HOME``), więc nie zakładamy ani nazwy użytkownika, ani
    układu katalogów na konkretnej maszynie.
    """
    info = platform_info or detect_platform()
    candidates: list[Path] = []
    home = _home_directory()

    if info.is_windows:
        for variable in ("LOCALAPPDATA", "APPDATA"):
            base = _path_from_env(variable)
            if base is not None:
                candidates.append(base)
    elif info.is_macos:
        if home is not None:
            candidates.append(home / "Library" / "Application Support")
    else:
        xdg_home = _path_from_env("XDG_DATA_HOME")
        if xdg_home is not None:
            candidates.append(xdg_home)
        elif home is not None:
            candidates.append(home / ".local" / "share")
        raw_dirs = os.environ.get("XDG_DATA_DIRS", "").strip()
        # Rozdzielnik listy katalogów różni się między systemami — bierzemy ten
        # właściwy dla platformy (``os.pathsep``), a nie zaszyty dwukropek.
        system_dirs = [item for item in raw_dirs.split(os.pathsep) if item.strip()]
        if not system_dirs:
            system_dirs = ["/usr/local/share", "/usr/share"]
        candidates.extend(Path(item.strip()).expanduser() for item in system_dirs)

    return _existing_unique(candidates)


def piper_voice_directories(settings: Settings | None = None) -> list[Path]:
    """Katalogi przeszukiwane w poszukiwaniu głosów Pipera (``*.onnx``).

    Kolejność (pierwsze trafienie wygrywa):

    1. ``PIPER_VOICES_DIR`` z ``.env`` — świadomy wybór użytkownika,
    2. ``models/piper`` w katalogu projektu — przenosi się razem z projektem,
    3. katalogi danych wykryte dla tego systemu (``piper``/``piper-voices``).

    Żaden z katalogów nie musi istnieć; funkcja niczego nie tworzy i niczego
    nie wymaga. Pusta lista wyników wyszukiwania oznacza po prostu „brak głosu",
    co wyłącza mowę, ale nie zatrzymuje asystenta.
    """
    active = settings or get_settings()
    candidates: list[Path] = []

    configured = active.piper_voices_dir.strip()
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)

    candidates.append(PIPER_DIR)

    for base in user_data_directories():
        candidates.append(base / "piper" / "voices")
        candidates.append(base / "piper-voices")
        candidates.append(base / "piper")

    return _existing_unique(candidates)


def find_piper_binary(settings: Settings | None = None) -> Path | None:
    """Znajdź program ``piper``. ``None`` = nie ma go na tej maszynie.

    Kolejność: ``PIPER_BINARY`` z ``.env`` (ścieżka albo nazwa polecenia),
    ``PATH``, katalog obok bieżącego interpretera, na końcu katalogi z głosami.

    Katalog interpretera jest tu istotny w praktyce: ``pip install piper-tts``
    w środowisku wirtualnym kładzie program w ``.venv/bin`` (``.venv\\Scripts``
    na Windowsie), a to **nie** jest w ``PATH``, gdy asystenta uruchamia się
    jako ``.venv/bin/python main.py`` — czyli dokładnie tak, jak radzą skrypty
    instalacyjne. Rozszerzenie ``.exe`` na Windowsie dokłada ``shutil.which``
    na podstawie ``PATHEXT``, więc nie ma tu rozgałęzienia na system.
    """
    active = settings or get_settings()

    configured = active.piper_binary.strip()
    if configured:
        expanded = Path(os.path.expandvars(configured)).expanduser()
        candidate = expanded if expanded.is_absolute() else PROJECT_ROOT / expanded
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # pragma: no cover - zależne od uprawnień
            pass
        found = shutil.which(configured)
        if found:
            return Path(found)
        logger.warning(
            "PIPER_BINARY=%r nie wskazuje na istniejący program — szukam dalej.", configured
        )

    found = shutil.which("piper")
    if found:
        return Path(found)

    directories: list[Path] = []
    interpreter = Path(sys.executable).parent if sys.executable else None
    if interpreter is not None:
        directories.append(interpreter)
    directories.extend(piper_voice_directories(active))

    for directory in directories:
        for name in ("piper", "piper.exe"):
            candidate = directory / name
            try:
                if candidate.is_file():
                    return candidate
            except OSError:  # pragma: no cover - zależne od uprawnień
                continue
    return None


# --------------------------------------------------------------------------- #
# Detekcja GPU / CUDA — nigdy nie zakłada, że NVIDIA istnieje i nigdy nie rzuca
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GPUInfo:
    cuda_available: bool
    source: Literal["nvidia-smi", "torch", "none"]
    device_name: str | None
    driver_version: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuda_available": self.cuda_available,
            "source": self.source,
            "device_name": self.device_name,
            "driver_version": self.driver_version,
            "detail": self.detail,
        }


_GPU_UNAVAILABLE: Final[GPUInfo] = GPUInfo(
    cuda_available=False,
    source="none",
    device_name=None,
    driver_version=None,
    detail=t("deps.gpu.none"),
)


def _detect_cuda_via_nvidia_smi() -> GPUInfo | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        completed = subprocess.run(  # nosec B603 - stała lista argumentów, bez powłoki
            [binary, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
            **_subprocess_kwargs(),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("nvidia-smi nie odpowiedziało: %s", exc)
        return None

    if completed.returncode != 0:
        logger.debug("nvidia-smi zakończone kodem %s", completed.returncode)
        return None

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None

    parts = [part.strip() for part in first_line.split(",")]
    name = parts[0] if parts else None
    driver = parts[1] if len(parts) > 1 else None
    return GPUInfo(
        cuda_available=True,
        source="nvidia-smi",
        device_name=name,
        driver_version=driver,
        detail=t(
            "deps.gpu.nvidia_smi",
            name=name or "GPU",
            driver=driver or t("deps.gpu.unknown_driver"),
        ),
    )


def _detect_cuda_via_torch() -> GPUInfo | None:
    try:
        if importlib.util.find_spec("torch") is None:
            return None
    except (ImportError, ValueError):
        return None

    try:
        import torch  # noqa: PLC0415 - import celowo leniwy, biblioteka jest ciężka

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else None
        driver = getattr(torch.version, "cuda", None)
    except Exception as exc:  # torch potrafi rzucić czym tylko chce
        logger.debug("torch nie zgłosił dostępnego CUDA: %s", exc)
        return None

    return GPUInfo(
        cuda_available=True,
        source="torch",
        device_name=name,
        driver_version=driver,
        detail=t(
            "deps.gpu.torch",
            name=name or "GPU",
            driver=driver or t("deps.gpu.unknown_version"),
        ),
    )


def detect_cuda() -> GPUInfo:
    """Sprawdź dostępność CUDA. Nigdy nie rzuca wyjątku — brak GPU to poprawny wynik."""
    try:
        info = _detect_cuda_via_nvidia_smi() or _detect_cuda_via_torch()
    except Exception as exc:  # pragma: no cover - ostatnia linia obrony
        logger.debug("Detekcja CUDA nieudana: %s", exc)
        return _GPU_UNAVAILABLE
    return info or _GPU_UNAVAILABLE


def resolve_compute_device(preferred: str, gpu: GPUInfo | None = None) -> str:
    """Zamień ustawienie ``auto`` na realnie dostępne urządzenie obliczeniowe."""
    if preferred != "auto":
        return preferred
    info = gpu if gpu is not None else detect_cuda()
    return "cuda" if info.cuda_available else "cpu"


# --------------------------------------------------------------------------- #
# Detekcja Ollamy (stdlib urllib — działa nawet gdy httpx nie jest zainstalowany)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    host: str
    reachable: bool
    version: str | None
    models: tuple[str, ...]
    requested_model: str
    model_present: bool
    binary_path: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "reachable": self.reachable,
            "version": self.version,
            "models": list(self.models),
            "requested_model": self.requested_model,
            "model_present": self.model_present,
            "binary_path": self.binary_path,
            "error": self.error,
        }


def _http_get_json(url: str, timeout: float) -> tuple[Any | None, str | None]:
    request = urllib.request.Request(  # nosec B310 - schemat wymuszony walidatorem OLLAMA_HOST
        url,
        headers={"Accept": "application/json", "User-Agent": f"{APP_NAME}/{APP_VERSION}"},
    )
    # Pusty ProxyHandler = ignoruj HTTP_PROXY/HTTPS_PROXY. Ollama jest usługą
    # lokalną, a proxy ustawione w systemie (częste w firmowych sieciach, także
    # gdy sieci akurat nie ma) przekierowałoby zapytanie w próżnię.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:  # nosec B310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        return None, t("net.http_status", code=exc.code)
    except urllib.error.URLError as exc:
        return None, t("net.no_connection", reason=exc.reason)
    except (TimeoutError, socket.timeout):
        return None, t("net.timeout", seconds=f"{timeout:.0f}")
    except (OSError, ValueError) as exc:
        return None, t("net.error", error=exc)

    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"nieprawidłowa odpowiedź JSON: {exc}"


def _model_matches(requested: str, available: Sequence[str]) -> bool:
    """Porównaj nazwę modelu tolerując brak/obecność sufiksu ``:latest``."""
    wanted = requested.strip().lower()
    if not wanted:
        return False
    catalogue = {name.strip().lower() for name in available}
    if wanted in catalogue:
        return True
    if ":" not in wanted and f"{wanted}:latest" in catalogue:
        return True
    if wanted.endswith(":latest") and wanted.removesuffix(":latest") in catalogue:
        return True
    return False


def detect_ollama(settings: Settings | None = None) -> OllamaStatus:
    """Sprawdź, czy Ollama odpowiada i czy wybrany model jest pobrany. Nie rzuca."""
    active = settings or get_settings()
    host = active.ollama_host
    binary = shutil.which("ollama")

    version_data, version_error = _http_get_json(
        f"{host}/api/version", active.ollama_connect_timeout
    )
    if version_data is None:
        return OllamaStatus(
            host=host,
            reachable=False,
            version=None,
            models=(),
            requested_model=active.ollama_model,
            model_present=False,
            binary_path=binary,
            error=version_error,
        )

    version = None
    if isinstance(version_data, dict):
        raw_version = version_data.get("version")
        version = str(raw_version) if raw_version is not None else None

    tags_data, tags_error = _http_get_json(f"{host}/api/tags", active.ollama_connect_timeout)
    models: tuple[str, ...] = ()
    if isinstance(tags_data, dict):
        entries = tags_data.get("models")
        if isinstance(entries, list):
            models = tuple(
                str(entry["name"])
                for entry in entries
                if isinstance(entry, dict) and entry.get("name")
            )

    return OllamaStatus(
        host=host,
        reachable=True,
        version=version,
        models=models,
        requested_model=active.ollama_model,
        model_present=_model_matches(active.ollama_model, models),
        binary_path=binary,
        error=tags_error,
    )


# --------------------------------------------------------------------------- #
# Warstwa 2: config/user_settings.json -> UserSettings
# --------------------------------------------------------------------------- #

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")
_NON_TAG_CHARS = re.compile(r"[^0-9A-Za-z\u00c0-\u024f]+")


def _strip_control_characters(value: str, *, keep_newlines: bool = False) -> str:
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    if not keep_newlines:
        cleaned = cleaned.replace("\n", " ")
    cleaned = _CONTROL_CHARS.sub("", cleaned)
    return _WHITESPACE_RUN.sub(" ", cleaned).strip()


class RVCSettings(BaseModel):
    """Konfiguracja konwersji głosu RVC (wykorzystywana od Fazy 15).

    Ścieżki podaje użytkownik w ``config/user_settings.json`` — nigdy nie są
    zaszyte w kodzie. Ścieżka względna jest liczona od katalogu projektu.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    enabled: bool = False
    model_path: str = ""
    index_path: str = ""
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator("model_path", "index_path")
    @classmethod
    def _clean_path(cls, value: str) -> str:
        return _strip_control_characters(value)

    @staticmethod
    def _resolve(raw: str) -> Path | None:
        if not raw.strip():
            return None
        candidate = Path(os.path.expandvars(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate

    @property
    def resolved_model_path(self) -> Path | None:
        return self._resolve(self.model_path)

    @property
    def resolved_index_path(self) -> Path | None:
        return self._resolve(self.index_path)

    def missing_files(self) -> list[Path]:
        """Zadeklarowane pliki RVC, których nie ma na dysku."""
        missing: list[Path] = []
        for path in (self.resolved_model_path, self.resolved_index_path):
            if path is not None and not path.is_file():
                missing.append(path)
        return missing

    def is_usable(self) -> bool:
        return self.enabled and self.resolved_model_path is not None and not self.missing_files()


class UserSettings(BaseModel):
    """Ustawienia edytowane przez użytkownika (i przez GUI z Fazy 10)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    assistant_name: str = Field(default="Miku", min_length=1, max_length=32)
    # Fraza wybudzająca. Puste = „hej <assistant_name>", więc zmiana imienia
    # sama zmienia słowo aktywujące. Wpisana wartość ma pierwszeństwo i może
    # być dowolna — kod nigdzie nie zakłada konkretnego brzmienia.
    wake_word: str = Field(default="", max_length=64)
    # Opcjonalny model openWakeWord dla tej frazy (.onnx/.tflite). Ścieżka
    # względna liczy się od models/wakeword. Puste = detektor whisperowy.
    wake_word_model: str = ""
    # Preferowany język mowy: kod ISO („pl", „en", „de"...), „auto" albo „detect".
    #
    # „auto" (domyślnie) znaczy „ten sam język, w którym asystent odpowiada"
    # (LANGUAGE) — a NIE „niech Whisper zgaduje przy każdej wypowiedzi. Zmiana
    # z pomiaru: przy zgadywaniu 3 na 10 polskich zdań zostały rozpoznane jako
    # urdu, niemiecki i rosyjski i przepisane w tych językach (WER 49,5% wobec
    # 27,0% przy wymuszonym „pl"), a transkrypcja trwała dwa razy dłużej.
    # „detect" przywraca zgadywanie — dla kogoś, kto naprawdę mówi w kilku
    # językach na zmianę.
    speech_language: str = "auto"
    ui_accent_color: str = Field(default="#39C5BB", pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
    personality_traits: str = Field(default="", max_length=2000)
    # Silnik mowy: nazwa zarejestrowana w audio/tts.py („piper", „none", a w
    # kolejnych fazach „xtts", „rvc"...). Świadomie NIE jest to Literal — nowy
    # dostawca ma się dodawać rejestracją w rejestrze, bez zmiany tego modelu.
    # Nieznana nazwa nie wywraca konfiguracji: kończy się ostrzeżeniem i mową
    # wyłączoną (albo dostawcą domyślnym).
    voice_engine: str = Field(default="piper", max_length=32)
    # Model głosu Pipera: nazwa („pl_PL-darkman-medium"), ścieżka do pliku
    # .onnx albo puste = wybierz automatycznie głos pasujący do języka.
    # Podmiana głosu to edycja TEGO pola — nic w kodzie nie jest zaszyte.
    piper_model: str = ""
    # Osobny głos per język, np. {"pl": "pl_PL-darkman-medium", "en": "en_US-amy-medium"}.
    # Puste = wszystkie języki obsługuje piper_model (albo automat).
    piper_voices: dict[str, str] = Field(default_factory=dict)
    # Numer mówcy w modelach wielogłosowych (jednogłosowe ignorują tę wartość).
    piper_speaker: int = Field(default=0, ge=0, le=10_000)
    # Tempo mowy: 1.0 = naturalne dla modelu, 1.3 = szybciej, 0.8 = wolniej.
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # Głośność odtwarzania (mnożnik amplitudy, nie miksera systemowego).
    voice_volume: float = Field(default=0.9, ge=0.0, le=1.0)
    rvc: RVCSettings = Field(default_factory=RVCSettings)

    @field_validator("assistant_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = _strip_control_characters(value)
        if not cleaned:
            raise ValueError(t("cfg.empty_assistant_name"))
        return cleaned

    @field_validator("personality_traits")
    @classmethod
    def _clean_traits(cls, value: str) -> str:
        return _strip_control_characters(value, keep_newlines=True)

    @field_validator("piper_model")
    @classmethod
    def _clean_piper_model(cls, value: str) -> str:
        return _strip_control_characters(value)

    @field_validator("voice_engine")
    @classmethod
    def _clean_voice_engine(cls, value: str) -> str:
        """Nazwa silnika bez wielkich liter i spacji; puste = mowa wyłączona."""
        engine = _strip_control_characters(value).lower().replace(" ", "")
        return engine or "none"

    @field_validator("piper_voices")
    @classmethod
    def _clean_piper_voices(cls, value: dict[str, str]) -> dict[str, str]:
        """Klucze to kody języków (``pl``, ``en``), wartości — nazwy modeli."""
        cleaned: dict[str, str] = {}
        for raw_language, raw_model in value.items():
            language = _strip_control_characters(str(raw_language)).lower()
            language = language.replace("_", "-").split("-", 1)[0][:2]
            model = _strip_control_characters(str(raw_model))
            if language and model:
                cleaned[language] = model
        return cleaned

    @property
    def speaks(self) -> bool:
        """Czy użytkownik w ogóle chce słyszeć odpowiedzi?"""
        return self.voice_engine not in ("none", "off", "brak")

    def voice_for_language(self, language: str | None) -> str:
        """Model głosu dla danego języka — z mapy ``piper_voices``, inaczej domyślny."""
        if language:
            code = language.strip().lower().replace("_", "-").split("-", 1)[0][:2]
            override = self.piper_voices.get(code, "").strip()
            if override:
                return override
        return self.piper_model.strip()

    @field_validator("wake_word", "wake_word_model")
    @classmethod
    def _clean_wake_word(cls, value: str) -> str:
        return _strip_control_characters(value)

    @field_validator("speech_language")
    @classmethod
    def _clean_speech_language(cls, value: str) -> str:
        """Kod języka, LISTA kodów, ``auto`` albo ``detect``.

        Lista („pl,en") jest odpowiedzią na sytuację zmierzoną na tym projekcie:
        użytkownik mówi w dwóch językach, a wymuszenie jednego niszczy drugi
        (WER 114% dla polskiej mowy przy wymuszonym angielskim). Przy liście
        język jest ROZPOZNAWANY, ale wybierany tylko spośród podanych — więc
        polskie zdanie nie zostanie nigdy przepisane po urdu.
        """
        code = _strip_control_characters(value).lower()
        if not code or code in ("auto", "automatyczny", "any"):
            return "auto"
        if code in ("detect", "wykryj", "rozpoznaj"):
            return "detect"
        codes: list[str] = []
        for part in code.replace(";", ",").replace(" ", ",").split(","):
            trimmed = part.strip()
            if not trimmed:
                continue
            # Whisper przyjmuje dwuliterowe kody ISO 639-1; „pl-PL" skracamy do „pl".
            short = trimmed.replace("_", "-").split("-", 1)[0][:2]
            if short and short not in codes:
                codes.append(short)
        return ",".join(codes) if codes else "auto"

    @property
    def is_speech_language_forced(self) -> bool:
        """Czy użytkownik podał kod (albo listę kodów) zamiast „auto"/„detect"."""
        return self.speech_language not in ("auto", "detect")

    @property
    def speech_languages(self) -> tuple[str, ...]:
        """Języki, którymi mówi użytkownik. Pusta krotka = „rozpoznawaj"."""
        if not self.is_speech_language_forced:
            return ()
        return tuple(code for code in self.speech_language.split(",") if code)

    @property
    def effective_wake_word(self) -> str:
        """Fraza wybudzająca użyta faktycznie — z pliku albo z imienia asystenta.

        Nigdzie w kodzie nie ma zaszytego „hej miku": domyślna fraza powstaje
        z ``assistant_name``, więc po zmianie imienia na ``Aiko`` asystent
        nasłuchuje „hej aiko" bez żadnej dodatkowej konfiguracji.
        """
        explicit = self.wake_word.strip()
        if explicit:
            return explicit
        return f"hej {self.assistant_name.strip()}"

    @property
    def display_tag(self) -> str:
        """Tag terminala budowany z imienia asystenta, np. ``AIKO`` dla ``Aiko``."""
        cleaned = _NON_TAG_CHARS.sub("", self.assistant_name)
        return (cleaned or "ASSISTANT").upper()

    @property
    def log_tag(self) -> str:
        return f"[{self.display_tag}]"


DEFAULT_USER_SETTINGS: Final[UserSettings] = UserSettings()

_user_settings_lock = threading.RLock()
_user_settings_cache: UserSettings | None = None
_user_settings_mtime: int | None = None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Zapisz JSON atomowo (tmp + ``os.replace``) — działa tak samo na Linuksie i Windowsie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:  # pragma: no cover
                pass


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Scal słowniki rekurencyjnie — nieznane klucze użytkownika zostają zachowane."""
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_raw_user_settings(path: Path) -> dict[str, Any]:
    """Odczytaj surowy JSON; pusty słownik przy braku/uszkodzeniu pliku."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def write_default_user_settings(path: Path | None = None) -> Path:
    """Utwórz ``config/user_settings.json`` z wartościami domyślnymi."""
    target = path or USER_SETTINGS_FILE
    _write_json_atomic(target, DEFAULT_USER_SETTINGS.model_dump())
    logger.info("Utworzono plik ustawień użytkownika: %s", target)
    return target


def load_user_settings(path: Path | None = None, *, force_reload: bool = False) -> UserSettings:
    """Wczytaj ustawienia użytkownika.

    * brak pliku  -> tworzy go z wartościami domyślnymi,
    * zły JSON / zła struktura -> ostrzeżenie w logach i wartości domyślne
      (plik użytkownika NIE jest nadpisywany, żeby dało się go poprawić ręcznie),
    * plik zmieniony na dysku -> automatyczne przeładowanie bez restartu programu.
    """
    global _user_settings_cache, _user_settings_mtime

    target = path or USER_SETTINGS_FILE
    with _user_settings_lock:
        mtime = _file_mtime_ns(target)
        if (
            not force_reload
            and _user_settings_cache is not None
            and mtime is not None
            and mtime == _user_settings_mtime
        ):
            return _user_settings_cache

        if mtime is None:
            try:
                write_default_user_settings(target)
            except OSError as exc:
                logger.warning(
                    "Nie udało się utworzyć %s (%s) — używam wartości domyślnych.",
                    target,
                    exc,
                )
                _user_settings_cache = DEFAULT_USER_SETTINGS
                _user_settings_mtime = None
                return _user_settings_cache
            mtime = _file_mtime_ns(target)

        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Nie można odczytać %s (%s) — używam wartości domyślnych.", target, exc)
            _user_settings_cache = DEFAULT_USER_SETTINGS
            _user_settings_mtime = mtime
            return _user_settings_cache

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Plik %s ma nieprawidłowy format JSON (linia %s, kolumna %s: %s). "
                "Używam wartości domyślnych — popraw plik albo usuń go, "
                "a zostanie odtworzony przy następnym starcie.",
                target,
                exc.lineno,
                exc.colno,
                exc.msg,
            )
            _user_settings_cache = DEFAULT_USER_SETTINGS
            _user_settings_mtime = mtime
            return _user_settings_cache

        if not isinstance(raw, dict):
            logger.warning(
                "Plik %s powinien zawierać obiekt JSON, a zawiera %s. Używam wartości domyślnych.",
                target,
                type(raw).__name__,
            )
            _user_settings_cache = DEFAULT_USER_SETTINGS
            _user_settings_mtime = mtime
            return _user_settings_cache

        try:
            settings = UserSettings.model_validate(raw)
        except ValidationError as exc:
            logger.warning(
                "Nieprawidłowe wartości w %s:\n%s\nUżywam wartości domyślnych.",
                target,
                _format_validation_error(exc),
            )
            _user_settings_cache = DEFAULT_USER_SETTINGS
            _user_settings_mtime = mtime
            return _user_settings_cache

        if settings.rvc.enabled:
            missing = settings.rvc.missing_files()
            if settings.rvc.resolved_model_path is None:
                logger.warning(
                    "rvc.enabled=true, ale rvc.model_path jest puste — "
                    "konwersja głosu pozostanie wyłączona."
                )
            elif missing:
                logger.warning(
                    "rvc.enabled=true, ale nie znaleziono plików: %s",
                    ", ".join(str(item) for item in missing),
                )

        _user_settings_cache = settings
        _user_settings_mtime = mtime
        return settings


def resolve_speech_languages(
    settings: Settings | None = None, user_settings: UserSettings | None = None
) -> tuple[str, ...]:
    """Języki, którymi mówi użytkownik. Pusta krotka = „rozpoznawaj dowolny".

    Jeden kod = wymuszenie. Kilka kodów = rozpoznawanie ograniczone do nich —
    to odpowiedź na przypadek zmierzony na tym projekcie: przy wymuszonym
    angielskim polska wypowiedź miała WER 114% (tekst nie nadawał się do niczego),
    a przy wymuszonym polskim angielska 55,8%. Rozpoznawanie ograniczone do
    „pl,en" daje 41,7% i 5,0% — czyli jedyny układ, w którym obie strony działają.
    """
    user = user_settings if user_settings is not None else get_user_settings()
    if user.is_speech_language_forced:
        return user.speech_languages
    if user.speech_language == "detect":
        return ()

    active = settings or get_settings()
    forced = active.whisper_language.strip().lower()
    if forced:
        return (forced,)

    # Język odpowiedzi asystenta. „auto" po tej stronie też znaczy „zgaduj".
    conversation = active.language.strip().lower()
    if conversation and conversation != "auto":
        return (conversation,)
    return ()


def resolve_speech_language(
    settings: Settings | None = None, user_settings: UserSettings | None = None
) -> str:
    """GŁÓWNY język mowy — pierwszy z listy. ``""`` = rozpoznawaj sam.

    Używany tam, gdzie potrzebna jest jedna wartość: wybór głosu TTS i domyślny
    język odpowiedzi. Rozpoznawanie mowy korzysta z pełnej listy
    (:func:`resolve_speech_languages`).
    """
    languages = resolve_speech_languages(settings, user_settings)
    return languages[0] if len(languages) == 1 else ""


def configured_reply_language(
    settings: Settings | None = None, user_settings: UserSettings | None = None
) -> str:
    """USTAWIONY język odpowiedzi asystenta. ``"auto"`` = tak jak padło pytanie.

    Zwraca samo ustawienie. Rozstrzygnięcie ``auto`` na podstawie treści
    wypowiedzi robi :func:`brain.personality.resolve_reply_language`.

    To co innego niż :func:`resolve_speech_languages` — tam chodzi o język, w
    którym użytkownik MÓWI. Rozdzielenie wzięło się z prawdziwego zgłoszenia:
    „jak mówię po pl to wykrywa, lecz nie odpowiada po ang". Użytkownik mówił po
    polsku i po angielsku, a odpowiedzi chciał po angielsku — jedno pole nie
    umiało opisać obu rzeczy naraz, bo ``speech_language`` szło wprost jako
    język odpowiedzi (przy liście „pl,en" prompt schodził na polski, bo to nie
    jest kod języka).

    Kolejność:

    1. ``LANGUAGE`` **ustawione jawnie** (w ``.env`` albo w środowisku) — wygrywa,
       bo to świadoma decyzja, a nie wartość domyślna,
    2. język mowy użytkownika, jeśli podał JEDEN kod (nie listę) — dzięki temu
       ktoś, kto ustawił tylko ``"speech_language": "pl"`` i nigdy nie dotknął
       ``.env``, dostaje polskie odpowiedzi tak jak dotąd,
    3. domyślny język aplikacji.
    """
    active = settings or get_settings()
    if "language" in active.model_fields_set:
        return active.language

    primary = resolve_speech_language(active, user_settings)
    return primary or active.language


def get_user_settings() -> UserSettings:
    """Aktualne ustawienia użytkownika (przeładowuje plik, jeśli zmienił się na dysku)."""
    return load_user_settings()


def reload_user_settings() -> UserSettings:
    """Wymuś ponowny odczyt ``config/user_settings.json``."""
    return load_user_settings(force_reload=True)


def save_user_settings(
    updates: Mapping[str, Any] | UserSettings, path: Path | None = None
) -> UserSettings:
    """Zapisz zmiany zachowując pozostałą zawartość pliku.

    GUI może nadpisać pojedyncze pole (także zagnieżdżone, np. ``{"rvc": {"pitch_shift": 2}}``)
    bez utraty reszty ustawień i bez utraty kluczy, których ten model nie zna.
    """
    global _user_settings_cache, _user_settings_mtime

    payload = updates.model_dump() if isinstance(updates, UserSettings) else dict(updates)
    target = path or USER_SETTINGS_FILE

    with _user_settings_lock:
        raw = _read_raw_user_settings(target)
        merged = _deep_merge(raw, payload)

        try:
            validated = UserSettings.model_validate(merged)
        except ValidationError as exc:
            raise ConfigError(
                t("cfg.not_saved", details=_format_validation_error(exc)),
                hint=t("cfg.check_saved_values", path=target),
            ) from exc

        to_write = _deep_merge(merged, validated.model_dump())
        try:
            _write_json_atomic(target, to_write)
        except OSError as exc:
            raise ConfigError(
                t("cfg.write_failed", path=target, error=exc),
                hint=t("cfg.write_hint"),
            ) from exc

        _user_settings_cache = validated
        _user_settings_mtime = _file_mtime_ns(target)
        return validated


# --------------------------------------------------------------------------- #
# Detekcja zależności
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PackageRequirement:
    """Pakiet z ``requirements.txt`` sprawdzany przez :func:`detect_dependencies`."""

    distribution: str
    module: str
    required: bool = True
    phase: int = 1
    purpose: str = ""


# Lista jest wspólna dla wszystkich faz — kolejne fazy DOPISUJĄ tu swoje pakiety
# (albo wołają register_package_requirement), zamiast budować własną detekcję.
PACKAGE_REQUIREMENTS: list[PackageRequirement] = [
    PackageRequirement("pydantic", "pydantic", purpose="deps.purpose.pydantic"),
    PackageRequirement(
        "pydantic-settings", "pydantic_settings", purpose="deps.purpose.pydantic_settings"
    ),
    PackageRequirement("python-dotenv", "dotenv", purpose="deps.purpose.dotenv"),
    PackageRequirement("httpx", "httpx", purpose="deps.purpose.httpx"),
    # --- Faza 2: rozpoznawanie mowy ---
    PackageRequirement("numpy", "numpy", required=False, phase=2, purpose="deps.purpose.numpy"),
    PackageRequirement(
        "sounddevice",
        "sounddevice",
        required=False,
        phase=2,
        purpose="deps.purpose.sounddevice",
    ),
    PackageRequirement(
        "faster-whisper",
        "faster_whisper",
        required=False,
        phase=2,
        purpose="deps.purpose.faster_whisper",
    ),
    PackageRequirement(
        "webrtcvad-wheels",
        "webrtcvad",
        required=False,
        phase=2,
        purpose="deps.purpose.webrtcvad",
    ),
    # --- Faza 3: słowo aktywujące ---
    PackageRequirement(
        "openwakeword",
        "openwakeword",
        required=False,
        phase=3,
        purpose="deps.purpose.openwakeword",
    ),
    # --- Faza 4: synteza mowy ---
    PackageRequirement(
        "piper-tts",
        "piper",
        required=False,
        phase=4,
        purpose="deps.purpose.piper",
    ),
    # --- Faza 6: pamięć semantyczna ---
    PackageRequirement(
        "sentence-transformers",
        "sentence_transformers",
        required=False,
        phase=6,
        purpose="deps.purpose.sentence_transformers",
    ),
    PackageRequirement(
        "faiss-cpu",
        "faiss",
        required=False,
        phase=6,
        purpose="deps.purpose.faiss",
    ),
]


def register_package_requirement(requirement: PackageRequirement) -> PackageRequirement:
    """Dopisz pakiet do wspólnej listy sprawdzeń (używane przez kolejne fazy)."""
    if all(existing.module != requirement.module for existing in PACKAGE_REQUIREMENTS):
        PACKAGE_REQUIREMENTS.append(requirement)
    return requirement


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """Pojedynczy sprawdzany element środowiska."""

    name: str
    category: str
    required: bool
    ok: bool
    detail: str = ""
    path: str | None = None
    hint: str = ""
    phase: int = 1

    @property
    def status(self) -> Literal["ok", "brak"]:
        return "ok" if self.ok else "brak"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "phase": self.phase,
            "required": self.required,
            "status": self.status,
            "detail": self.detail,
            "path": self.path,
            "hint": self.hint,
        }


@dataclass(frozen=True, slots=True)
class DependencyContext:
    """Dane wejściowe dla sprawdzeń dopisywanych przez kolejne fazy."""

    settings: Settings
    platform_info: PlatformInfo
    gpu: GPUInfo
    ollama: OllamaStatus
    user_settings: UserSettings
    # Rozstrzygnięty tryb pracy — sprawdzenia kolejnych faz mają go dostać
    # gotowego, zamiast liczyć go po raz drugi każde z osobna.
    offline: bool = False


DependencyCheckFn = Callable[[DependencyContext], Sequence[DependencyCheck]]
_EXTRA_DEPENDENCY_CHECKS: list[DependencyCheckFn] = []


def register_dependency_check(function: DependencyCheckFn) -> DependencyCheckFn:
    """Zarejestruj dodatkowe sprawdzenie (dekorator dla Faz 2, 3, 4, 15...)."""
    if function not in _EXTRA_DEPENDENCY_CHECKS:
        _EXTRA_DEPENDENCY_CHECKS.append(function)
    return function


@dataclass(frozen=True, slots=True)
class DependencyReport:
    generated_at: str
    platform_info: PlatformInfo
    gpu: GPUInfo
    ollama: OllamaStatus
    checks: tuple[DependencyCheck, ...] = field(default_factory=tuple)
    offline: bool = False

    @property
    def missing_required(self) -> tuple[DependencyCheck, ...]:
        return tuple(check for check in self.checks if check.required and not check.ok)

    @property
    def missing_optional(self) -> tuple[DependencyCheck, ...]:
        return tuple(check for check in self.checks if not check.required and not check.ok)

    @property
    def ok(self) -> bool:
        return not self.missing_required

    @property
    def chat_available(self) -> bool:
        """Czy da się prowadzić rozmowę tekstową (Ollama + model + klient HTTP)."""
        http_ready = all(
            check.ok
            for check in self.checks
            if check.category == "package" and check.name.endswith("httpx")
        )
        return self.ollama.reachable and self.ollama.model_present and http_ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "app": {"name": APP_NAME, "version": APP_VERSION},
            "project_root": str(PROJECT_ROOT),
            "summary": {
                "total": len(self.checks),
                "ok": sum(1 for check in self.checks if check.ok),
                "missing": sum(1 for check in self.checks if not check.ok),
                "missing_required": len(self.missing_required),
                "chat_available": self.chat_available,
                "offline": self.offline,
            },
            "platform": self.platform_info.to_dict(),
            "gpu": self.gpu.to_dict(),
            "ollama": self.ollama.to_dict(),
            "install_script": self.platform_info.install_script,
            "checks": [check.to_dict() for check in self.checks],
        }


def _module_location(module: str) -> tuple[bool, str | None]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError, AttributeError):
        return False, None
    if spec is None:
        return False, None
    origin = spec.origin
    if origin in (None, "built-in", "frozen") and spec.submodule_search_locations:
        origin = next(iter(spec.submodule_search_locations), None)
    return True, origin


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - uszkodzone metadane środowiska
        return None


def _check_directory_writable(name: str, directory: Path) -> DependencyCheck:
    probe = directory / f".write-test-{os.getpid()}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return DependencyCheck(
            name=name,
            category="filesystem",
            required=True,
            ok=False,
            detail=t("deps.dir.no_write", error=exc),
            path=str(directory),
            hint=t("deps.dir.hint"),
        )
    return DependencyCheck(
        name=name,
        category="filesystem",
        required=True,
        ok=True,
        detail=t("deps.dir.writable"),
        path=str(directory),
    )


def _check_ollama_compute(
    settings: Settings,
    ollama: OllamaStatus,
    gpu: GPUInfo,
    platform_info: PlatformInfo,
) -> DependencyCheck:
    """Czy załadowany model siedzi w pamięci karty, czy liczy się na CPU.

    Odpowiedź daje ``/api/ps``: pole ``size_vram`` mówi, ile modelu leży w VRAM.
    Zero przy obecnym GPU znaczy, że Ollama nie ma obsługi CUDA — i to jest
    najczęstsza przyczyna wolnych odpowiedzi, której nie widać nigdzie indziej.
    """
    if not ollama.reachable:
        return DependencyCheck(
            name=t("deps.compute.name"),
            category="hardware",
            required=False,
            ok=False,
            detail=t("deps.ollama.unreachable"),
            phase=1,
        )

    data, _error = _http_get_json(f"{settings.ollama_host}/api/ps", settings.ollama_connect_timeout)
    loaded = data.get("models", []) if isinstance(data, dict) else []
    in_vram = 0
    for entry in loaded:
        if isinstance(entry, dict):
            try:
                in_vram += int(entry.get("size_vram") or 0)
            except (TypeError, ValueError):  # pragma: no cover - nietypowa odpowiedź
                continue

    if not loaded:
        # Bez załadowanego modelu nie ma czego mierzyć — i nie zgadujemy.
        return DependencyCheck(
            name=t("deps.compute.name"),
            category="hardware",
            required=False,
            ok=True,
            detail=t("deps.compute.unknown"),
            phase=1,
        )

    if in_vram > 0:
        return DependencyCheck(
            name=t("deps.compute.name"),
            category="hardware",
            required=False,
            ok=True,
            detail=t("deps.compute.gpu", size=f"{in_vram / 1024**3:.1f} GiB"),
            phase=1,
        )

    hint = ""
    if gpu.cuda_available:
        detail = t("deps.compute.cpu_with_gpu", gpu=gpu.device_name or "GPU")
        hint = (
            t("deps.compute.hint_pacman")
            if platform_info.package_manager is PackageManager.PACMAN
            else t("deps.compute.hint_generic")
        )
    else:
        detail = t("deps.compute.cpu")
    return DependencyCheck(
        name=t("deps.compute.name"),
        category="hardware",
        # Praca na CPU nie jest awarią — jest wolna. Dlatego „brak", a nie błąd
        # wymagany: asystent działa, tylko dłużej każe na siebie czekać.
        required=False,
        ok=not gpu.cuda_available,
        detail=detail,
        hint=hint,
        phase=1,
    )


def detect_dependencies(
    settings: Settings | None = None, *, write_report: bool = True
) -> DependencyReport:
    """Sprawdź środowisko i zapisz raport do ``config/dependency_status.json``.

    Raport jest nadpisywany przy każdym uruchomieniu — to cache/diagnostyka,
    nie źródło prawdy. Zapis następuje ZAWSZE, niezależnie od wyniku.
    """
    ensure_directories()

    active = settings or get_settings()
    platform_info = detect_platform()
    gpu = detect_cuda()
    ollama = detect_ollama(active)
    user_settings = load_user_settings()
    install_hint = install_instruction(platform_info)
    offline = is_offline(active)
    pip_hint = pip_install_hint(offline)

    checks: list[DependencyCheck] = []

    # --- Tryb pracy (offline/online) ---
    checks.append(
        DependencyCheck(
            name=t("deps.mode.name"),
            category="mode",
            required=False,
            ok=True,
            detail=describe_offline_mode(active),
            hint="" if offline else t("deps.mode.hint"),
        )
    )

    # --- Python ---
    python_ok = sys.version_info[:2] >= REQUIRED_PYTHON
    checks.append(
        DependencyCheck(
            name=t("deps.python.name"),
            category="python",
            required=True,
            ok=python_ok,
            detail=t(
                "deps.python.detail",
                required=f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
                detected=platform_info.python_version,
            ),
            path=platform_info.python_executable,
            hint="" if python_ok else install_hint,
        )
    )

    # --- Pakiety z requirements.txt ---
    for requirement in PACKAGE_REQUIREMENTS:
        found, origin = _module_location(requirement.module)
        version = _distribution_version(requirement.distribution) if found else None
        purpose = translate_or_text(requirement.purpose) if requirement.purpose else ""
        detail = purpose
        if found and version:
            detail = (
                t("deps.package.version_purpose", version=version, purpose=purpose)
                if purpose
                else t("deps.package.version", version=version)
            )
        elif not found:
            detail = (
                t("deps.package.missing_purpose", purpose=purpose)
                if purpose
                else t("deps.package.missing")
            )
        checks.append(
            DependencyCheck(
                name=t("deps.package.name", name=requirement.distribution),
                category="package",
                required=requirement.required,
                ok=found,
                detail=detail,
                path=origin,
                hint="" if found else f"{pip_hint}  ({install_hint})",
                phase=requirement.phase,
            )
        )

    # --- Magazyn kół pip (instalacja bez internetu) ---
    wheels = wheelhouse_packages()
    checks.append(
        DependencyCheck(
            name=t("deps.wheels.name"),
            category="package",
            required=False,
            ok=wheels > 0,
            detail=(t("deps.wheels.present", count=wheels) if wheels else t("deps.wheels.missing")),
            path=str(WHEELHOUSE_DIR),
            hint="" if wheels else t("deps.wheels.hint"),
        )
    )

    # --- Ollama: aplikacja (opcjonalna — serwer może stać na innej maszynie) ---
    checks.append(
        DependencyCheck(
            name=t("deps.ollama.app"),
            category="service",
            required=False,
            ok=ollama.binary_path is not None,
            detail=(
                t("deps.ollama.in_path") if ollama.binary_path else t("deps.ollama.not_in_path")
            ),
            path=ollama.binary_path,
            hint="" if ollama.binary_path else install_hint,
        )
    )

    # --- Ollama: usługa HTTP ---
    checks.append(
        DependencyCheck(
            name=t("deps.ollama.service"),
            category="service",
            required=True,
            ok=ollama.reachable,
            detail=(
                t(
                    "deps.ollama.responds",
                    version=ollama.version or t("deps.gpu.unknown_driver"),
                )
                if ollama.reachable
                else (ollama.error or t("deps.ollama.no_answer"))
            ),
            path=ollama.host,
            hint="" if ollama.reachable else t("deps.ollama.start_hint", install=install_hint),
        )
    )

    # --- Ollama: model ---
    if ollama.reachable:
        model_detail = (
            t("deps.model.present")
            if ollama.model_present
            else t(
                "deps.model.absent",
                models=(
                    ", ".join(ollama.models[:10]) if ollama.models else t("deps.model.no_models")
                ),
            )
        )
    else:
        model_detail = t("deps.model.not_checked")
    checks.append(
        DependencyCheck(
            name=t("deps.model.name", model=active.ollama_model),
            category="model",
            required=True,
            ok=ollama.model_present,
            detail=model_detail,
            path=ollama.host,
            hint=(
                ""
                if ollama.model_present
                else (
                    f"ollama pull {active.ollama_model}"
                    + (t("deps.model.offline_note") if offline else "")
                )
            ),
        )
    )

    # --- Katalogi robocze ---
    checks.append(_check_directory_writable(t("deps.dir.config"), CONFIG_DIR))
    checks.append(_check_directory_writable(t("deps.dir.logs"), LOGS_DIR))

    # --- GPU (nigdy wymagane) ---
    checks.append(
        DependencyCheck(
            name=t("deps.cuda.name"),
            category="hardware",
            required=False,
            ok=gpu.cuda_available,
            detail=gpu.detail,
            path=None,
            hint="" if gpu.cuda_available else t("deps.cuda.hint"),
        )
    )

    # --- Czym Ollama faktycznie liczy: GPU czy CPU ---
    # To jedno pytanie potrafi wyjaśnić „dlaczego asystent tak długo myśli".
    # Sam fakt, że w komputerze jest karta, niczego nie gwarantuje: pakiet Ollamy
    # bywa zbudowany bez CUDA i wtedy model liczy się na procesorze kilka razy
    # wolniej, bez żadnego komunikatu.
    checks.append(_check_ollama_compute(active, ollama, gpu, platform_info))

    # --- Sprawdzenia dopisane przez kolejne fazy ---
    context = DependencyContext(
        settings=active,
        platform_info=platform_info,
        gpu=gpu,
        ollama=ollama,
        user_settings=user_settings,
        offline=offline,
    )
    for extra_check in _EXTRA_DEPENDENCY_CHECKS:
        try:
            checks.extend(extra_check(context))
        except Exception as exc:  # jedno wadliwe sprawdzenie nie psuje raportu
            logger.warning(
                "Sprawdzenie zależności %s zgłosiło błąd: %s",
                getattr(extra_check, "__name__", repr(extra_check)),
                exc,
            )

    report = DependencyReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        platform_info=platform_info,
        gpu=gpu,
        ollama=ollama,
        checks=tuple(checks),
        offline=offline,
    )

    if write_report:
        try:
            _write_json_atomic(DEPENDENCY_STATUS_FILE, report.to_dict())
        except OSError as exc:
            logger.warning("Nie udało się zapisać raportu %s: %s", DEPENDENCY_STATUS_FILE, exc)

    return report


def load_dependency_report() -> dict[str, Any] | None:
    """Odczytaj ostatni zapisany raport (używane np. przez GUI z Fazy 10)."""
    try:
        content = DEPENDENCY_STATUS_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "CONFIG_DIR",
    "DEFAULT_DATABASE_NAME",
    "DEPENDENCY_STATUS_FILE",
    "EMBEDDINGS_DIR",
    "ENV_FILE",
    "ERROR_LOG_FILE",
    "LOGS_DIR",
    "LOG_FILE",
    "MEMORY_DATABASE",
    "MODELS_DIR",
    "OFFLINE_BUNDLE_FILE",
    "PIPER_DIR",
    "PROJECT_ROOT",
    "REQUIRED_PYTHON",
    "TAG_ERROR",
    "TAG_MIC",
    "TAG_SYSTEM",
    "TAG_TOOL",
    "TAG_USER",
    "TAG_WAKE",
    "USER_SETTINGS_EXAMPLE_FILE",
    "USER_SETTINGS_FILE",
    "WAKEWORD_DIR",
    "WHEELHOUSE_DIR",
    "WHISPER_CACHE_DIR",
    "ConfigError",
    "DependencyCheck",
    "DependencyContext",
    "DependencyReport",
    "GPUInfo",
    "OSFamily",
    "OllamaStatus",
    "PackageManager",
    "PackageRequirement",
    "PlatformInfo",
    "RVCSettings",
    "Settings",
    "UserSettings",
    "app_data_directory",
    "apply_offline_environment",
    "database_file",
    "describe_offline_mode",
    "detect_cuda",
    "detect_dependencies",
    "detect_ollama",
    "detect_platform",
    "ensure_directories",
    "find_local_embedding_model",
    "find_local_whisper_model",
    "find_piper_binary",
    "get_settings",
    "get_user_settings",
    "home_directory",
    "install_instruction",
    "is_local_host",
    "is_offline",
    "iter_local_whisper_models",
    "load_dependency_report",
    "load_user_settings",
    "path_from_env",
    "pip_install_hint",
    "piper_voice_directories",
    "register_dependency_check",
    "register_package_requirement",
    "reload_user_settings",
    "resolve_compute_device",
    "configured_reply_language",
    "resolve_speech_language",
    "resolve_speech_languages",
    "save_user_settings",
    "subprocess_no_window_kwargs",
    "user_data_directories",
    "wheelhouse_packages",
]
