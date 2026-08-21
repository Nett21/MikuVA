"""Konwersja barwy głosu metodą RVC (Faza 15) — warstwa niezależna od TTS.

Ten moduł nie wie nic o Piperze, o zdaniach ani o karcie dźwiękowej. Umie
jedną rzecz: wziąć kawałek PCM i oddać ten sam kawałek powiedziany innym
głosem. Opakowanie tego w dostawcę mowy siedzi w ``audio/tts_rvc.py``.

**Modelu RVC nie ma w tym repozytorium i nigdy nie będzie.** Pliki ``.pth``
i ``.index`` są własnością tego, kto je wytrenował, a głos Hatsune Miku jest
obciążony prawami Crypton Future Media (patrz sekcja Ograniczeń w README).
Kod czyta wyłącznie ścieżki podane przez użytkownika w
``config/user_settings.json``.

Backendu też tu nie ma. Implementacji RVC krąży po świecie kilka i żadna nie
jest kanoniczna — jedne są pakietem pip, inne katalogiem sklonowanym z GitHuba.
Dlatego moduł definiuje **kontrakt** (:class:`RvcBackend`) i szuka czegoś, co
go spełnia:

1. ``RVC_BACKEND`` wskazujący konkretną nazwę albo moduł — wygrywa zawsze,
2. **Applio**, jeśli jest — domyślne, bo najszybsze (liczby są w README),
3. ``rvc-python`` w osobnym środowisku, jako zapas,
4. nic — i to nie jest błąd, tylko powrót do zwykłego głosu Pipera.

Obie implementacje liczą w OSOBNYM procesie i osobnym środowisku Pythona.
Nie jest to ostrożność, tylko konieczność: ``rvc-python`` stoi na ``fairseq``,
który nie działa powyżej Pythona 3.10, Applio wymaga 3.12+ i własnego katalogu
roboczego, a asystent chodzi na 3.12+. Żadnej z nich nie da się uruchomić
w tym procesie — dlatego adaptera „w procesie" tu nie ma.

Własny backend to moduł Pythona z funkcją ``create_backend(model_path,
index_path, device)`` zwracającą obiekt z metodą ``convert``. Wskazuje się go
przez ``RVC_BACKEND=moj_pakiet.moj_modul``. Nic poza tym nie jest wymagane —
świadomie, bo inaczej każda nowa implementacja RVC wymagałaby zmiany w tym
pliku.

Czego ten moduł NIE robi:

* nie decyduje, kiedy się poddać — od tego jest dostawca w ``audio/tts_rvc.py``,
* nie zakłada, że jest GPU: urządzenie wybiera :func:`config.resolve_compute_device`,
  a praca na CPU kończy się ostrzeżeniem w logu, nie odmową,
* nie zakłada żadnej ścieżki na dysku poza tymi z ustawień użytkownika
  i katalogiem plików tymczasowych systemu (``tempfile``).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import subprocess  # nosec B404 - wyłącznie własny skrypt, bez powłoki
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from config import (
    PROJECT_ROOT,
    GPUInfo,
    RVCSettings,
    Settings,
    detect_cuda,
    get_settings,
    resolve_compute_device,
    subprocess_no_window_kwargs,
)
from i18n import t

logger = logging.getLogger(__name__)

# Nazwa backendu wbudowanego — jedyna, którą ten plik zna z nazwy.
BACKEND_RVC_PYTHON: Final[str] = "rvc_python"
# Backend liczący w OSOBNYM procesie i osobnym środowisku Pythona.
BACKEND_SUBPROCESS: Final[str] = "subprocess"
# Applio: ten sam RVC bez `fairseq`, w osobnym procesie i własnym katalogu.
BACKEND_APPLIO: Final[str] = "applio"
# Skrypt stawiający domyślny backend. W podpowiedzi musi paść nazwa skryptu,
# a nie `pip install ...`: Applio nie jest pakietem z PyPI, tylko repozytorium
# do sklonowania razem z wagami.
INSTALL_APPLIO_SCRIPT: Final[str] = "./scripts/install-applio.sh"


class RvcError(RuntimeError):
    """Błąd konwersji głosu z komunikatem gotowym dla użytkownika."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n" + t("cli.voice.hint", detail=self.hint)
        return self.message


class RvcUnavailableError(RvcError):
    """RVC nie da się uruchomić tutaj: brak backendu, brak pliku modelu, brak pamięci.

    Zawsze kończy się powrotem do zwykłego głosu — nigdy wywróceniem programu.
    """


# --------------------------------------------------------------------------- #
# Kontrakt backendu
# --------------------------------------------------------------------------- #


@runtime_checkable
class RvcBackend(Protocol):
    """Minimum, jakiego wymagamy od implementacji RVC.

    Celowo wąskie: jedna metoda robocza i sprzątanie. Wszystko, co ponad to
    (ładowanie modelu, wybór urządzenia, cache indeksu), jest sprawą wewnętrzną
    backendu — dostawca mowy nie ma prawa o tym wiedzieć.
    """

    name: str

    def convert(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> tuple[np.ndarray, int]:
        """Zamień barwę. Wejście i wyjście: mono ``int16``. Wyjście ma WŁASNĄ częstotliwość."""
        ...

    def close(self) -> None:
        """Zwolnij model i pamięć GPU. Wywołanie na już zamkniętym jest bezpieczne."""
        ...


@dataclass(frozen=True, slots=True)
class RvcDevice:
    """Urządzenie wybrane dla RVC wraz z powodem — do logu i do ``--check-deps``."""

    name: str
    gpu: GPUInfo

    @property
    def is_cpu(self) -> bool:
        return not self.name.startswith("cuda")

    def describe(self) -> str:
        if self.is_cpu:
            return t("rvc.device.cpu")
        return t("rvc.device.gpu", name=self.gpu.device_name or self.name)


def resolve_rvc_device(settings: Settings, gpu: GPUInfo | None = None) -> RvcDevice:
    """Na czym liczyć konwersję.

    Nie ma tu żadnej własnej logiki wykrywania sprzętu — pytamy ``config.py``,
    tak samo jak robi to Whisper i embeddingi. Jedna funkcja wykrywająca CUDA
    w całym projekcie oznacza, że wszystkie warstwy widzą ten sam sprzęt.
    """
    info = gpu if gpu is not None else detect_cuda()
    device = resolve_compute_device(settings.rvc_device, info)
    # `rvc_python` (i większość implementacji opartych o torch) oczekuje indeksu
    # karty, nie samego słowa „cuda". Bez numeru część z nich wraca na CPU.
    if device == "cuda":
        device = "cuda:0"
    return RvcDevice(name=device, gpu=info)


# --------------------------------------------------------------------------- #
# Zamiana PCM ↔ plik WAV
# --------------------------------------------------------------------------- #
#
# Backendy RVC pracują na PLIKACH — taki mają interfejs i nie mamy na to wpływu.
# Przy fragmentach rzędu pół sekundy to kilkadziesiąt kilobajtów na przebieg,
# w katalogu tymczasowym systemu (`tempfile` honoruje TMPDIR na Linuksie i
# TEMP na Windowsie, więc nie zakładamy tu żadnej ścieżki).


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Zapisz mono ``int16`` do pliku WAV."""
    data = np.ascontiguousarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(data.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Wczytaj WAV jako mono ``int16``.

    Backend może oddać stereo albo 32 bity — sprowadzamy do wspólnego formatu
    zamiast wymagać, żeby oddawał dokładnie to, co my.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.int16)
    elif width == 4:
        # Część narzędzi zapisuje float32 w zakresie [-1, 1], część int32.
        as_float = np.frombuffer(raw, dtype=np.float32)
        if as_float.size and np.max(np.abs(as_float)) <= 1.5:
            data = np.clip(as_float * 32767.0, -32768, 32767).astype(np.int16)
        else:
            data = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
    elif width == 1:
        data = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    else:  # pragma: no cover - formaty spoza tej trójki po prostu nie występują
        raise RvcUnavailableError(t("rvc.wav_unsupported", width=width * 8))

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)
    return data, int(rate)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Przelicz częstotliwość próbkowania.

    Potrzebne, bo model RVC prawie nigdy nie liczy w tym samym tempie co Piper
    (typowo 40 000 albo 48 000 Hz wobec 22 050 Hz). Gdybyśmy oddawali fragmenty
    o zmiennej częstotliwości, warstwa odtwarzania musiałaby przy KAŻDYM
    przełączeniu zamknąć i otworzyć strumień — a to słychać jako kliknięcie
    i gubi to, co zostało w kolejce. Dlatego cały dostawca trzyma jedną
    częstotliwość, a przeliczanie jest tutaj.

    ``scipy`` daje porządny filtr; bez niej wystarczy interpolacja liniowa,
    która brzmi gorzej, ale działa — a scipy nie jest zależnością wymaganą.
    """
    if source_rate == target_rate or samples.size == 0:
        return np.ascontiguousarray(samples, dtype=np.int16)
    if source_rate <= 0 or target_rate <= 0:
        return np.ascontiguousarray(samples, dtype=np.int16)

    try:
        from math import gcd  # noqa: PLC0415 - lokalnie, razem ze scipy

        from scipy.signal import resample_poly  # type: ignore[import-untyped] # noqa: PLC0415

        divisor = gcd(int(source_rate), int(target_rate))
        converted = resample_poly(
            samples.astype(np.float32),
            int(target_rate) // divisor,
            int(source_rate) // divisor,
        )
    except Exception:  # brak scipy albo nietypowe wymiary
        length = max(1, round(samples.size * target_rate / source_rate))
        source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
        target_x = np.linspace(0.0, 1.0, num=length, endpoint=False, dtype=np.float64)
        converted = np.interp(target_x, source_x, samples.astype(np.float64))

    limited: np.ndarray = np.clip(converted, -32768, 32767).astype(np.int16)
    return limited


# --------------------------------------------------------------------------- #
# Backend oparty o pliki
# --------------------------------------------------------------------------- #


class FileRvcBackend:
    """Backend, który rozmawia z modelem przez pliki WAV.

    Cała robota z plikami tymczasowymi jest tutaj, żeby adapter konkretnej
    biblioteki sprowadzał się do jednej metody: „przerób ten plik na tamten".
    """

    name = "file"

    def __init__(self, *, name: str = "file") -> None:
        self.name = name
        self._workdir: tempfile.TemporaryDirectory[str] | None = None
        self._counter = 0

    # --- do nadpisania ------------------------------------------------------- #

    def infer_file(
        self,
        source: Path,
        target: Path,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> None:  # pragma: no cover - klasa bazowa nic nie umie
        raise NotImplementedError

    # --- kontrakt RvcBackend ------------------------------------------------- #

    def _directory(self) -> Path:
        if self._workdir is None:
            self._workdir = tempfile.TemporaryDirectory(prefix="miku-rvc-")
        return Path(self._workdir.name)

    def convert(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> tuple[np.ndarray, int]:
        directory = self._directory()
        self._counter += 1
        source = directory / f"in-{self._counter:06d}.wav"
        target = directory / f"out-{self._counter:06d}.wav"
        try:
            write_wav(source, samples, sample_rate)
            self.infer_file(source, target, pitch_shift=pitch_shift, index_rate=index_rate)
            if not target.is_file():
                raise RvcUnavailableError(t("rvc.no_output"))
            return read_wav(target)
        finally:
            # Fragmentów jest kilkaset na rozmowę — bez sprzątania katalog
            # tymczasowy rósłby przez cały czas działania asystenta.
            for path in (source, target):
                try:
                    path.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - zajęty plik na Windowsie
                    logger.debug("Nie udało się usunąć %s", path)

    def close(self) -> None:
        if self._workdir is not None:
            try:
                self._workdir.cleanup()
            except OSError:  # pragma: no cover
                logger.debug("Katalog tymczasowy RVC został w systemie")
            self._workdir = None


# --------------------------------------------------------------------------- #
# Backend w osobnym procesie i osobnym środowisku Pythona
# --------------------------------------------------------------------------- #


def rvc_worker_script() -> Path:
    """Skrypt pracownika. Leży w ``scripts/``, obok instalatorów."""
    return PROJECT_ROOT / "scripts" / "rvc_worker.py"


def find_rvc_worker_python(settings: Settings) -> Path | None:
    """Interpreter środowiska, w którym mieszka RVC.

    Kolejność: ustawienie użytkownika, potem ``.venv-rvc`` w katalogu projektu.
    Nazwa katalogu z binarkami różni się między systemami (``bin`` na Uniksach,
    ``Scripts`` na Windowsie), więc sprawdzamy obie — bez pytania o system,
    bo istnienie pliku jest odpowiedzią pewniejszą niż nazwa platformy.
    """
    wskazany = settings.rvc_worker_python.strip()
    if wskazany:
        candidate = Path(os.path.expandvars(wskazany)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate if candidate.is_file() else None

    for relative in (
        Path(".venv-rvc") / "bin" / "python",
        Path(".venv-rvc") / "Scripts" / "python.exe",
    ):
        candidate = PROJECT_ROOT / relative
        if candidate.is_file():
            return candidate
    return None


def find_applio_python(settings: Settings) -> Path | None:
    """Interpreter środowiska Applio.

    Osobny od :func:`find_rvc_worker_python`, bo to dwa niekompatybilne
    światy: `rvc-python` stoi na ``fairseq`` i Pythonie 3.10, Applio przypina
    ``torch==2.11`` i wymaga 3.12+. Wspólny venv nie istnieje.
    """
    wskazany = settings.rvc_applio_python.strip()
    if wskazany:
        candidate = Path(os.path.expandvars(wskazany)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate if candidate.is_file() else None

    for relative in (
        Path(".venv-applio") / "bin" / "python",
        Path(".venv-applio") / "Scripts" / "python.exe",
    ):
        candidate = PROJECT_ROOT / relative
        if candidate.is_file():
            return candidate
    return None


def find_applio_root(settings: Settings) -> Path | None:
    """Katalog z kodem Applio.

    Sprawdzamy obecność podkatalogu ``rvc``, a nie samego katalogu: pusty
    albo przerwany w połowie klon istnieje na dysku, ale nie da się z niego
    nic zaimportować, a błąd wyszedłby dopiero przy pierwszym zdaniu.
    """
    wskazany = settings.rvc_applio_path.strip()
    if wskazany:
        candidate = Path(os.path.expandvars(wskazany)).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        candidate = PROJECT_ROOT / "third_party" / "Applio"
    return candidate if (candidate / "rvc").is_dir() else None


class SubprocessRvcBackend(FileRvcBackend):
    """RVC liczone przez proces-pracownika w drugim środowisku Pythona.

    Istnieje, bo biblioteki RVC nie działają na Pythonie nowszym niż 3.10,
    a asystent działa na 3.12+. Zamiast cofać cały projekt o cztery wersje,
    rozmawiamy z drugim interpreterem przez potok — po jednym obiekcie JSON
    w linii. Szczegóły protokołu są w ``scripts/rvc_worker.py``.

    Model ładuje się RAZ, przy starcie procesu, i zostaje w pamięci. Gdyby
    proces startował na fragment, sam jego rozruch kosztowałby więcej niż
    wszystkie konwersje razem.
    """

    name = BACKEND_SUBPROCESS

    def __init__(
        self,
        model_path: Path,
        index_path: Path | None,
        device: str,
        *,
        settings: Settings,
        interpreter: Path | None = None,
        extra_args: Sequence[str] = (),
        working_dir: Path | None = None,
        name: str = BACKEND_SUBPROCESS,
    ) -> None:
        # Parametry po `settings` istnieją dla Applio, które jedzie tym samym
        # protokołem i tym samym skryptem, ale innym interpreterem i — co
        # najważniejsze — z innego katalogu roboczego. Domyślne wartości dają
        # dokładnie dotychczasowe zachowanie backendu `subprocess`.
        super().__init__(name=name)
        if interpreter is None:
            interpreter = find_rvc_worker_python(settings)
        if interpreter is None:
            raise RvcUnavailableError(
                t("rvc.no_worker_python"), hint=t("rvc.no_worker_python_hint")
            )
        script = rvc_worker_script()
        if not script.is_file():  # pragma: no cover - plik jest w repozytorium
            raise RvcUnavailableError(t("rvc.no_worker_script", path=str(script)))

        command = [
            str(interpreter),
            str(script),
            "--model",
            str(model_path),
            "--device",
            device,
        ]
        if index_path is not None:
            command += ["--index", str(index_path)]
        command += list(extra_args)

        started = time.monotonic()
        try:
            self._process = subprocess.Popen(  # nosec B603 - własny skrypt, lista argumentów, bez powłoki
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=str(working_dir or PROJECT_ROOT),
                **subprocess_no_window_kwargs(),
            )
        except OSError as exc:
            raise RvcUnavailableError(
                t("rvc.worker_spawn_failed", path=str(interpreter), detail=str(exc))
            ) from exc

        self._lock = threading.Lock()
        # Hojniej niż limit zewnętrzny z RvcConverter, bo tamten i tak przerwie
        # całość wcześniej. Ten istnieje po to, żeby nie czekać w nieskończoność
        # na pracownika, który przestał odpowiadać, ale procesu nie zamknął.
        self._read_timeout = max(settings.rvc_timeout_s * 2, 30.0)
        # Wątek czytający stderr pracownika. Bez niego bufor potoku zapełniłby
        # się gadatliwymi logami torcha i proces stanąłby na zapisie — czyli
        # zawiesiłby mowę w sposób trudny do powiązania z przyczyną.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="rvc-worker-stderr", daemon=True
        )
        self._stderr_thread.start()

        witaj = self._read_line(settings.rvc_worker_start_s)
        if witaj is None:
            self._terminate()
            raise RvcUnavailableError(
                t("rvc.worker_start_timeout", seconds=f"{settings.rvc_worker_start_s:.0f}"),
                hint=t("rvc.worker_start_timeout_hint"),
            )
        if not witaj.get("ready"):
            self._terminate()
            raise RvcUnavailableError(
                t("rvc.worker_load_failed", detail=str(witaj.get("error", "?")))
            )
        logger.info(
            "RVC: %s",
            t(
                "rvc.worker_ready",
                seconds=f"{time.monotonic() - started:.1f}",
                device=str(witaj.get("device", device)),
            ),
        )

    # --- potok --------------------------------------------------------------- #

    def _drain_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:  # pragma: no cover
            return
        for line in stream:
            text = line.rstrip()
            if text:
                logger.debug("rvc-worker: %s", text)

    def _read_line(self, timeout_s: float) -> dict[str, Any] | None:
        """Przeczytaj jedną odpowiedź. ``None`` = nie doczekaliśmy się.

        Czytanie z potoku blokuje i nie da się tego przerwać — więc czyta wątek
        daemon, a my czekamy na niego z limitem czasu. Wątek daemon zginie
        razem z procesem, więc zawieszony pracownik nie zablokuje wyjścia
        z asystenta (patrz komentarz przy ``RvcConverter``).
        """
        slot: list[str] = []

        def czytaj() -> None:
            stream = self._process.stdout
            if stream is None:  # pragma: no cover
                return
            line = stream.readline()
            if line:
                slot.append(line)

        thread = threading.Thread(target=czytaj, name="rvc-worker-read", daemon=True)
        thread.start()
        thread.join(timeout_s)
        if thread.is_alive() or not slot:
            return None
        try:
            wynik: dict[str, Any] = json.loads(slot[0])
        except json.JSONDecodeError:  # pragma: no cover - pracownik chroni stdout
            return None
        return wynik

    def _terminate(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - uparty proces
            process.kill()

    # --- kontrakt FileRvcBackend --------------------------------------------- #

    def infer_file(
        self,
        source: Path,
        target: Path,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> None:
        if self._process.poll() is not None:
            raise RvcUnavailableError(t("rvc.worker_died", code=self._process.returncode))

        request = json.dumps(
            {
                "cmd": "convert",
                "in": str(source),
                "out": str(target),
                "pitch": pitch_shift,
                "index_rate": index_rate,
            }
        )
        # Jeden pracownik, jedno pytanie naraz: bez zamka dwie odpowiedzi
        # mogłyby trafić do niewłaściwych pytających.
        with self._lock:
            stdin = self._process.stdin
            if stdin is None:  # pragma: no cover
                raise RvcUnavailableError(t("rvc.worker_died", code=-1))
            try:
                stdin.write(request + "\n")
                stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RvcUnavailableError(
                    t("rvc.worker_died", code=self._process.returncode or -1)
                ) from exc
            # Limit czasu jest tu hojny, bo ZEWNĘTRZNY limit i tak pilnuje
            # całości w RvcConverter. Ten służy tylko do tego, żeby nie czekać
            # w nieskończoność na pracownika, który przestał odpowiadać.
            answer = self._read_line(timeout_s=self._read_timeout)

        if answer is None:
            raise RvcUnavailableError(t("rvc.worker_no_answer"))
        if not answer.get("ok"):
            raise RvcUnavailableError(
                t("rvc.convert_failed", detail=str(answer.get("error", "?")))
            )

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            with contextlib.suppress(Exception):
                stdin = process.stdin
                if stdin is not None:
                    stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    stdin.flush()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._terminate()
        super().close()


# --------------------------------------------------------------------------- #
# Wybór backendu
# --------------------------------------------------------------------------- #


class ApplioBackend(SubprocessRvcBackend):
    """Applio przez tego samego pracownika, ale z jego własnego katalogu.

    Applio jest szybsze od ``rvc-python`` z powodu, który da się wskazać
    palcem: nie ma ``fairseq``. Cechy mowy liczy embedder z ``transformers``,
    a wysokość tonu — predyktor wybrany w ``RVC_F0_METHOD``.

    Jedna rzecz jest tu nieoczywista i dlatego ma osobną klasę zamiast flagi:
    **katalog roboczy procesu musi być katalogiem Applio**. Jego moduł
    inferencji wykonuje przy imporcie ``now_dir = os.getcwd()`` i po tym
    katalogu szuka wag embeddera oraz predyktora. Uruchomiony skądinąd
    zaimportuje się bez słowa skargi i wywróci dopiero na pierwszej
    konwersji, komunikatem o brakującym pliku — czyli w miejscu, które
    z przyczyną nie ma nic wspólnego.
    """

    name = BACKEND_APPLIO

    def __init__(
        self,
        model_path: Path,
        index_path: Path | None,
        device: str,
        *,
        settings: Settings,
    ) -> None:
        root = find_applio_root(settings)
        if root is None:
            raise RvcUnavailableError(t("rvc.no_applio_path"), hint=t("rvc.no_applio_path_hint"))
        interpreter = find_applio_python(settings)
        if interpreter is None:
            raise RvcUnavailableError(
                t("rvc.no_applio_python"), hint=t("rvc.no_applio_python_hint")
            )
        super().__init__(
            model_path,
            index_path,
            device,
            settings=settings,
            interpreter=interpreter,
            extra_args=[
                "--engine",
                BACKEND_APPLIO,
                "--applio-root",
                str(root),
                "--f0-method",
                settings.rvc_f0_method,
                "--embedder",
                settings.rvc_embedder,
            ],
            working_dir=root,
            name=BACKEND_APPLIO,
        )


def _load_custom_backend(
    reference: str,
    model_path: Path,
    index_path: Path | None,
    device: str,
) -> RvcBackend:
    """Zbuduj backend z modułu wskazanego przez użytkownika.

    Kontrakt jest jednym zdaniem: moduł ma funkcję ``create_backend(model_path,
    index_path, device)`` i zwraca coś z metodą ``convert``. To wystarczy, żeby
    podpiąć dowolną instalację RVC bez dotykania tego pliku.
    """
    try:
        module = importlib.import_module(reference)
    except Exception as exc:
        raise RvcUnavailableError(
            t("rvc.backend_import_failed", backend=reference, detail=str(exc)),
            hint=t("rvc.backend_custom_hint"),
        ) from exc

    factory = getattr(module, "create_backend", None)
    if not callable(factory):
        raise RvcUnavailableError(
            t("rvc.backend_api_mismatch", backend=reference, symbol="create_backend"),
            hint=t("rvc.backend_custom_hint"),
        )

    try:
        backend = factory(model_path=model_path, index_path=index_path, device=device)
    except Exception as exc:
        raise RvcUnavailableError(
            t("rvc.model_load_failed", path=str(model_path), detail=str(exc))
        ) from exc

    if not hasattr(backend, "convert"):
        raise RvcUnavailableError(
            t("rvc.backend_api_mismatch", backend=reference, symbol="convert"),
            hint=t("rvc.backend_custom_hint"),
        )
    if not hasattr(backend, "name"):
        backend.name = reference  # type: ignore[attr-defined]
    checked: RvcBackend = backend
    return checked


def available_rvc_backends(settings: Settings | None = None) -> list[str]:
    """Backendy, które na TEJ maszynie dają się zaimportować.

    Używane przez ``--check-deps`` i przez komunikat o braku RVC. Pusta lista
    nie jest błędem — mówi tylko, że asystent będzie mówił zwykłym Piperem.
    """
    settings = settings if settings is not None else get_settings()
    found: list[str] = []
    # Wszystko liczy się w OSOBNYM procesie — nie z ostrożności, tylko dlatego,
    # że żadna z tych bibliotek nie da się zainstalować obok asystenta: RVC
    # wymaga Pythona 3.10, Applio 3.12+ i własnego katalogu roboczego.
    if rvc_worker_script().is_file():
        if find_applio_python(settings) is not None and find_applio_root(settings) is not None:
            found.append(BACKEND_APPLIO)
        if find_rvc_worker_python(settings) is not None:
            found.append(BACKEND_SUBPROCESS)
    return found


def rvc_backend_chain(settings: Settings) -> list[str]:
    """Backendy do wypróbowania po kolei, gdy poprzedni zawiedzie.

    Kolejność jest ta sama, co przy zwykłym wyborze: najpierw Applio, potem
    `rvc-python` w osobnym środowisku.

    **Wskazany wprost ``RVC_BACKEND`` nie jest podmieniany.** Kto wpisał
    `applio`, prosił o Applio, a nie o „cokolwiek, co akurat ruszy" — cichy
    skok na inny silnik zmieniłby barwę głosu bez pytania. Łańcuch działa
    więc tylko w trybie automatycznym, czyli przy pustym ustawieniu.
    """
    wanted = settings.rvc_backend.strip()
    if wanted:
        return [wanted]
    dostepne = available_rvc_backends(settings)
    kolejka = [nazwa for nazwa in (BACKEND_APPLIO, BACKEND_SUBPROCESS) if nazwa in dostepne]
    # Pusta lista i tak musi skończyć się komunikatem o tym, CZEGO brakuje —
    # a ten składa `create_rvc_backend` wywołane z pustą nazwą.
    return kolejka or [""]


def create_rvc_backend(
    settings: Settings, rvc: RVCSettings, *, device: str, backend: str | None = None
) -> RvcBackend:
    """Zbuduj backend zgodny z ustawieniami albo powiedz wprost, czego brakuje.

    Kolejność jest celowa: wybór użytkownika (``RVC_BACKEND``) wygrywa nad
    automatem, bo w środowisku z kilkoma instalacjami RVC automat trafiłby
    w tę, której akurat nikt nie chce.
    """
    model_path = rvc.resolved_model_path
    if model_path is None:
        raise RvcUnavailableError(t("rvc.no_model_path"), hint=t("rvc.no_model_path_hint"))

    missing = rvc.missing_files()
    if missing:
        raise RvcUnavailableError(
            t("rvc.missing_files", paths=", ".join(str(path) for path in missing)),
            hint=t("rvc.missing_files_hint"),
        )

    index_path = rvc.resolved_index_path

    def przez_pracownika() -> RvcBackend:
        return SubprocessRvcBackend(model_path, index_path, device, settings=settings)

    wbudowane: dict[str, Callable[[], RvcBackend]] = {
        BACKEND_APPLIO: lambda: ApplioBackend(model_path, index_path, device, settings=settings),
        BACKEND_SUBPROCESS: przez_pracownika,
        # `rvc_python` i `subprocess` to dziś jedno i to samo: pakiet da się
        # uruchomić WYŁĄCZNIE w osobnym środowisku. Nazwa zostaje, żeby nie
        # unieważnić RVC_BACKEND=rvc_python w cudzych plikach `.env`.
        BACKEND_RVC_PYTHON: przez_pracownika,
    }

    # `backend` podany wprost wygrywa z ustawieniem — tak dostawca mowy prosi
    # o KONKRETNE ogniwo łańcucha, zamiast dostawać wciąż to samo, pierwsze.
    wanted = (settings.rvc_backend if backend is None else backend).strip()
    if wanted:
        buduj = wbudowane.get(wanted)
        if buduj is not None:
            return buduj()
        return _load_custom_backend(wanted, model_path, index_path, device)

    # Automat: Applio, a gdy go nie ma — `rvc-python`. Kolejność wynika
    # z pomiaru, nie z upodobania: Applio jest 1,6x szybsze przy tej samej
    # metodzie wykrywania tonu (tabela w README).
    dostepne = available_rvc_backends(settings)
    for nazwa in (BACKEND_APPLIO, BACKEND_SUBPROCESS):
        if nazwa in dostepne:
            return wbudowane[nazwa]()

    raise RvcUnavailableError(t("rvc.no_backend"), hint=t("rvc.no_backend_hint"))


# --------------------------------------------------------------------------- #
# Konwerter: backend + limit czasu + statystyki
# --------------------------------------------------------------------------- #


class RvcConverter:
    """Backend opakowany w to, czego wymaga praca na żywo.

    Dwie rzeczy, których sam backend nie daje:

    * **Limit czasu.** Konwersja leci w osobnym wątku, a my czekamy na wynik
      najwyżej ``rvc_timeout_s``. Zawieszonego wątku nie da się w Pythonie
      zabić — więc go porzucamy razem z całym konwerterem i wracamy do Pipera.
      Asystent ma zamilknąć na jedno zdanie, nie na całą rozmowę.

      Wątek jest **daemonem**, i to nie jest szczegół. ``ThreadPoolExecutor``
      wygląda tu na oczywisty wybór, ale rejestruje hak ``atexit``, który przy
      wyjściu z programu CZEKA na swoje wątki — również po ``shutdown(wait=False)``.
      Zawieszony backend zatrzymywałby wtedy zamykanie asystenta, a pod systemd
      kończyłoby się to SIGKILL-em po ``TimeoutStopSec``. Wątek daemon ginie
      razem z procesem i o to właśnie chodzi.
    * **Pomiar.** Ile realnie zajmuje jedno przejście przez model. Bez tego
      „RVC jest wolne" jest opinią, a nie liczbą.
    """

    def __init__(
        self,
        backend: RvcBackend,
        *,
        settings: Settings,
        device: RvcDevice,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._device = device
        self._closed = False
        self.conversions = 0
        self.total_convert_s = 0.0
        self.total_audio_s = 0.0

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "rvc")

    @property
    def device(self) -> RvcDevice:
        return self._device

    @property
    def average_convert_s(self) -> float:
        return self.total_convert_s / self.conversions if self.conversions else 0.0

    @property
    def realtime_factor(self) -> float:
        """Ile sekund liczenia na sekundę dźwięku. Poniżej 1.0 = nadąża za mową."""
        return self.total_convert_s / self.total_audio_s if self.total_audio_s > 0 else 0.0

    def convert(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        pitch_shift: int,
        index_rate: float,
    ) -> tuple[np.ndarray, int]:
        if self._closed:
            raise RvcUnavailableError(t("rvc.converter_closed"))

        started = time.monotonic()
        wynik: list[tuple[np.ndarray, int]] = []
        blad: list[BaseException] = []

        def przebieg() -> None:
            try:
                wynik.append(
                    self._backend.convert(
                        samples,
                        sample_rate,
                        pitch_shift=pitch_shift,
                        index_rate=index_rate,
                    )
                )
            except BaseException as exc:  # wątek nie ma jak przekazać tego inaczej
                blad.append(exc)

        thread = threading.Thread(target=przebieg, name="rvc-convert", daemon=True)
        thread.start()
        thread.join(self._settings.rvc_timeout_s)

        if thread.is_alive():
            # Nie odzyskamy tego wątku. Zamykamy konwerter, żeby kolejne
            # fragmenty nie ustawiały się w kolejce za zablokowanym modelem.
            self._closed = True
            raise RvcUnavailableError(
                t("rvc.timeout", seconds=f"{self._settings.rvc_timeout_s:.0f}"),
                hint=t("rvc.timeout_hint"),
            )
        if blad:
            powod = blad[0]
            if isinstance(powod, RvcError):
                raise powod
            raise RvcUnavailableError(
                t("rvc.convert_failed", detail=str(powod))
            ) from powod
        if not wynik:  # pragma: no cover - wątek skończył bez wyniku i bez wyjątku
            raise RvcUnavailableError(t("rvc.no_output"))
        result = wynik[0]

        elapsed = time.monotonic() - started
        self.conversions += 1
        self.total_convert_s += elapsed
        if sample_rate > 0:
            self.total_audio_s += samples.size / sample_rate

        converted, out_rate = result
        logger.debug(
            "RVC: %.0f ms dźwięku w %.0f ms (%s, %s)",
            (samples.size / sample_rate * 1000) if sample_rate else 0.0,
            elapsed * 1000,
            self.backend_name,
            self._device.name,
        )
        return np.ascontiguousarray(converted, dtype=np.int16), int(out_rate)

    def close(self) -> None:
        if self._closed and self._backend is None:  # pragma: no cover
            return
        self._closed = True
        try:
            self._backend.close()
        except Exception:  # pragma: no cover
            logger.debug("Zamknięcie backendu RVC nie powiodło się")
