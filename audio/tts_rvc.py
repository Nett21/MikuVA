"""Głos Miku: dostawca mowy, który przepuszcza Pipera przez RVC (Faza 15).

To jest ta druga implementacja :class:`~audio.tts.TTSProvider`, o której mówi
docstring z Fazy 4 — nie nowy silnik syntezy, tylko **nakładka**. Tekst nadal
zamienia na dźwięk Piper; RVC zmienia wyłącznie barwę tego dźwięku.

Droga jednego zdania::

    tekst → Piper (fragmenty po ~20 ms)
          → bufor: sklej do ~0,5 s          ← audio/tts_rvc.py (ten plik)
          → RVC: zamiana barwy               ← audio/rvc.py
          → SpeechChunk
          → kolejka odtwarzania              ← audio/output.py
          → głośnik

Trzy decyzje, które tu zapadły, i powody:

**Buforujemy, zamiast konwertować każdą ramkę Pipera.** RVC musi widzieć
kawałek dźwięku, żeby wyliczyć wysokość — na 20 ms nie ma czego liczyć i wynik
brzmi jak bulgot. ``RVC_CHUNK_MIN_MS`` jest więc dokładnie tym pokrętłem, które
zamienia opóźnienie na jakość. Nie czekamy natomiast na całe zdanie: pierwszy
kawałek leci do głośnika, kiedy Piper mówi jeszcze dalszą część.

**Cała wypowiedź ma jedną częstotliwość próbkowania.** Model RVC liczy zwykle
w 40 000 Hz, Piper w 22 050 Hz. Gdyby fragmenty szły raz w jednej, raz w drugiej,
warstwa odtwarzania musiałaby przy każdej zmianie zamknąć i otworzyć strumień —
słychać to jako kliknięcie i gubi zawartość kolejki. Dlatego pierwszy fragment
ustala częstotliwość, a wszystko po nim jest do niej przeliczane.

**Awaria RVC nie jest awarią mowy.** Brak pliku modelu, brak backendu, wyjątek
w środku konwersji, przekroczenie limitu czasu — każde z tych zdarzeń kończy
się wpisem ``[ERROR]`` w logu i przejściem na czysty głos Pipera **w trakcie
tej samej wypowiedzi**. Asystent ma mówić nie swoim głosem, a nie milczeć.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from audio.rvc import (
    RvcConverter,
    RvcDevice,
    RvcError,
    RvcUnavailableError,
    create_rvc_backend,
    resample,
    resolve_rvc_device,
    rvc_backend_chain,
)
from audio.tts import PiperTTSProvider, SpeechChunk, TTSProvider
from config import Settings, UserSettings, get_settings, get_user_settings
from i18n import t

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


class RvcVoiceProvider(TTSProvider):
    """Piper opakowany konwersją barwy RVC.

    Dostawca bazowy jest wstrzykiwany (``base``) — domyślnie Piper, ale
    świadomie nie jest to zaszyte. Testy podstawiają tu atrapę, a gdyby kiedyś
    doszedł inny silnik syntezy, RVC nałoży się i na niego bez zmiany w tym pliku.
    """

    name = "rvc_miku"

    def __init__(
        self,
        settings: Settings | None = None,
        user_settings: UserSettings | None = None,
        *,
        base: TTSProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._user_override = user_settings
        self._base: TTSProvider = base or PiperTTSProvider(self._settings, user_settings)
        self._converter: RvcConverter | None = None
        # Czy w ogóle próbowaliśmy zbudować konwerter. Bez tego flaga
        # „nie ma konwertera" nie odróżnia „jeszcze nie próbowaliśmy" od
        # „próbowaliśmy i się nie udało" — a to druga próba przy każdym zdaniu.
        self._attempted = False
        # Backendy jeszcze niewypróbowane, w kolejności. Po awarii Applio
        # sięgamy po następny zamiast od razu wracać do samego Pipera.
        self._chain: list[str] = []
        self._degraded = False
        self._degraded_reason = ""
        self._output_rate = 0
        self._device: RvcDevice | None = None

    # --- ustawienia --------------------------------------------------------- #

    def _user(self) -> UserSettings:
        # Ustawienia użytkownika czytamy przy każdej wypowiedzi — tak samo jak
        # robi to Piper. Dzięki temu zmiana pitchu w pliku działa bez restartu.
        return self._user_override or get_user_settings()

    # --- cykl życia --------------------------------------------------------- #

    def load(self) -> None:
        """Załaduj Pipera i (jeśli się da) model RVC.

        Nie rzuca z powodu RVC. Brak modelu jest tu normalnym stanem świata,
        a nie błędem — kończy się komunikatem i zwykłym głosem.
        """
        self._base.load()
        self._ensure_converter()

    def unload(self) -> None:
        self._release_converter()
        self._attempted = False
        self._degraded = False
        self._degraded_reason = ""
        self._output_rate = 0
        self._base.unload()

    def close(self) -> None:
        self._release_converter()
        self._base.close()

    def __enter__(self) -> TTSProvider:
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def is_loaded(self) -> bool:
        return self._base.is_loaded

    def _release_converter(self) -> None:
        if self._converter is not None:
            self._converter.close()
            self._converter = None

    # --- opis --------------------------------------------------------------- #

    @property
    def sample_rate(self) -> int:
        return self._output_rate or self._base.sample_rate

    @property
    def is_speaking_enabled(self) -> bool:
        return self._base.is_speaking_enabled

    def supports_language(self, language: str | None) -> bool:
        # O język pyta się warstwa bazowa: RVC zmienia barwę, nie język.
        return self._base.supports_language(language)

    def voice_name(self) -> str:
        base_voice = self._base.voice_name()
        if self._converter is None:
            return base_voice
        model = self._user().rvc.resolved_model_path
        # Sama nazwa pliku, bez katalogu: pełna ścieżka trafia do /status i do
        # GUI, a zawiera katalog domowy użytkownika.
        label = model.stem if model is not None else self.name
        return f"{label} + {base_voice}" if base_voice else label

    def describe(self) -> str:
        if self._converter is None:
            reason = self._degraded_reason or t("rvc.status_off")
            return t("rvc.describe_fallback", base=self._base.describe(), reason=reason)
        return t(
            "rvc.describe_active",
            voice=self.voice_name(),
            backend=self._converter.backend_name,
            device=self._converter.device.describe(),
        )

    def is_active(self) -> bool:
        """Czy dźwięk naprawdę idzie przez RVC (a nie samym Piperem)."""
        return self._converter is not None and not self._degraded

    # --- budowa konwertera --------------------------------------------------- #

    def _ensure_converter(self) -> RvcConverter | None:
        """Zbuduj konwerter, biorąc kolejny backend z kolejki.

        Wołane na początku KAŻDEJ wypowiedzi, ale kosztuje tylko wtedy, gdy
        naprawdę jest co robić:

        * konwerter działa — zwracamy go od razu,
        * kolejka pusta — zwracamy ``None`` bez ani jednej próby, bo powody
          awarii RVC (brak pliku, brak pamięci GPU, niezgodne API) same
          z siebie nie znikają,
        * kolejka niepusta — sięgamy po następny backend. Zdarza się to co
          najwyżej tyle razy, ile jest backendów, a nie raz na zdanie.
        """
        if self._converter is not None and not self._degraded:
            return self._converter

        rvc = self._user().rvc
        if not self._attempted:
            self._attempted = True
            if not rvc.enabled:
                self._degraded_reason = t("rvc.disabled")
                logger.info("RVC: %s", self._degraded_reason)
                return None
            self._chain = rvc_backend_chain(self._settings)

        if not self._chain:
            return None

        device = resolve_rvc_device(self._settings)
        self._device = device
        backend = None
        while self._chain and backend is None:
            nazwa = self._chain.pop(0)
            try:
                backend = create_rvc_backend(self._settings, rvc, device=device.name, backend=nazwa)
            except RvcError as exc:
                # [ERROR], bo użytkownik POPROSIŁ o głos Miku i go nie dostanie.
                # Milczenie w tym miejscu jest tym, czego ta faza ma nie robić.
                self._degraded_reason = exc.message
                logger.error("RVC: %s", exc.user_message)
            except Exception as exc:  # kod backendu jest obcy
                self._degraded_reason = str(exc)
                logger.error("RVC: %s", t("rvc.convert_failed", detail=str(exc)))
        if backend is None:
            return None

        # Udało się — poprzednia awaria przestaje obowiązywać.
        self._degraded = False
        self._degraded_reason = ""
        self._converter = RvcConverter(backend, settings=self._settings, device=device)
        if device.is_cpu:
            # Ostrzeżenie, nie błąd: na CPU to działa, tylko wolniej. Liczba
            # w logu pojawi się po pierwszym zdaniu i wtedy widać, o ile.
            logger.warning("RVC: %s", t("rvc.on_cpu"))
        else:
            logger.info("RVC: %s", t("rvc.on_gpu", name=device.gpu.device_name or device.name))
        logger.info(
            "RVC: %s",
            t("rvc.ready", backend=self._converter.backend_name, device=device.name),
        )
        return self._converter

    def _degrade(self, reason: str) -> None:
        """Odetnij bieżący backend i powiedz w logu dlaczego.

        Do końca TEJ wypowiedzi mówi Piper — i to nie podlega dyskusji.
        Przesiadka na następny backend kosztuje sekundy na wczytanie modelu,
        a zrobiona w połowie zdania byłaby dokładnie tą ciszą, której ta faza
        ma nie dopuścić. Kolejny backend wchodzi więc dopiero przy NASTĘPNEJ
        wypowiedzi, na granicy zdania, i tylko raz na ogniwo kolejki.

        Gdy kolejka jest pusta, zostaje zwykły Piper do końca sesji.
        """
        if self._degraded:
            return
        self._degraded = True
        self._degraded_reason = reason
        if self._chain:
            logger.error(
                "RVC: %s", t("rvc.fallback_to_backend", reason=reason, backend=self._chain[0])
            )
        else:
            logger.error("RVC: %s", t("rvc.fallback_to_piper", reason=reason))
        self._release_converter()

    # --- synteza ------------------------------------------------------------- #

    def _block_limits(self, sample_rate: int) -> tuple[int, int]:
        """Ile próbek zbierać przed konwersją (minimum, maksimum)."""
        minimum = max(1, int(sample_rate * self._settings.rvc_chunk_min_ms / 1000))
        maximum = max(minimum, int(sample_rate * self._settings.rvc_chunk_max_ms / 1000))
        return minimum, maximum

    def _emit(self, samples: np.ndarray, sample_rate: int) -> SpeechChunk:
        """Wypuść fragment, sprowadzając go do jednej częstotliwości wypowiedzi."""
        if self._output_rate == 0:
            self._output_rate = int(sample_rate)
        if sample_rate != self._output_rate:
            samples = resample(samples, sample_rate, self._output_rate)
        return SpeechChunk(
            samples=np.ascontiguousarray(samples, dtype=np.int16),
            sample_rate=self._output_rate,
        )

    def _convert(self, block: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Jedno przejście przez model. Awaria = ten sam dźwięk bez zmian."""
        converter = self._converter
        if converter is None or self._degraded:
            return block, sample_rate

        rvc = self._user().rvc
        try:
            return converter.convert(
                block,
                sample_rate,
                pitch_shift=rvc.pitch_shift,
                index_rate=rvc.index_rate,
            )
        except RvcUnavailableError as exc:
            self._degrade(exc.message)
        except Exception as exc:  # backend jest kodem obcym
            self._degrade(str(exc))
        return block, sample_rate

    def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
        spoken = text.strip()
        if not spoken:
            return

        started = time.monotonic()
        converter = self._ensure_converter()
        first_base_audio: float | None = None
        emitted = 0

        pending: list[np.ndarray] = []
        pending_size = 0
        base_rate = 0

        def flush() -> Iterator[SpeechChunk]:
            """Oddaj to, co uzbierane — po kawałkach nie dłuższych niż maksimum."""
            nonlocal pending, pending_size
            if not pending:
                return
            block = np.concatenate(pending) if len(pending) > 1 else pending[0]
            pending = []
            pending_size = 0
            _, maximum = self._block_limits(base_rate)
            for start in range(0, block.size, maximum):
                piece = block[start : start + maximum]
                if piece.size == 0:
                    continue
                converted, rate = self._convert(piece, base_rate)
                yield self._emit(converted, rate)

        for chunk in self._base.synthesize(spoken, language=language):
            if chunk.is_empty:
                continue
            if first_base_audio is None:
                first_base_audio = time.monotonic() - started
            base_rate = chunk.sample_rate

            if converter is None or self._degraded:
                # Bez RVC nie ma po co buforować — im szybciej fragment trafi
                # do kolejki, tym wcześniej zaczyna grać.
                if pending:
                    yield from flush()
                emitted += 1
                yield self._emit(chunk.samples, base_rate)
                if emitted == 1:
                    self._log_latency(started, first_base_audio, converted=False)
                continue

            pending.append(np.ascontiguousarray(chunk.samples, dtype=np.int16))
            pending_size += chunk.samples.size
            minimum, _ = self._block_limits(base_rate)
            if pending_size >= minimum:
                for piece in flush():
                    emitted += 1
                    yield piece
                    if emitted == 1:
                        self._log_latency(started, first_base_audio, converted=True)

        for piece in flush():
            emitted += 1
            yield piece
            if emitted == 1:
                self._log_latency(started, first_base_audio, converted=self.is_active())

    def _log_latency(
        self, started: float, first_base_audio: float | None, *, converted: bool
    ) -> None:
        """Zapisz, ile minęło od tekstu do pierwszego gotowego fragmentu.

        To jest ta liczba, o którą chodzi w „opóźnienie rzędu sekundy": widać
        z niej OSOBNO czas Pipera i czas RVC, więc gdy robi się wolno, wiadomo,
        które ogniwo za to odpowiada. Bez rozbicia zostaje zgadywanie.
        """
        total_ms = (time.monotonic() - started) * 1000
        piper_ms = (first_base_audio or 0.0) * 1000
        target_ms = self._settings.rvc_latency_target_ms
        message = t(
            "rvc.latency",
            total=f"{total_ms:.0f}",
            piper=f"{piper_ms:.0f}",
            rvc=f"{max(0.0, total_ms - piper_ms):.0f}",
            engine=self.name if converted else "piper",
        )
        if total_ms > target_ms:
            logger.warning("RVC: %s", t("rvc.latency_over", detail=message, target=target_ms))
        else:
            logger.info("RVC: %s", message)

    def cancel(self) -> None:
        self._base.cancel()


def create_rvc_voice_provider(settings: Settings, user_settings: UserSettings) -> TTSProvider:
    """Fabryka dla rejestru dostawców z ``audio/tts.py``."""
    return RvcVoiceProvider(settings, user_settings)
