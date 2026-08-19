"""Słowo aktywujące (wake word) — w całości lokalnie, bez żadnej usługi w chmurze.

Fraza pochodzi z ``config/user_settings.json`` (pole ``wake_word``), a gdy jest
puste — z imienia asystenta (``hej <assistant_name>``). W kodzie nie ma żadnej
frazy zaszytej na sztywno.

Dlaczego domyślnym silnikiem jest Whisper, a nie openWakeWord
--------------------------------------------------------------
Wymaganie „dowolna fraza z pliku ustawień" rozstrzyga wybór samo:

* **openWakeWord** rozpoznaje wyłącznie frazy, dla których ma **wytrenowany
  model** (``hey jarvis``, ``alexa``, ...). Własna fraza wymaga wytrenowania
  modelu — godziny generowania mowy syntetycznej i ``torch``. Użytkownik, który
  zmieni imię asystenta na „Aiko", nie dostałby działającego „hej aiko".
  Za to gdy model dla frazy istnieje, jest bezkonkurencyjny: ok. 1-2% jednego
  rdzenia przy nasłuchu ciągłym (ONNX, ramki 80 ms), więc obsługujemy go jako
  silnik opcjonalny — wystarczy wskazać plik modelu w ustawieniach.
* **Detektor whisperowy** (domyślny) używa modelu ``tiny`` w int8 na krótkich
  buforach. Na słabym CPU to ok. 0,2-0,5 czasu rzeczywistego, czyli jakieś 0,3-0,7 s
  na półtorasekundowy fragment. Kluczowe jest to, że **nie działa w sposób
  ciągły**: uruchamia go dopiero VAD (czysty NumPy, koszt pomijalny), a
  fragmenty dłuższe niż ``WAKE_MAX_UTTERANCE_S`` są pomijane bez liczenia —
  nikt nie mówi „hej miku" przez dziesięć sekund. W ciszy zużycie CPU wynosi
  praktycznie zero, a model ``tiny`` to 39 MB, które i tak pobiera
  ``scripts/prepare_offline.py``.

Uczciwie o kompromisie: detektor whisperowy **transkrybuje** krótkie fragmenty
mowy z tła, żeby sprawdzić, czy padła fraza. Gwarancja, którą daje potok, jest
inna i mocniejsza: dopóki fraza nie padnie, **główny model Whispera i model
językowy nie dostają niczego** — rozmowa w tle kończy się na taniej atrapie
``tiny`` i jest odrzucana. Kto chce, żeby nawet to nie miało miejsca, instaluje
openWakeWord i wskazuje model swojej frazy.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable

import numpy as np

from audio.microphone import AudioFrame
from audio.resample import resample_int16
from audio.vad import Utterance
from config import (
    WAKEWORD_DIR,
    Settings,
    UserSettings,
    get_settings,
    get_user_settings,
)

logger = logging.getLogger(__name__)

WakeMode = Literal["stream", "utterance"]

# openWakeWord liczy na 16 kHz i porcjach 80 ms — to wymóg jego modeli.
OPENWAKEWORD_SAMPLE_RATE: Final[int] = 16_000
OPENWAKEWORD_CHUNK: Final[int] = 1_280

_WORD_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)

# Słowa, które Whisper dokleja na początku wypowiedzi i które nie zmieniają
# sensu frazy („o hej miku", „no hej miku").
_FILLER_WORDS: Final[frozenset[str]] = frozenset({"o", "no", "a", "e", "eee", "yy", "um", "uh"})


class WakeWordError(RuntimeError):
    """Nie udało się zbudować detektora frazy."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    @property
    def user_message(self) -> str:
        if self.hint:
            return f"{self.message}\n       Podpowiedź: {self.hint}"
        return self.message


@dataclass(frozen=True, slots=True)
class WakeMatch:
    """Wynik dopasowania frazy."""

    phrase: str
    score: float
    heard: str = ""
    # Reszta wypowiedzi po frazie („hej miku, jaka pogoda" -> „jaka pogoda").
    command: str = ""

    @property
    def has_command(self) -> bool:
        return bool(self.command.strip())


# --------------------------------------------------------------------------- #
# Dopasowanie tekstu do frazy
# --------------------------------------------------------------------------- #


def normalize_token(word: str) -> str:
    """Sprowadź słowo do postaci porównywalnej: bez ogonków, małymi literami.

    „Miku!" i „MIKU" to to samo słowo, a „hej" i „hej," różnią się tylko
    interpunkcją, której Whisper i tak dokłada według własnego uznania.
    """
    decomposed = unicodedata.normalize("NFKD", word.casefold())
    stripped = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and (character.isalnum() or character.isspace())
    )
    # ł/Ł nie rozkłada się przez NFKD — trzeba je zamienić osobno.
    return stripped.replace("ł", "l").replace("đ", "d").replace("ø", "o").strip()


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Podziel tekst na słowa: ``(znormalizowane, początek, koniec)`` w oryginale."""
    tokens: list[tuple[str, int, int]] = []
    for match in _WORD_PATTERN.finditer(text):
        normalized = normalize_token(match.group(0))
        if normalized:
            tokens.append((normalized, match.start(), match.end()))
    return tokens


def similarity(left: str, right: str) -> float:
    """Podobieństwo dwóch łańcuchów w skali 0..1."""
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


# Pomyłki, które Whisper popełnia regularnie na polszczyźnie i które nie zmieniają
# brzmienia słowa: „micu"/„miku", „mykq"/„miku", „hej"/„ej". Zwijanie stosujemy
# WYŁĄCZNIE jako dodatkowe porównanie — nigdy zamiast zwykłego, żeby nie zlepić
# ze sobą słów, które naprawdę brzmią różnie.
_FOLD_MAP: Final[dict[int, str]] = str.maketrans({"c": "k", "q": "k", "y": "i", "v": "w", "h": ""})


def fold_sounds(word: str) -> str:
    """Zwiń zapis do postaci „mniej więcej tak to brzmi"."""
    folded = word.translate(_FOLD_MAP)
    if not folded:
        return ""
    # Podwojone litery nie zmieniają brzmienia („mikku" = „miku").
    squeezed = [folded[0]]
    for character in folded[1:]:
        if character != squeezed[-1]:
            squeezed.append(character)
    return "".join(squeezed)


class PhraseMatcher:
    """Wyszukuje frazę w transkrypcji, tolerując przekręcenia i interpunkcję.

    Whisper zapisuje tę samą wypowiedź raz jako „Hej, Miku!", raz „Hey Miku",
    a na słabszym modelu bywa i „Ej micu". Dokładne porównanie łańcuchów
    odrzuciłoby wszystkie warianty poza jednym, dlatego porównujemy
    podobieństwo okna słów do frazy — z zapasem na jedno słowo w każdą stronę,
    bo detektor gubi albo dokłada krótkie wyrazy.
    """

    def __init__(
        self, phrase: str, *, threshold: float = 0.72, name_threshold: float = 0.80
    ) -> None:
        cleaned = phrase.strip()
        if not cleaned:
            raise WakeWordError(
                "Fraza wybudzająca jest pusta.",
                hint="ustaw pole wake_word w config/user_settings.json",
            )
        self._phrase = cleaned
        self._threshold = threshold
        self._tokens = [token for token, _, _ in tokenize(cleaned)]
        if not self._tokens:
            raise WakeWordError(
                f"Fraza wybudzająca {cleaned!r} nie zawiera żadnego słowa.",
                hint="użyj liter, np. „hej miku”",
            )
        self._target = " ".join(self._tokens)
        self._target_joined = "".join(self._tokens)
        # Ostatnie słowo frazy to nazwa asystenta — jedyny człon, który naprawdę
        # odróżnia zawołanie od rozmowy w tle. „hej" Whisper zapisuje jako „tej",
        # „kej", „ok", albo gubi całkiem.
        self._name = self._tokens[-1]
        self._name_threshold = name_threshold

    @property
    def phrase(self) -> str:
        return self._phrase

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def name_threshold(self) -> float:
        return self._name_threshold

    def _window_score(self, candidate: str) -> float:
        """Najlepsze z trzech porównań okna słów z frazą.

        Sklejona postać ratuje najczęstszy błąd detektora: „Hej Miku" wychodzi
        z niego jako JEDNO słowo „tymiku" albo „dajmiko", które przy porównaniu
        ze spacją przegrywa próg, choć brzmi prawidłowo.
        """
        joined = candidate.replace(" ", "")
        return max(
            similarity(candidate, self._target),
            similarity(joined, self._target_joined),
            similarity(fold_sounds(joined), fold_sounds(self._target_joined)),
        )

    def _name_score(self, token: str) -> float:
        """Jak bardzo pojedyncze słowo przypomina nazwę asystenta."""
        best = max(
            similarity(token, self._name),
            similarity(fold_sounds(token), fold_sounds(self._name)),
        )
        if len(token) > len(self._name):
            # Nazwa sklejona z poprzednim słowem („tymiku", „wtymiku").
            window = len(self._name) + 1
            best = max(
                best,
                *(
                    similarity(token[index : index + window], self._name)
                    for index in range(len(token) - len(self._name) + 1)
                ),
            )
        return best

    def match(self, text: str) -> WakeMatch | None:
        """Zwróć dopasowanie albo ``None``, gdy frazy nie ma w tekście."""
        if not text or not text.strip():
            return None

        tokens = tokenize(text)
        if not tokens:
            return None

        wanted = len(self._tokens)
        best_score = 0.0
        best_end: int | None = None

        for start in range(len(tokens)):
            for length in {wanted - 1, wanted, wanted + 1}:
                if length <= 0 or start + length > len(tokens):
                    continue
                window = tokens[start : start + length]
                candidate = " ".join(token for token, _, _ in window)
                score = self._window_score(candidate)
                if score > best_score:
                    best_score = score
                    best_end = window[-1][2]
            # Fraza wybudzająca stoi na początku wypowiedzi. Przeszukiwanie
            # całości sprzyjałoby fałszywym trafieniom w środku dłuższego
            # zdania z tła, więc dopuszczamy tylko kilka słów wypełniacza.
            if tokens[start][0] not in _FILLER_WORDS and start >= 2:
                break

        if best_score < self._threshold or best_end is None:
            # Druga droga: sama nazwa. Whisper gubi albo przekręca „hej" znacznie
            # częściej niż nazwę, a użytkownik i tak mówi „Miku, zrób…". Próg
            # jest wyższy niż dla całej frazy, bo jedno krótkie słowo łatwiej
            # pomylić — pomiar na korpusie: przy 0,80 zero fałszywych pobudek,
            # przy 0,75 zaczyna reagować „mikser", „mikrofon" i „Mika".
            for token, _, end in tokens[:4]:
                score = self._name_score(token)
                if score >= self._name_threshold:
                    best_score, best_end = score, end
                    break
            else:
                return None

        command = text[best_end:].lstrip(" ,.!?;:-—…\t\n")
        return WakeMatch(
            phrase=self._phrase,
            score=round(best_score, 3),
            heard=text.strip(),
            command=command.strip(),
        )

    def strip_phrase(self, text: str) -> str:
        """Usuń frazę z początku tekstu (jeśli tam jest) i oddaj resztę."""
        found = self.match(text)
        if found is None:
            return text
        return found.command or text


# --------------------------------------------------------------------------- #
# Silniki
# --------------------------------------------------------------------------- #


@runtime_checkable
class WakeWordEngine(Protocol):
    """Wspólny kontrakt detektorów frazy."""

    @property
    def name(self) -> str: ...

    @property
    def mode(self) -> WakeMode: ...

    @property
    def phrase(self) -> str: ...

    def reset(self) -> None: ...

    def process_frame(self, frame: AudioFrame) -> WakeMatch | None: ...

    def process_utterance(self, utterance: Utterance) -> WakeMatch | None: ...

    def strip_phrase(self, text: str) -> str: ...


class _BaseEngine:
    """Wspólna część: przechowanie matchera i domyślne „nic nie wykryto"."""

    _mode: WakeMode = "utterance"

    def __init__(self, matcher: PhraseMatcher) -> None:
        self._matcher = matcher

    @property
    def mode(self) -> WakeMode:
        return self._mode

    @property
    def phrase(self) -> str:
        return self._matcher.phrase

    def reset(self) -> None:
        """Silniki bezstanowe nie mają czego zerować."""

    def process_frame(self, frame: AudioFrame) -> WakeMatch | None:
        return None

    def process_utterance(self, utterance: Utterance) -> WakeMatch | None:
        return None

    def strip_phrase(self, text: str) -> str:
        return self._matcher.strip_phrase(text)


TranscribeFn = Callable[[Utterance], str]


class WhisperWakeWord(_BaseEngine):
    """Detektor frazy oparty o krótką transkrypcję modelem ``tiny``.

    Działa z **dowolną frazą w dowolnym języku**, bo nie potrzebuje modelu
    słowa kluczowego — rozpoznaje mowę i porównuje tekst. Uruchamiany wyłącznie
    na wypowiedziach wyciętych przez VAD i krótszych niż ``max_duration_s``.
    """

    _mode: WakeMode = "utterance"

    def __init__(
        self,
        matcher: PhraseMatcher,
        transcribe: TranscribeFn,
        *,
        max_duration_s: float = 4.0,
        describe: str = "whisper",
    ) -> None:
        super().__init__(matcher)
        self._transcribe = transcribe
        self._max_duration_s = max_duration_s
        self._describe = describe
        self.skipped_long = 0
        self.checked = 0

    @property
    def name(self) -> str:
        return self._describe

    def process_utterance(self, utterance: Utterance) -> WakeMatch | None:
        if utterance.duration_s > self._max_duration_s:
            # Dłuższa wypowiedź to rozmowa w tle, nie zawołanie. Pomijamy ją
            # bez transkrypcji — to główna oszczędność CPU tego silnika.
            self.skipped_long += 1
            logger.debug(
                "Pomijam fragment %.1f s — dłuższy niż limit frazy %.1f s",
                utterance.duration_s,
                self._max_duration_s,
            )
            return None

        self.checked += 1
        started = time.monotonic()
        try:
            text = self._transcribe(utterance)
        except Exception as exc:  # detektor nie może wysadzić nasłuchu
            logger.warning("Detektor frazy nie rozpoznał fragmentu: %s", exc)
            return None

        found = self._matcher.match(text)
        logger.debug(
            "Fraza %s w %.2f s: %r -> %s",
            "wykryta" if found else "niewykryta",
            time.monotonic() - started,
            text,
            f"{found.score:.2f}" if found else "-",
        )
        return found


class OpenWakeWordEngine(_BaseEngine):
    """Adapter na ``openwakeword`` — nasłuch ciągły, bez transkrypcji.

    Wymaga **pliku modelu wytrenowanego dla konkretnej frazy**. Ścieżkę podaje
    użytkownik (``wake_word_model`` w ``config/user_settings.json``) albo
    wystarczy wrzucić plik ``.onnx``/``.tflite`` do ``models/wakeword`` —
    katalog jest przeszukiwany automatycznie. Nic nie jest pobierane w locie:
    brak modelu to jasny błąd i zejście na detektor whisperowy.
    """

    _mode: WakeMode = "stream"

    def __init__(
        self,
        matcher: PhraseMatcher,
        model_paths: Sequence[Path],
        *,
        threshold: float = 0.5,
        sample_rate: int = OPENWAKEWORD_SAMPLE_RATE,
    ) -> None:
        super().__init__(matcher)
        if not model_paths:
            raise WakeWordError(
                "Nie wskazano modelu openWakeWord.",
                hint=(
                    "wpisz ścieżkę w wake_word_model (config/user_settings.json) "
                    f"albo wrzuć plik modelu do {WAKEWORD_DIR}"
                ),
            )
        try:
            from openwakeword.model import Model  # noqa: PLC0415 - ciężki import, celowo leniwy
        except ImportError as exc:
            raise WakeWordError(
                "Pakiet 'openwakeword' nie jest zainstalowany.",
                hint="pip install openwakeword albo zostaw WAKE_ENGINE=auto",
            ) from exc

        try:
            self._model = Model(wakeword_models=[str(path) for path in model_paths])
        except Exception as exc:  # brak onnxruntime, uszkodzony model, zła wersja
            raise WakeWordError(
                f"Nie udało się wczytać modelu openWakeWord ({exc}).",
                hint="sprawdź plik modelu i instalację onnxruntime",
            ) from exc

        self._threshold = threshold
        self._sample_rate = sample_rate
        self._model_paths = tuple(model_paths)
        self._buffer = np.empty(0, dtype=np.int16)

    @property
    def name(self) -> str:
        names = ", ".join(path.stem for path in self._model_paths)
        return f"openwakeword ({names})"

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.int16)
        reset = getattr(self._model, "reset", None)
        if callable(reset):
            reset()

    def process_frame(self, frame: AudioFrame) -> WakeMatch | None:
        samples = frame.samples
        if frame.sample_rate != OPENWAKEWORD_SAMPLE_RATE:
            samples = resample_int16(samples, frame.sample_rate, OPENWAKEWORD_SAMPLE_RATE)

        self._buffer = np.concatenate((self._buffer, samples))
        best = 0.0
        while self._buffer.size >= OPENWAKEWORD_CHUNK:
            chunk = self._buffer[:OPENWAKEWORD_CHUNK]
            self._buffer = self._buffer[OPENWAKEWORD_CHUNK:]
            try:
                scores = self._model.predict(chunk)
            except Exception as exc:  # pragma: no cover - awaria biblioteki natywnej
                logger.warning("openWakeWord zgłosił błąd: %s", exc)
                return None
            best = max(best, max((float(value) for value in scores.values()), default=0.0))

        if best < self._threshold:
            return None

        self.reset()
        return WakeMatch(phrase=self._matcher.phrase, score=round(best, 3), heard="")


# --------------------------------------------------------------------------- #
# Budowa detektora
# --------------------------------------------------------------------------- #


def find_wakeword_models(
    user_settings: UserSettings | None = None, directory: Path | None = None
) -> list[Path]:
    """Modele openWakeWord dostępne lokalnie: ze wskazanej ścieżki albo z katalogu."""
    active = user_settings if user_settings is not None else get_user_settings()
    explicit = active.wake_word_model.strip()
    found: list[Path] = []

    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_absolute():
            candidate = WAKEWORD_DIR / candidate
        if candidate.is_file():
            found.append(candidate)
        else:
            logger.warning("Wskazany model słowa aktywującego nie istnieje: %s", candidate)

    root = directory or WAKEWORD_DIR
    try:
        if root.is_dir():
            found.extend(
                sorted(
                    path
                    for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() in (".onnx", ".tflite")
                )
            )
    except OSError as exc:  # pragma: no cover - zależne od uprawnień
        logger.warning("Nie udało się przejrzeć katalogu modeli frazy %s: %s", root, exc)

    # Zachowaj kolejność, usuń duplikaty.
    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)
    return unique


def resolve_wake_phrase(user_settings: UserSettings | None = None) -> str:
    """Fraza wybudzająca z warstwy ustawień użytkownika."""
    active = user_settings if user_settings is not None else get_user_settings()
    return active.effective_wake_word


def create_wake_word_engine(
    settings: Settings | None = None,
    *,
    user_settings: UserSettings | None = None,
    transcribe: TranscribeFn | None = None,
    models_dir: Path | None = None,
) -> WakeWordEngine | None:
    """Zbuduj detektor zgodnie z ``WAKE_ENGINE``. ``None`` = wyłączony.

    ``auto`` próbuje openWakeWord (jeśli jest pakiet i model), a w razie
    niepowodzenia schodzi na detektor whisperowy — dokładnie tak samo jak
    ``VAD_ENGINE=auto`` schodzi z webrtcvad na detektor energetyczny.
    """
    active = settings or get_settings()
    user = user_settings if user_settings is not None else get_user_settings()

    if not active.wake_enabled or active.wake_engine == "none":
        logger.info("Słowo aktywujące wyłączone (WAKE_ENABLED/WAKE_ENGINE).")
        return None

    matcher = PhraseMatcher(
        resolve_wake_phrase(user),
        threshold=active.wake_similarity,
        name_threshold=active.wake_name_similarity,
    )
    engine = active.wake_engine

    if engine in ("auto", "openwakeword"):
        models = find_wakeword_models(user, models_dir)
        try:
            detector = OpenWakeWordEngine(
                matcher,
                models,
                threshold=active.wake_openwakeword_threshold,
            )
        except WakeWordError as exc:
            if engine == "openwakeword":
                raise
            logger.info(
                "openWakeWord niedostępny (%s) — używam detektora whisperowego.", exc.message
            )
        else:
            logger.info("Słowo aktywujące: %s, fraza %r", detector.name, matcher.phrase)
            return detector

    if transcribe is None:
        raise WakeWordError(
            "Detektor whisperowy wymaga funkcji transkrypcji.",
            hint="potok audio wstrzykuje ją sam — ten błąd oznacza błędne użycie API",
        )

    whisper_detector = WhisperWakeWord(
        matcher,
        transcribe,
        max_duration_s=active.wake_max_utterance_s,
    )
    logger.info(
        "Słowo aktywujące: detektor whisperowy, fraza %r, próg podobieństwa %.2f",
        matcher.phrase,
        active.wake_similarity,
    )
    return whisper_detector


__all__ = [
    "OPENWAKEWORD_CHUNK",
    "OPENWAKEWORD_SAMPLE_RATE",
    "OpenWakeWordEngine",
    "PhraseMatcher",
    "WakeMatch",
    "WakeMode",
    "WakeWordEngine",
    "WakeWordError",
    "WhisperWakeWord",
    "create_wake_word_engine",
    "find_wakeword_models",
    "normalize_token",
    "resolve_wake_phrase",
    "similarity",
    "tokenize",
]
