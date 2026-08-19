"""Testy syntezy mowy (Faza 4).

Żaden test nie uruchamia prawdziwego Pipera ani nie dotyka karty dźwiękowej:
pakiet ``piper`` jest podmieniany w ``sys.modules``, binarka — atrapą klasy
``subprocess.Popen``, a głośnik — atrapą ``sounddevice.OutputStream``. Testy
mają sprawdzać kod, nie to, co akurat jest zainstalowane na maszynie.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from conftest import FakePiper, FakePiperVoice, FakeSoundDevice, make_tone, make_voice_files

from audio.tts import (
    CollectingSink,
    NullTTSProvider,
    PiperProcessBackend,
    PiperPythonBackend,
    PiperTTSProvider,
    SentenceBuffer,
    SpeechChunk,
    SpeechOutput,
    TTSError,
    TTSProvider,
    TTSUnavailableError,
    available_tts_engines,
    clean_for_speech,
    create_piper_backend,
    create_tts_provider,
    describe_voice_file,
    find_piper_voice,
    iter_piper_voices,
    register_tts_provider,
    select_piper_voice,
    split_sentences,
    write_wav,
)
from config import Settings, UserSettings
from i18n import t


@pytest.fixture
def voices_dir(settings: Settings) -> Path:
    """Katalog wskazany w ustawieniach jako źródło głosów."""
    directory = Path(settings.piper_voices_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_provider(
    settings: Settings,
    voices_dir: Path,
    *,
    user: UserSettings | None = None,
    backend: Any = None,
) -> PiperTTSProvider:
    if not any(voices_dir.glob("*.onnx")):
        make_voice_files(voices_dir)
    return PiperTTSProvider(
        settings, user if user is not None else UserSettings(), backend=backend
    )


# --------------------------------------------------------------------------- #
# Kontrakt interfejsu
# --------------------------------------------------------------------------- #


def test_interfejs_wymusza_tylko_synthesize() -> None:
    """Nowy silnik ma być tani do napisania: jedna metoda i tyle."""

    class MinimalProvider(TTSProvider):
        name = "minimal"

        def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
            yield SpeechChunk(samples=make_tone(160), sample_rate=16_000)

    provider = MinimalProvider()
    provider.load()  # domyślna implementacja nie może wymagać nadpisania

    chunks = list(provider.synthesize("cokolwiek"))

    assert len(chunks) == 1
    assert provider.describe() == "minimal"
    assert provider.supports_language("pl") is True


def test_nie_da_sie_utworzyc_dostawcy_bez_synthesize() -> None:
    """Klasa bez metody syntezy nie ma prawa powstać — to cały sens ABC."""

    class Broken(TTSProvider):
        name = "broken"

    with pytest.raises(TypeError):
        Broken()  # type: ignore[abstract]


def test_synthesize_all_skleja_fragmenty() -> None:
    class TwoChunks(TTSProvider):
        name = "dwa"

        def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
            yield SpeechChunk(samples=make_tone(100), sample_rate=22_050)
            yield SpeechChunk(samples=make_tone(50), sample_rate=22_050)

    whole = TwoChunks().synthesize_all("tekst")

    assert whole is not None
    assert whole.samples.size == 150
    assert whole.sample_rate == 22_050
    assert whole.duration_s == pytest.approx(150 / 22_050)


def test_rejestr_pozwala_dopisac_wlasny_silnik(settings: Settings) -> None:
    """Faza 15 (RVC/XTTS) ma dodać silnik rejestracją, nie zmianą kodu wyboru."""

    class Fake(NullTTSProvider):
        name = "atrapa"

    register_tts_provider("atrapa", lambda active, user: Fake())
    try:
        assert "atrapa" in available_tts_engines()
        provider = create_tts_provider(
            settings, UserSettings(voice_engine="atrapa")
        )
        assert isinstance(provider, Fake)
    finally:
        from audio.tts import _PROVIDERS

        _PROVIDERS.pop("atrapa", None)


def test_nieznany_silnik_nie_wywraca_programu(settings: Settings) -> None:
    provider = create_tts_provider(settings, UserSettings(voice_engine="cos-czego-nie-ma"))

    assert provider.is_speaking_enabled is False
    assert list(provider.synthesize("cisza")) == []


def test_wylaczenie_w_env_ma_pierwszenstwo(settings: Settings) -> None:
    """TTS_ENABLED=false to wyłącznik infrastrukturalny — ustawienia użytkownika nie wygrywają."""
    provider = create_tts_provider(
        settings.model_copy(update={"tts_enabled": False}),
        UserSettings(voice_engine="piper"),
    )

    assert provider.is_speaking_enabled is False


def test_voice_engine_none_wylacza_mowe(settings: Settings) -> None:
    provider = create_tts_provider(settings, UserSettings(voice_engine="none"))

    assert isinstance(provider, NullTTSProvider)


# --------------------------------------------------------------------------- #
# Wyszukiwanie i wybór głosu
# --------------------------------------------------------------------------- #


def test_opis_glosu_czyta_dane_z_pliku_json(voices_dir: Path) -> None:
    path = make_voice_files(voices_dir, "pl_PL-gosia-medium", sample_rate=16_000, speakers=3)

    voice = describe_voice_file(path)

    assert voice.name == "pl_PL-gosia-medium"
    assert voice.language == "pl"
    assert voice.sample_rate == 16_000
    assert voice.quality == "medium"
    assert voice.is_multispeaker is True


def test_glos_bez_json_a_nadal_da_sie_uzyc(voices_dir: Path) -> None:
    """Brak opisu to nie powód do milczenia — język bierzemy z nazwy pliku."""
    path = make_voice_files(voices_dir, "en_US-amy-low", with_config=False)

    voice = describe_voice_file(path)

    assert voice.language == "en"
    assert voice.config_path is None
    assert voice.sample_rate == 22_050  # udokumentowana wartość domyślna Pipera


def test_glosy_sa_znajdowane_takze_w_podkatalogach(settings: Settings, voices_dir: Path) -> None:
    """Oficjalne paczki rozpakowują się do <język>/<nazwa>/ — to musi działać."""
    make_voice_files(voices_dir / "pl" / "pl_PL" / "darkman", "pl_PL-darkman-medium")

    found = iter_piper_voices(settings)

    assert [voice.name for voice in found] == ["pl_PL-darkman-medium"]


def test_wyszukiwanie_glosu_ignoruje_wielkosc_liter(settings: Settings, voices_dir: Path) -> None:
    make_voice_files(voices_dir, "pl_PL-Darkman-medium")

    assert find_piper_voice("pl_pl-darkman-medium", settings) is not None
    assert find_piper_voice("nie-ma-takiego", settings) is None


def test_glos_mozna_wskazac_sciezka_bezwzgledna(settings: Settings, tmp_path: Path) -> None:
    """Model spoza katalogów Pipera też ma działać — bez zmiany kodu."""
    path = make_voice_files(tmp_path / "gdzies" / "indziej", "wlasny-glos")

    voice = find_piper_voice(str(path), settings)

    assert voice is not None
    assert voice.path == path


def test_piper_model_z_ustawien_wygrywa(settings: Settings, voices_dir: Path) -> None:
    make_voice_files(voices_dir, "pl_PL-pierwszy-medium")
    make_voice_files(voices_dir, "pl_PL-drugi-medium")

    chosen = select_piper_voice(
        settings, UserSettings(piper_model="pl_PL-drugi-medium"), language="pl"
    )

    assert chosen is not None
    assert chosen.name == "pl_PL-drugi-medium"


def test_bez_piper_model_glos_dobiera_sie_do_jezyka(settings: Settings, voices_dir: Path) -> None:
    make_voice_files(voices_dir, "pl_PL-polski-medium")
    make_voice_files(voices_dir, "en_US-angielski-medium")

    polish = select_piper_voice(settings, UserSettings(), language="pl")
    english = select_piper_voice(settings, UserSettings(), language="en")

    assert polish is not None and polish.language == "pl"
    assert english is not None and english.language == "en"


def test_mapa_piper_voices_wybiera_glos_per_jezyk(settings: Settings, voices_dir: Path) -> None:
    make_voice_files(voices_dir, "pl_PL-polski-medium")
    make_voice_files(voices_dir, "en_US-angielski-medium")
    user = UserSettings(
        piper_model="pl_PL-polski-medium",
        piper_voices={"en": "en_US-angielski-medium"},
    )

    assert select_piper_voice(settings, user, language="pl").name == "pl_PL-polski-medium"
    assert select_piper_voice(settings, user, language="en").name == "en_US-angielski-medium"


def test_brak_glosu_dla_jezyka_nie_odbiera_mowy(settings: Settings, voices_dir: Path) -> None:
    """Lepiej powiedzieć z obcym akcentem niż zamilknąć."""
    make_voice_files(voices_dir, "pl_PL-polski-medium")

    chosen = select_piper_voice(settings, UserSettings(), language="de")

    assert chosen is not None
    assert chosen.name == "pl_PL-polski-medium"


def test_bledna_nazwa_glosu_konczy_sie_wyborem_zastepczym(
    settings: Settings, voices_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Literówka w user_settings.json ma dać ostrzeżenie, a nie wyjątek."""
    make_voice_files(voices_dir, "pl_PL-polski-medium")

    with caplog.at_level("WARNING"):
        chosen = select_piper_voice(settings, UserSettings(piper_model="literowka"), language="pl")

    assert chosen is not None
    assert "literowka" in caplog.text


def test_brak_jakiegokolwiek_glosu_daje_czytelny_blad(settings: Settings) -> None:
    provider = PiperTTSProvider(settings, UserSettings())

    with pytest.raises(TTSUnavailableError) as info:
        provider.load()

    assert "voice" in info.value.message
    assert "prepare_offline" in info.value.user_message


# --------------------------------------------------------------------------- #
# Piper: liczenie w pakiecie Pythona
# --------------------------------------------------------------------------- #


def test_synteza_pakietem_zwraca_probki(
    settings: Settings, voices_dir: Path, fake_piper: FakePiper
) -> None:
    provider = make_provider(settings, voices_dir)
    provider.load()

    chunks = list(provider.synthesize("Dzień dobry."))

    assert chunks, "silnik nie zwrócił żadnego dźwięku"
    assert all(chunk.samples.dtype == np.int16 for chunk in chunks)
    assert all(chunk.sample_rate == 22_050 for chunk in chunks)
    assert FakePiperVoice.instances[0].calls[0]["text"] == "Dzień dobry."


def test_nowsze_api_pakietu_tez_dziala(
    settings: Settings, voices_dir: Path, fake_modern_piper: FakePiper
) -> None:
    """Aktualizacja piper-tts nie może uciszyć asystenta."""
    provider = make_provider(settings, voices_dir)

    chunks = list(provider.synthesize("Test nowego API."))

    assert chunks
    assert {chunk.sample_rate for chunk in chunks} == {24_000}


def test_tempo_mowy_przechodzi_do_pipera(
    settings: Settings, voices_dir: Path, fake_piper: FakePiper
) -> None:
    """voice_speed z user_settings.json steruje length_scale (odwrotność tempa)."""
    provider = make_provider(settings, voices_dir, user=UserSettings(voice_speed=2.0))

    list(provider.synthesize("Szybko."))

    assert FakePiperVoice.instances[0].calls[0]["length_scale"] == pytest.approx(0.5)


def test_tempo_i_mowca_ida_przez_synthesis_config(
    settings: Settings, voices_dir: Path, fake_modern_piper: FakePiper
) -> None:
    """W nowszym API parametry jadą w obiekcie — pominięcie go uciszyłoby voice_speed.

    To nie jest teoretyczny przypadek: tak wygląda pakiet, który dziś instaluje
    ``pip install piper-tts``. Bez tej ścieżki tempo mowy byłoby po cichu
    ignorowane, a wszystko wyglądałoby na działające.
    """
    make_voice_files(voices_dir, "pl_PL-wielu-medium", speakers=5)
    provider = PiperTTSProvider(
        settings, UserSettings(piper_model="pl_PL-wielu-medium", voice_speed=0.5, piper_speaker=3)
    )

    list(provider.synthesize("Wolno i innym głosem."))

    call = FakePiperVoice.instances[0].calls[0]
    assert call["length_scale"] == pytest.approx(2.0)
    assert call["speaker_id"] == 3


def test_numer_mowcy_tylko_dla_modeli_wielogosowych(
    settings: Settings, voices_dir: Path, fake_piper: FakePiper
) -> None:
    make_voice_files(voices_dir, "pl_PL-jeden-medium", speakers=1)
    provider = PiperTTSProvider(settings, UserSettings(piper_speaker=3))

    list(provider.synthesize("Test."))

    assert FakePiperVoice.instances[0].calls[0]["speaker_id"] is None


def test_zmiana_jezyka_przelacza_glos(
    settings: Settings, voices_dir: Path, fake_piper: FakePiper
) -> None:
    make_voice_files(voices_dir, "pl_PL-polski-medium")
    make_voice_files(voices_dir, "en_US-angielski-medium")
    provider = PiperTTSProvider(settings, UserSettings())

    list(provider.synthesize("Cześć.", language="pl"))
    polish = provider.voice_name()
    list(provider.synthesize("Hello.", language="en"))
    english = provider.voice_name()

    assert polish == "pl_PL-polski-medium"
    assert english == "en_US-angielski-medium"


def test_blad_pakietu_zamienia_sie_w_tts_error(
    settings: Settings, voices_dir: Path, fake_piper: FakePiper
) -> None:
    provider = make_provider(settings, voices_dir)
    provider.load()
    FakePiperVoice.instances.clear()
    list(provider.synthesize("rozgrzewka"))
    FakePiperVoice.instances[0].error = RuntimeError("atrapa padła")

    with pytest.raises(TTSError) as info:
        list(provider.synthesize("kolejne zdanie"))

    assert "atrapa padła" in info.value.message


def test_brak_pipera_daje_podpowiedz_instalacji(settings: Settings, no_piper: None) -> None:
    with pytest.raises(TTSUnavailableError) as info:
        create_piper_backend(settings)

    assert "piper" in info.value.message.lower()
    assert "pip install piper-tts" in info.value.user_message


def test_pakiet_ma_pierwszenstwo_przed_binarka(
    settings: Settings, fake_piper: FakePiper, monkeypatch: pytest.MonkeyPatch
) -> None:
    import audio.tts

    monkeypatch.setattr(audio.tts, "find_piper_binary", lambda *args, **kwargs: Path("/nieistotne"))

    assert isinstance(create_piper_backend(settings), PiperPythonBackend)


# --------------------------------------------------------------------------- #
# Piper: liczenie w osobnym procesie (binarka)
# --------------------------------------------------------------------------- #


class FakePopen:
    """Atrapa procesu Pipera: przyjmuje tekst na stdin, oddaje PCM na stdout."""

    instances: ClassVar[list[FakePopen]] = []

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.killed = False
        self._returncode: int | None = 0
        self.written = b""
        FakePopen.instances.append(self)

        payload = make_tone(2_000).tobytes()
        self.stdin = _FakeWritable(self)
        self.stdout = _FakeReadable(payload)
        self.stderr = _FakeReadable(b"")

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode if self._returncode is not None else 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9


class _FakeWritable:
    def __init__(self, owner: FakePopen) -> None:
        self._owner = owner

    def write(self, data: bytes) -> int:
        self._owner.written += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeReadable:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._position = 0

    def read1(self, size: int) -> bytes:
        return self.read(size)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._position
        data = self._payload[self._position : self._position + size]
        self._position += len(data)
        return data

    def __iter__(self) -> Iterator[bytes]:
        return iter(())

    def close(self) -> None:
        return None


@pytest.fixture
def fake_process(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakePopen]]:
    FakePopen.instances.clear()
    import audio.tts

    monkeypatch.setattr(audio.tts.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        audio.tts,
        "_detect_cli_flags",
        lambda binary: audio.tts.PiperCliFlags(
            output_raw="--output-raw", length_scale="--length-scale", speaker="--speaker"
        ),
    )
    yield FakePopen
    FakePopen.instances.clear()


def test_binarka_dostaje_model_i_tekst(
    settings: Settings, voices_dir: Path, fake_process: type[FakePopen]
) -> None:
    path = make_voice_files(voices_dir, "pl_PL-testowy-medium")
    backend = PiperProcessBackend(Path("piper"))
    voice = describe_voice_file(path)

    chunks = list(backend.stream("Powiedz coś.", voice))

    process = FakePopen.instances[0]
    assert str(path) in process.command
    assert "--output-raw" in process.command
    assert process.written == "Powiedz coś.".encode()
    assert sum(chunk.samples.size for chunk in chunks) == 2_000
    assert chunks[0].sample_rate == 22_050


def test_binarka_dostaje_tempo_i_mowce(
    settings: Settings, voices_dir: Path, fake_process: type[FakePopen]
) -> None:
    path = make_voice_files(voices_dir, "pl_PL-testowy-medium", speakers=4)
    backend = PiperProcessBackend(Path("piper"))

    list(backend.stream("Tekst.", describe_voice_file(path), speaker=2, length_scale=0.5))

    command = FakePopen.instances[0].command
    assert command[command.index("--length-scale") + 1] == "0.500"
    assert command[command.index("--speaker") + 1] == "2"


def test_nazwy_opcji_sa_odczytywane_z_pomocy_programu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starsze wydania Pipera używają podkreśleń — wykrywamy to, zamiast zgadywać."""
    import audio.tts

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            command, 0, stdout="usage: piper --model M --output_raw --length_scale L", stderr=""
        )

    monkeypatch.setattr(audio.tts.subprocess, "run", fake_run)

    flags = audio.tts._detect_cli_flags(Path("piper"))

    assert flags.output_raw == "--output_raw"
    assert flags.length_scale == "--length_scale"


def test_niezerowy_kod_wyjscia_konczy_sie_bledem(
    settings: Settings, voices_dir: Path, fake_process: type[FakePopen]
) -> None:
    path = make_voice_files(voices_dir, "pl_PL-testowy-medium")
    backend = PiperProcessBackend(Path("piper"))

    def failing(command: list[str], **kwargs: Any) -> FakePopen:
        process = FakePopen(command, **kwargs)
        process._returncode = 2
        process.stderr = _FakeReadable(b"")
        return process

    import audio.tts

    audio.tts.subprocess.Popen = failing  # type: ignore[assignment]
    try:
        with pytest.raises(TTSError) as info:
            list(backend.stream("Tekst.", describe_voice_file(path)))
    finally:
        audio.tts.subprocess.Popen = FakePopen  # type: ignore[assignment]

    assert "code 2" in info.value.message


def test_brak_programu_daje_czytelny_komunikat(
    settings: Settings, voices_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_voice_files(voices_dir, "pl_PL-testowy-medium")
    import audio.tts

    def raising(*args: Any, **kwargs: Any) -> None:
        raise OSError("nie ma takiego pliku")

    monkeypatch.setattr(audio.tts.subprocess, "Popen", raising)
    monkeypatch.setattr(audio.tts, "_detect_cli_flags", lambda binary: audio.tts.PiperCliFlags())
    backend = PiperProcessBackend(Path("/nie/ma/piper"))

    with pytest.raises(TTSUnavailableError) as info:
        list(backend.stream("Tekst.", describe_voice_file(path)))

    assert "PIPER_BINARY" in info.value.user_message


# --------------------------------------------------------------------------- #
# Dzielenie tekstu na zdania
# --------------------------------------------------------------------------- #


def test_zdania_sa_oddawane_w_miare_pisania() -> None:
    buffer = SentenceBuffer(min_chars=0)

    assert buffer.push("Cześć") == []
    assert buffer.push("! Co ") == ["Cześć!"]
    assert buffer.push("słychać? ") == ["Co słychać?"]


def test_skroty_i_liczby_nie_koncza_zdania() -> None:
    pieces = split_sentences("Mam np. 3.5 litra wody itd. i tyle. Koniec.", min_chars=0)

    assert pieces == ["Mam np. 3.5 litra wody itd. i tyle.", "Koniec."]


def test_krotkie_fragmenty_czekaja_na_dalszy_ciag() -> None:
    """Synteza dwuwyrazowych urywków brzmi rwanie — zbieramy je w całość."""
    buffer = SentenceBuffer(min_chars=30)

    assert buffer.push("Tak. ") == []
    ready = buffer.push("A teraz coś znacznie dłuższego do powiedzenia. ")

    assert ready == ["Tak. A teraz coś znacznie dłuższego do powiedzenia."]


def test_bardzo_dlugi_ciag_bez_kropki_jest_dzielony() -> None:
    """Model potrafi „lecieć" bez interpunkcji — odtwarzanie musi kiedyś ruszyć."""
    text = "słowo " * 100

    pieces = split_sentences(text, min_chars=0, max_chars=60)

    assert len(pieces) > 1
    assert all(len(piece) <= 70 for piece in pieces)


def test_flush_oddaje_resztke() -> None:
    buffer = SentenceBuffer(min_chars=0)
    buffer.push("Zdanie bez kropki na końcu")

    assert buffer.flush() == ["Zdanie bez kropki na końcu"]
    assert buffer.flush() == []


def test_nowa_linia_konczy_zdanie() -> None:
    buffer = SentenceBuffer(min_chars=0)

    assert buffer.push("Pierwsza linia\nDruga") == ["Pierwsza linia"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**Ważne** i _pochyłe_", "Ważne i pochyłe"),
        ("### Nagłówek", "Nagłówek"),
        ("- punkt pierwszy", "punkt pierwszy"),
        ("Zajrzyj na https://example.com/bardzo/dluga", "Zajrzyj na link"),
        ("[dokumentacja](https://example.com)", "dokumentacja"),
        ("Super 🎉 dzięki", "Super dzięki"),
        ("Kod: `print(1)`", "Kod: print(1)"),
    ],
)
def test_tekst_jest_czyszczony_przed_wypowiedzeniem(source: str, expected: str) -> None:
    assert clean_for_speech(source) == expected


def test_bloki_kodu_nie_sa_czytane_na_glos() -> None:
    text = "Oto rozwiązanie:\n```python\nprint('x')\n```\nGotowe."

    spoken = clean_for_speech(text)

    assert "print" not in spoken
    assert spoken == "Oto rozwiązanie: Gotowe."


def test_sama_odpowiedz_kodem_nie_konczy_sie_cisza() -> None:
    """Gdyby cała odpowiedź była kodem, lepiej przeczytać ją niż milczeć."""
    spoken = clean_for_speech("```\nprint('x')\n```")

    assert "print" in spoken


# --------------------------------------------------------------------------- #
# Mówienie w trakcie generowania odpowiedzi
# --------------------------------------------------------------------------- #


class ScriptedProvider(TTSProvider):
    """Silnik, który zapisuje, co i kiedy dostał do powiedzenia."""

    name = "scripted"

    def __init__(self, *, sample_rate: int = 22_050, error: Exception | None = None) -> None:
        self.spoken: list[str] = []
        self.languages: list[str | None] = []
        self.sample_rate_value = sample_rate
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.cancelled = False

    @property
    def sample_rate(self) -> int:
        return self.sample_rate_value

    def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
        self.spoken.append(text)
        self.languages.append(language)
        self.started.set()
        if self.error is not None:
            raise self.error
        self.release.wait(timeout=5.0)
        yield SpeechChunk(samples=make_tone(320), sample_rate=self.sample_rate_value)

    def cancel(self) -> None:
        self.cancelled = True
        self.release.set()


def make_speech(
    settings: Settings, provider: TTSProvider, **updates: Any
) -> tuple[SpeechOutput, CollectingSink]:
    sink = CollectingSink()
    active = settings.model_copy(update={"tts_min_sentence_chars": 0, **updates})
    return SpeechOutput(provider, sink, settings=active), sink


def test_pierwsze_zdanie_idzie_do_glosnika_przed_koncem_odpowiedzi(
    settings: Settings,
) -> None:
    """Sedno Fazy 4: odtwarzanie rusza, gdy model wciąż pisze dalszy ciąg."""
    provider = ScriptedProvider()
    speech, sink = make_speech(settings, provider)

    speech.begin("pl")
    speech.feed("Pierwsze zdanie już gotowe. ")
    assert provider.started.wait(timeout=5.0), "synteza nie ruszyła przed końcem generowania"

    # Model dopiero teraz dopisuje resztę — a pierwsze zdanie już zagrało.
    for _ in range(100):
        if sink.chunks:
            break
        time.sleep(0.01)
    assert sink.chunks, "pierwszy fragment nie trafił do odtwarzania"

    speech.feed("Drugie zdanie.")
    speech.end()

    assert provider.spoken == ["Pierwsze zdanie już gotowe.", "Drugie zdanie."]
    assert provider.languages == ["pl", "pl"]
    assert sink.opened_rates == [22_050]


def test_bez_strumieniowania_synteza_czeka_na_calosc(settings: Settings) -> None:
    provider = ScriptedProvider()
    speech, _ = make_speech(settings, provider, tts_stream_sentences=False)

    speech.begin()
    speech.feed("Pierwsze zdanie. ")
    speech.feed("Drugie zdanie.")
    assert provider.spoken == []

    speech.end()

    assert provider.spoken == ["Pierwsze zdanie. Drugie zdanie."]


def test_anulowanie_przerywa_mowe(settings: Settings) -> None:
    provider = ScriptedProvider()
    provider.release.clear()  # synteza „trwa"
    speech, sink = make_speech(settings, provider)

    speech.begin()
    speech.feed("Zdanie, które zostanie przerwane. ")
    assert provider.started.wait(timeout=5.0)
    speech.cancel()

    assert provider.cancelled is True
    assert sink.cancelled is True
    assert speech.is_speaking is False


def test_blad_syntezy_wylacza_mowe_ale_nie_program(settings: Settings) -> None:
    errors: list[TTSError] = []
    provider = ScriptedProvider(error=TTSError("atrapa nie umie mówić"))
    sink = CollectingSink()
    speech = SpeechOutput(
        provider,
        sink,
        settings=settings.model_copy(update={"tts_min_sentence_chars": 0}),
        on_error=errors.append,
    )

    speech.begin()
    speech.feed("Cokolwiek. ")
    speech.feed("Drugie zdanie, którego już nie powie. ")
    speech.end()

    assert len(errors) == 1
    assert speech.failed is True
    assert speech.enabled is False
    assert sink.chunks == []


def test_zmiana_czestotliwosci_otwiera_wyjscie_ponownie(settings: Settings) -> None:
    """Przełączenie na głos o innym sample rate nie może zniekształcić dźwięku."""

    class TwoRates(TTSProvider):
        name = "dwie-czestotliwosci"

        def __init__(self) -> None:
            self.calls = 0

        def synthesize(self, text: str, *, language: str | None = None) -> Iterator[SpeechChunk]:
            self.calls += 1
            rate = 22_050 if self.calls == 1 else 16_000
            yield SpeechChunk(samples=make_tone(160), sample_rate=rate)

    speech, sink = make_speech(settings, TwoRates())
    speech.begin()
    speech.feed("Pierwsze. ")
    speech.feed("Drugie. ")
    speech.end()

    assert sink.opened_rates == [22_050, 16_000]


def test_mowa_wylaczona_niczego_nie_syntezuje(settings: Settings) -> None:
    speech, sink = make_speech(settings, NullTTSProvider())

    speech.begin()
    speech.feed("Cisza. ")
    speech.end()

    assert sink.chunks == []
    assert speech.enabled is False


# --------------------------------------------------------------------------- #
# Zapis do pliku (maszyny bez głośnika)
# --------------------------------------------------------------------------- #


def test_zapis_do_wav_dziala_bez_karty_dzwiekowej(tmp_path: Path) -> None:
    chunks = [
        SpeechChunk(samples=make_tone(1_000), sample_rate=22_050),
        SpeechChunk(samples=make_tone(500), sample_rate=22_050),
    ]

    path = write_wav(tmp_path / "podkatalog" / "proba.wav", chunks)

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 22_050
        assert handle.getnframes() == 1_500


def test_zapis_pustej_syntezy_zglasza_blad(tmp_path: Path) -> None:
    with pytest.raises(TTSError):
        write_wav(tmp_path / "pusty.wav", [])


# --------------------------------------------------------------------------- #
# Sprawdzenia zależności (Faza 4 w --check-deps)
# --------------------------------------------------------------------------- #


def test_raport_zaleznosci_zawiera_pozycje_fazy_4(
    settings: Settings, voices_dir: Path, fake_sounddevice: FakeSoundDevice, fake_piper: FakePiper
) -> None:
    import audio.dependencies
    from config import DependencyContext, GPUInfo, OllamaStatus, detect_platform

    make_voice_files(voices_dir, "pl_PL-testowy-medium")
    context = DependencyContext(
        settings=settings,
        platform_info=detect_platform(),
        gpu=GPUInfo(False, "none", None, None, "CPU"),
        ollama=OllamaStatus("http://x", False, None, (), "model", False, None, None),
        user_settings=UserSettings(),
        offline=True,
    )

    checks = {check.name: check for check in audio.dependencies.check_audio_stack(context)}

    assert checks[t("deps.tts.engine_name")].ok is True
    assert checks[t("deps.voice.name")].ok is True
    assert "pl_PL-testowy-medium" in checks[t("deps.voice.name")].detail
    assert checks[t("deps.speaker.name")].ok is True
    assert all(check.required is False for check in checks.values())


def test_raport_podpowiada_skad_wziac_glos(
    settings: Settings, fake_sounddevice: FakeSoundDevice, no_piper: None
) -> None:
    import audio.dependencies
    from config import DependencyContext, GPUInfo, OllamaStatus, detect_platform

    context = DependencyContext(
        settings=settings,
        platform_info=detect_platform(),
        gpu=GPUInfo(False, "none", None, None, "CPU"),
        ollama=OllamaStatus("http://x", False, None, (), "model", False, None, None),
        user_settings=UserSettings(),
        offline=True,
    )

    checks = {check.name: check for check in audio.dependencies.check_audio_stack(context)}

    assert checks[t("deps.voice.name")].ok is False
    assert "prepare_offline.py --piper" in checks[t("deps.voice.name")].hint
    assert checks[t("deps.tts.engine_name")].ok is False
    assert "PIPER_BINARY" in checks[t("deps.tts.engine_name")].hint


def test_wylaczona_mowa_jest_widoczna_w_raporcie(
    settings: Settings, fake_sounddevice: FakeSoundDevice
) -> None:
    import audio.dependencies
    from config import DependencyContext, GPUInfo, OllamaStatus, detect_platform

    context = DependencyContext(
        settings=settings.model_copy(update={"tts_enabled": False}),
        platform_info=detect_platform(),
        gpu=GPUInfo(False, "none", None, None, "CPU"),
        ollama=OllamaStatus("http://x", False, None, (), "model", False, None, None),
        user_settings=UserSettings(),
        offline=True,
    )

    checks = {check.name: check for check in audio.dependencies.check_audio_stack(context)}

    assert checks[t("deps.tts.name")].ok is False
    assert "TTS_ENABLED" in checks[t("deps.tts.name")].detail


def test_import_audio_tts_nie_wymaga_pipera() -> None:
    """Brak pakietu nie może wywalić importu — inaczej padłby cały --check-deps."""
    assert "piper" not in sys.modules or sys.modules["piper"] is not None
    import audio.tts

    assert audio.tts.available_tts_engines() == ["none", "piper"]


# --------------------------------------------------------------------------- #
# Cała droga: tekst modelu → Piper → karta dźwiękowa
# --------------------------------------------------------------------------- #


def test_pelna_sciezka_od_tekstu_do_karty_dzwiekowej(
    settings: Settings,
    voices_dir: Path,
    fake_piper: FakePiper,
    fake_sounddevice: FakeSoundDevice,
) -> None:
    """Spięcie wszystkich elementów Fazy 4 bez ani jednego prawdziwego urządzenia."""
    from audio.output import AudioOutput

    make_voice_files(voices_dir, "pl_PL-testowy-medium")
    provider = PiperTTSProvider(settings, UserSettings(voice_volume=1.0))
    output = AudioOutput(settings, volume=1.0)
    speech = SpeechOutput(
        provider, output, settings=settings.model_copy(update={"tts_min_sentence_chars": 0})
    )

    speech.begin("pl")
    speech.feed("Pierwsze zdanie. ")
    speech.feed("Drugie zdanie.")
    speech.end(wait=False)
    speech.wait(timeout=10.0)

    stream = fake_sounddevice.output_streams[0]
    played = stream.pull_until_silent(frames=1_024)

    assert stream.samplerate == 22_050
    assert int(np.count_nonzero(played)) > 0
    assert [call["text"] for call in FakePiperVoice.instances[0].calls] == [
        "Pierwsze zdanie.",
        "Drugie zdanie.",
    ]
    speech.close()


# --------------------------------------------------------------------------- #
# Wpięcie w main.py: mowa jest wyjściem OPCJONALNYM
# --------------------------------------------------------------------------- #


def test_brak_mowy_nie_blokuje_rozmowy(
    settings: Settings, no_piper: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Na maszynie bez Pipera asystent ma działać dalej — tekstowo."""
    import main

    speaker = main.VoiceOutput(settings)

    assert speaker.enable() is False
    assert speaker.enabled is False
    # Wołanie metod mówienia na wyłączonym wyjściu nie może rzucać.
    speaker.begin("pl")
    speaker.feed("cokolwiek")
    speaker.end()
    speaker.cancel()
    speaker.close()
    # Nie porównujemy tekstu (interfejs bywa w różnych językach) — pytamy o STAN.
    assert speaker.is_unavailable


def test_glos_wylaczony_w_ustawieniach_nie_probuje_nic_ladowac(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import audio.tts
    import main

    monkeypatch.setattr(
        audio.tts, "get_user_settings", lambda: UserSettings(voice_engine="none")
    )
    speaker = main.VoiceOutput(settings)

    assert speaker.enable(quiet=True) is False
    assert speaker.enabled is False


def test_komenda_glos_przelacza_mowe(
    settings: Settings,
    voices_dir: Path,
    fake_piper: FakePiper,
    fake_sounddevice: FakeSoundDevice,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import main

    make_voice_files(voices_dir, "pl_PL-testowy-medium")
    speaker = main.VoiceOutput(settings)

    speaker.handle_command("on")
    assert speaker.enabled is True

    speaker.handle_command("off")
    assert speaker.enabled is False

    speaker.handle_command("lista")
    speaker.close()

    output = capsys.readouterr().out
    assert "pl_PL-testowy-medium" in output


def test_komenda_glos_model_zapisuje_wybor(
    settings: Settings,
    voices_dir: Path,
    tmp_path: Path,
    fake_piper: FakePiper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Podmiana głosu ma być zapisem w user_settings.json, nie zmianą kodu."""
    import config
    import main

    make_voice_files(voices_dir, "pl_PL-pierwszy-medium")
    make_voice_files(voices_dir, "pl_PL-drugi-medium")
    settings_file = tmp_path / "user_settings.json"
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(config, "_user_settings_cache", None, raising=False)

    speaker = main.VoiceOutput(settings)
    speaker.handle_command("model pl_PL-drugi-medium")
    speaker.close()

    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved["piper_model"] == "pl_PL-drugi-medium"


def test_komenda_glos_zapisuje_probke_do_pliku(
    settings: Settings,
    voices_dir: Path,
    tmp_path: Path,
    fake_piper: FakePiper,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ścieżka dla maszyn bez głośnika: sprawdź głos, zapisując go do WAV-a."""
    import main

    make_voice_files(voices_dir, "pl_PL-testowy-medium")
    target = tmp_path / "proba.wav"
    speaker = main.VoiceOutput(settings)

    speaker.handle_command(f"zapisz {target}")

    assert target.is_file()
    from i18n import t

    assert t("cli.speech.sample_saved", path=target) in capsys.readouterr().out


def test_odpowiedz_modelu_jest_mowiona_w_trakcie_pisania(settings: Settings) -> None:
    """``_stream_answer`` musi karmić syntezę tym samym strumieniem, co ekran."""
    import asyncio

    import main
    from brain.conversation import ConversationHistory

    class FakeClient:
        async def stream_chat(
            self, messages: Any, *, system: str | None = None, on_thinking: Any = None
        ) -> Any:
            for piece in ("Pierwsze zdanie. ", "Drugie ", "zdanie."):
                yield piece

    class RecordingSpeaker:
        def __init__(self) -> None:
            self.enabled = True
            self.pieces: list[str] = []
            self.language: str | None = None
            self.ended = False

        def begin(self, language: str | None = None) -> None:
            self.language = language

        def feed(self, text: str) -> None:
            self.pieces.append(text)

        def end(self) -> None:
            self.ended = True

        def cancel(self) -> None:
            self.ended = False

    speaker = RecordingSpeaker()
    history = ConversationHistory()
    history.add_user("pytanie")

    answer = asyncio.run(
        main._stream_answer(
            FakeClient(),  # type: ignore[arg-type]
            history,
            "[MIKU]",
            "prompt",
            speaker=speaker,  # type: ignore[arg-type]
            language="pl",
        )
    )

    assert answer == "Pierwsze zdanie. Drugie zdanie."
    assert speaker.pieces == ["Pierwsze zdanie. ", "Drugie ", "zdanie."]
    assert speaker.language == "pl"
    assert speaker.ended is True


def test_przerwana_odpowiedz_ucina_mowe(settings: Settings) -> None:
    import asyncio

    import main
    from brain.conversation import ConversationHistory

    class FailingClient:
        async def stream_chat(
            self, messages: Any, *, system: str | None = None, on_thinking: Any = None
        ) -> Any:
            yield "Zaczynam mówić. "
            raise RuntimeError("model padł w połowie")

    class RecordingSpeaker:
        def __init__(self) -> None:
            self.enabled = True
            self.cancelled = False

        def begin(self, language: str | None = None) -> None:
            return None

        def feed(self, text: str) -> None:
            return None

        def end(self) -> None:
            return None

        def cancel(self) -> None:
            self.cancelled = True

    speaker = RecordingSpeaker()
    history = ConversationHistory()
    history.add_user("pytanie")

    with pytest.raises(RuntimeError):
        asyncio.run(
            main._stream_answer(
                FailingClient(),  # type: ignore[arg-type]
                history,
                "[MIKU]",
                "prompt",
                speaker=speaker,  # type: ignore[arg-type]
            )
        )

    assert speaker.cancelled is True
