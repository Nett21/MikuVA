"""Testy transkrypcji — na atrapie faster_whisper, bez pobierania modeli."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from conftest import FakeFasterWhisper, FakeSegment, FakeWhisperModel, make_tone

import config
from audio.vad import Utterance
from audio.whisper import (
    PAD_LEAD_S,
    PAD_TAIL_S,
    TranscriptionError,
    WhisperTranscriber,
    is_hallucination,
    resolve_device,
)
from config import GPUInfo, Settings, UserSettings

CUDA_PRESENT = GPUInfo(
    cuda_available=True,
    source="nvidia-smi",
    device_name="Atrapa GPU",
    driver_version="999",
    detail="atrapa",
)
CUDA_ABSENT = GPUInfo(
    cuda_available=False,
    source="none",
    device_name=None,
    driver_version=None,
    detail="brak GPU",
)


def utterance_of(seconds: float = 1.0) -> Utterance:
    samples = make_tone(int(16_000 * seconds))
    return Utterance(samples=samples, sample_rate=16_000, started_at=0.0, ended_at=seconds)


# --------------------------------------------------------------------------- #
# Dobór urządzenia
# --------------------------------------------------------------------------- #


def test_auto_wybiera_cpu_bez_gpu() -> None:
    settings = Settings(_env_file=None, whisper_device="auto", whisper_compute_type="auto")
    assert resolve_device(settings, CUDA_ABSENT) == ("cpu", "int8")


def test_auto_wybiera_cuda_gdy_jest_gpu() -> None:
    settings = Settings(_env_file=None, whisper_device="auto", whisper_compute_type="auto")
    assert resolve_device(settings, CUDA_PRESENT) == ("cuda", "float16")


def test_wymuszone_cuda_bez_gpu_spada_na_cpu() -> None:
    settings = Settings(_env_file=None, whisper_device="cuda", whisper_compute_type="auto")
    assert resolve_device(settings, CUDA_ABSENT) == ("cpu", "int8")


def test_jawny_compute_type_ma_pierwszenstwo() -> None:
    settings = Settings(_env_file=None, whisper_device="cpu", whisper_compute_type="float32")
    assert resolve_device(settings, CUDA_ABSENT) == ("cpu", "float32")


# --------------------------------------------------------------------------- #
# Ładowanie modelu
# --------------------------------------------------------------------------- #


def test_model_laduje_sie_do_katalogu_projektu(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    from config import WHISPER_CACHE_DIR

    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()

    model = FakeWhisperModel.instances[-1]
    assert model.download_root == str(WHISPER_CACHE_DIR)
    assert model.device == "cpu"
    assert model.local_files_only is False


def test_awaria_cuda_konczy_sie_fallbackiem_na_cpu(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Najczęstszy przypadek z życia: sterownik jest, brakuje cuDNN."""
    module = FakeFasterWhisper(fail_on_devices=("cuda",))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FakeWhisperModel.instances.clear()

    cuda_settings = settings.model_copy(
        update={"whisper_device": "auto", "whisper_compute_type": "auto"}
    )
    transcriber = WhisperTranscriber(cuda_settings, gpu=CUDA_PRESENT)
    transcriber.load()

    assert transcriber.device == "cpu"
    assert transcriber.compute_type == "int8"
    assert FakeWhisperModel.instances[-1].device == "cpu"


def test_gpu_ktore_pada_dopiero_przy_inferencji_konczy_sie_na_cpu(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Regresja z prawdziwej maszyny: RTX 3060 widoczne, ale brak libcublas.

    Konstruktor WhisperModel przechodzi bez błędu, bo CTranslate2 ładuje
    biblioteki CUDA dopiero przy pierwszym liczeniu. Rozgrzewka w load() ma to
    wykryć, zanim użytkownik powie pierwsze zdanie.
    """
    module = FakeFasterWhisper(fail_inference_on_devices=("cuda",))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FakeWhisperModel.instances.clear()

    cuda_settings = settings.model_copy(
        update={"whisper_device": "auto", "whisper_compute_type": "auto"}
    )
    transcriber = WhisperTranscriber(cuda_settings, gpu=CUDA_PRESENT)
    transcriber.load()

    assert transcriber.device == "cpu"
    assert [model.device for model in FakeWhisperModel.instances] == ["cuda", "cpu"]

    transcript = transcriber.transcribe(utterance_of())
    assert transcript.text == "Cześć, tu atrapa."


def test_awaria_gpu_w_trakcie_pracy_przelacza_na_cpu(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    """Druga linia obrony: model już działał, a GPU padło w środku sesji."""
    module = FakeFasterWhisper()
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FakeWhisperModel.instances.clear()

    cuda_settings = settings.model_copy(
        update={"whisper_device": "auto", "whisper_compute_type": "auto"}
    )
    transcriber = WhisperTranscriber(cuda_settings, gpu=CUDA_PRESENT)
    transcriber.load()
    assert transcriber.device == "cuda"

    # Symulacja awarii sterownika już po rozgrzewce.
    FakeWhisperModel.instances[-1].inference_error = RuntimeError("CUDA error: out of memory")
    transcript = transcriber.transcribe(utterance_of())

    assert transcriber.device == "cpu"
    assert transcript.text == "Cześć, tu atrapa."


def test_brak_pakietu_daje_czytelny_blad(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    with pytest.raises(TranscriptionError, match="faster-whisper"):
        transcriber.load()


def test_brak_modelu_w_trybie_offline_ma_wlasciwa_podpowiedz(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    module = FakeFasterWhisper(fail_on_devices=("cpu",))
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    offline = settings.model_copy(update={"whisper_allow_download": False})
    transcriber = WhisperTranscriber(offline, gpu=CUDA_ABSENT)
    with pytest.raises(TranscriptionError) as error:
        transcriber.load()
    assert "WHISPER_ALLOW_DOWNLOAD" in error.value.hint


# --------------------------------------------------------------------------- #
# Transkrypcja
# --------------------------------------------------------------------------- #


def test_transkrypcja_zwraca_tekst_i_jezyk(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    FakeWhisperModel.instances[-1].segments = [FakeSegment("Dzień dobry.")]

    transcript = transcriber.transcribe(utterance_of(1.0))

    assert transcript.text == "Dzień dobry."
    assert transcript.language == "pl"
    assert transcript.is_empty is False
    assert transcript.audio_duration_s == pytest.approx(1.0, abs=0.01)


def test_auto_jezyk_idzie_za_jezykiem_asystenta(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """„auto" znaczy „ten sam język, w którym asystent odpowiada" — NIE „zgaduj".

    Zgadywanie jest mierzalnie gorsze: na dziesięciu krótkich polskich zdaniach
    Whisper rozpoznał trzy jako urdu, niemiecki i rosyjski i przepisał je w tych
    językach (WER 49,5% wobec 27,0%), a każda transkrypcja trwała dwa razy
    dłużej. Kto naprawdę mówi w kilku językach, wpisuje „detect".
    """
    # Ustawienia użytkownika podstawiamy jawnie: bez tego test czytałby PRAWDZIWY
    # config/user_settings.json z maszyny, na której akurat działa.
    _force_user_settings(monkeypatch, UserSettings(speech_language="auto"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == settings.language


def test_wymuszony_jezyk_z_konfiguracji(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    polish = settings.model_copy(update={"whisper_language": "pl"})
    transcriber = WhisperTranscriber(polish, gpu=CUDA_ABSENT)
    transcriber.load()
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "pl"


def test_jezyk_podany_w_wywolaniu_wygrywa(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    polish = settings.model_copy(update={"whisper_language": "pl"})
    transcriber = WhisperTranscriber(polish, gpu=CUDA_ABSENT)
    transcriber.load()
    FakeWhisperModel.instances[-1].info.language = "en"
    transcript = transcriber.transcribe(utterance_of(), language="en")

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "en"
    assert transcript.language == "en"


def test_wlasny_vad_wylacza_vad_whispera(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["vad_filter"] is False


def test_segmenty_o_wysokim_no_speech_sa_odrzucane(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    FakeWhisperModel.instances[-1].segments = [
        FakeSegment("Prawdziwe zdanie.", no_speech_prob=0.05),
        FakeSegment("Szum z ciszy", no_speech_prob=0.95),
    ]

    transcript = transcriber.transcribe(utterance_of())
    assert transcript.text == "Prawdziwe zdanie."


def test_halucynacje_z_ciszy_sa_odsiewane(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    FakeWhisperModel.instances[-1].segments = [
        FakeSegment("Napisy stworzone przez społeczność Amara.org")
    ]

    assert transcriber.transcribe(utterance_of()).is_empty is True


@pytest.mark.parametrize(
    "text",
    [
        "Napisy stworzone przez X",
        "Subtitles by ABC",
        "Thank you.",
        "...",
        "   ",
        # Zapętlenie na szumie — zaobserwowane na realnym mikrofonie.
        "No, no, no, no, no, no, no, no, no, no",
        "tak tak tak tak tak tak tak",
    ],
)
def test_wykrywanie_halucynacji(text: str) -> None:
    assert is_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Dziękuję, to działa",
        "Thanks a lot for the help",
        "Cześć",
        # Powtórzenia w normalnym zdaniu nie mogą go skasować.
        "tak, tak jest, to jest to",
        "no i co teraz zrobimy z tym fantem",
    ],
)
def test_normalny_tekst_nie_jest_halucynacja(text: str) -> None:
    assert is_hallucination(text) is False


def test_zbyt_krotki_fragment_nie_idzie_do_modelu(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    strict = settings.model_copy(update={"whisper_min_duration_s": 0.5})
    transcriber = WhisperTranscriber(strict, gpu=CUDA_ABSENT)
    transcriber.load()

    transcript = transcriber.transcribe(utterance_of(0.2))

    assert transcript.is_empty is True
    assert FakeWhisperModel.instances[-1].calls == []


def test_przeprobkowanie_przed_transkrypcja(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()

    samples_48k = make_tone(48_000)  # 1 s przy 48 kHz
    transcriber.transcribe(samples_48k, sample_rate=48_000)

    # Sekunda mowy + cisza dokładana na brzegach (patrz PAD_LEAD_S/PAD_TAIL_S).
    padding = int(16_000 * (PAD_LEAD_S + PAD_TAIL_S))
    assert FakeWhisperModel.instances[-1].calls[-1]["samples"] == 16_000 + padding


def test_wejscie_jest_konwertowane_na_float32(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    captured: dict[str, np.ndarray] = {}

    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]
    original = model.transcribe

    def spy(audio: np.ndarray, **kwargs: object) -> object:
        captured["audio"] = audio
        return original(audio, **kwargs)

    model.transcribe = spy  # type: ignore[method-assign]
    transcriber.transcribe(utterance_of())

    assert captured["audio"].dtype == np.float32
    assert float(np.max(np.abs(captured["audio"]))) <= 1.0


def test_transkrypcja_asynchroniczna(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    import asyncio

    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcript = asyncio.run(transcriber.transcribe_async(utterance_of()))
    assert transcript.text == "Cześć, tu atrapa."


# --------------------------------------------------------------------------- #
# Język mowy (preferencja użytkownika)
# --------------------------------------------------------------------------- #


def _force_user_settings(monkeypatch: pytest.MonkeyPatch, user: UserSettings) -> None:
    """Odetnij testy od prawdziwego config/user_settings.json."""
    monkeypatch.setattr(config, "get_user_settings", lambda: user)


def test_jezyk_z_ustawien_uzytkownika_wymusza_jezyk_transkrypcji(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    _force_user_settings(monkeypatch, UserSettings(speech_language="en"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "en"


def test_detect_zostawia_rozpoznawanie_jezyka_whisperowi(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """„detect" to jedyna wartość, która oddaje decyzję Whisperowi."""
    _force_user_settings(monkeypatch, UserSettings(speech_language="detect"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] is None


def test_auto_bierze_jezyk_z_ustawien_asystenta(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper
) -> None:
    _force_user_settings(monkeypatch, UserSettings(speech_language="auto"))
    polskie = Settings(_env_file=None, language="pl", whisper_language="")
    transcriber = WhisperTranscriber(polskie, gpu=CUDA_ABSENT)
    transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "pl"


def test_whisper_language_z_env_dziala_gdy_uzytkownik_ma_auto(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """`.env` zostaje technicznym nadpisaniem, ale przegrywa z wyborem użytkownika."""
    _force_user_settings(monkeypatch, UserSettings(speech_language="auto"))
    with_env = settings.model_copy(update={"whisper_language": "de"})
    WhisperTranscriber(with_env, gpu=CUDA_ABSENT).transcribe(utterance_of())
    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "de"

    _force_user_settings(monkeypatch, UserSettings(speech_language="pl"))
    WhisperTranscriber(with_env, gpu=CUDA_ABSENT).transcribe(utterance_of())
    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] == "pl"


def test_kod_jezyka_jest_normalizowany() -> None:
    assert UserSettings(speech_language="PL-pl").speech_language == "pl"
    assert UserSettings(speech_language="  En  ").speech_language == "en"
    assert UserSettings(speech_language="").speech_language == "auto"
    assert UserSettings(speech_language="auto").is_speech_language_forced is False
    assert UserSettings(speech_language="pl").is_speech_language_forced is True


def test_nieznany_kod_jezyka_wraca_do_rozpoznawania_automatycznego(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Literówka w user_settings.json nie może wysypywać każdej transkrypcji."""
    _force_user_settings(monkeypatch, UserSettings(speech_language="xx"))
    monkeypatch.setattr("audio.whisper.supported_languages", lambda: frozenset({"pl", "en"}))

    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcript = transcriber.transcribe(utterance_of())

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["language"] is None
    assert transcript.text == "Cześć, tu atrapa."


# --------------------------------------------------------------------------- #
# Podpowiedź frazy i cisza na brzegach
# --------------------------------------------------------------------------- #


def test_podpowiedz_frazy_trafia_do_modelu(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Nazwa własna bez podpowiedzi wychodzi z modelu jako „tymiku" albo „micu".

    Pomiar na korpusie 480 transkrypcji: sama podpowiedź podniosła wykrywanie
    frazy z 45% na 95% (model ``base``).
    """
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.transcribe(utterance_of(), hotwords="hej Miku")

    assert FakeWhisperModel.instances[-1].calls[-1]["kwargs"]["hotwords"] == "hej Miku"


def test_bez_podpowiedzi_nie_dokladamy_argumentu(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.transcribe(utterance_of())

    assert "hotwords" not in FakeWhisperModel.instances[-1].calls[-1]["kwargs"]


def test_starsza_wersja_bez_podpowiedzi_dziala_dalej(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Nieznany argument kończyłby się wyjątkiem przy KAŻDEJ wypowiedzi."""
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]

    def stara_sygnatura(
        audio, *, language=None, beam_size=5, vad_filter=False, condition_on_previous_text=False
    ):
        model.calls.append({"samples": audio.size, "kwargs": {"language": language}})
        return list(model.segments), model.info

    monkeypatched = transcriber._model
    monkeypatched.transcribe = stara_sygnatura  # type: ignore[method-assign]

    wynik = transcriber.transcribe(utterance_of(), hotwords="hej Miku")

    assert wynik.text  # transkrypcja się udała, tyle że bez podpowiedzi


def test_cisza_jest_dokladana_ale_nie_liczy_sie_do_dlugosci(
    fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Próg „za krótkie nagranie" ma dotyczyć mowy, a nie naszego dopełnienia."""
    krotkie = Settings(_env_file=None, whisper_min_duration_s=1.0)
    transcriber = WhisperTranscriber(krotkie, gpu=CUDA_ABSENT)
    transcriber.load()

    # Pół sekundy mowy: krócej niż próg, mimo że z ciszą byłoby 1,1 s.
    wynik = transcriber.transcribe(make_tone(8_000), sample_rate=16_000)

    assert wynik.text == ""
    assert not FakeWhisperModel.instances[-1].calls


# --------------------------------------------------------------------------- #
# Użytkownik mówiący w dwóch językach
# --------------------------------------------------------------------------- #


def test_lista_jezykow_ogranicza_rozpoznawanie_zamiast_wymuszac(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """„pl,en" = rozpoznawaj, ale wybierz tylko spośród MOICH języków.

    Zgłoszone z prawdziwego użycia: „po ang działa dobrze, jak mówię po pl to
    prawie wgl nie wykrywa". Przy wymuszonym angielskim polska wypowiedź miała
    WER 114% — tekst nie nadawał się do niczego. Wymuszenie polskiego psuło
    angielski tak samo (55,8%). Lista jest jedynym układem, w którym obie
    strony działają.
    """
    _force_user_settings(monkeypatch, UserSettings(speech_language="pl,en"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]
    model.detected_languages = [("pl", 0.91), ("en", 0.05), ("ur", 0.02)]

    transcriber.transcribe(utterance_of())

    assert model.calls[-1]["kwargs"]["language"] == "pl"


def test_z_listy_nie_wychodzi_jezyk_spoza_niej(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Nawet gdy Whisper jest pewny urdu, dostajemy język z listy użytkownika.

    To nie jest teoria: „Ten mikser jest zepsuty" wróciło z modelu zapisane
    pismem arabskim, a „Dziękuję" cyrylicą.
    """
    _force_user_settings(monkeypatch, UserSettings(speech_language="pl,en"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]
    model.detected_languages = [("ur", 0.80), ("en", 0.12), ("pl", 0.05)]

    transcriber.transcribe(utterance_of())

    assert model.calls[-1]["kwargs"]["language"] == "en"


def test_jeden_kod_dalej_wymusza_bez_rozpoznawania(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    """Jeden język = zero dodatkowej pracy: nie ma czego rozpoznawać."""
    _force_user_settings(monkeypatch, UserSettings(speech_language="pl"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]

    transcriber.transcribe(utterance_of())

    assert model.calls[-1]["kwargs"]["language"] == "pl"
    assert model.language_detections == 0


def test_awaria_rozpoznawania_nie_przerywa_transkrypcji(
    monkeypatch: pytest.MonkeyPatch, fake_faster_whisper: FakeFasterWhisper, settings: Settings
) -> None:
    _force_user_settings(monkeypatch, UserSettings(speech_language="pl,en"))
    transcriber = WhisperTranscriber(settings, gpu=CUDA_ABSENT)
    transcriber.load()
    model = FakeWhisperModel.instances[-1]
    model.detection_error = RuntimeError("model nie umie rozpoznawać")

    wynik = transcriber.transcribe(utterance_of())

    assert wynik.text  # transkrypcja poszła dalej, z pełnym rozpoznawaniem
    assert model.calls[-1]["kwargs"]["language"] is None
