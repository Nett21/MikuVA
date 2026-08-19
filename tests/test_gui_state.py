"""Testy stanu pokazywanego w oknie: historii rozmowy i panelu stanu (Faza 10).

Bez ekranu i bez wątków — sprawdzamy to, co widget tylko rysuje. Najciekawsze
przypadki nie dotyczą „szczęśliwej ścieżki", a sytuacji, w których interfejs
łatwo pokazałby nieprawdę: przerwana odpowiedź (pusty dymek), migawka stanu
zmieniona w jednym polu i raport zależności, w którym czegoś nie ma.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from gui.state import (
    ChatLog,
    ChatRole,
    ListeningState,
    ServiceState,
    ServiceStatus,
    StatusSnapshot,
    default_services,
    services_from_report,
)
from gui.theme import build_palette
from i18n import t

PALETTE = build_palette("#39C5BB", "dark")


# --------------------------------------------------------------------------- #
# Historia rozmowy
# --------------------------------------------------------------------------- #


def test_wiadomosci_lada_w_kolejnosci() -> None:
    log = ChatLog()
    log.add(ChatRole.USER, "cześć")
    log.add(ChatRole.ASSISTANT, "hej")

    assert [message.role for message in log.messages] == [ChatRole.USER, ChatRole.ASSISTANT]
    assert len(log) == 2


def test_strumien_odpowiedzi_dopisuje_do_jednego_babelka() -> None:
    """Fragmenty modelu nie mogą tworzyć nowej wiadomości na każde słowo."""
    log = ChatLog()
    log.start_assistant()
    log.append_chunk("Pierwsze ")
    log.append_chunk("zdanie.")

    assert len(log) == 1
    assert log.is_streaming
    message = log.finish()
    assert message is not None
    assert message.text == "Pierwsze zdanie."
    assert not message.streaming
    assert not log.is_streaming


def test_przerwana_odpowiedz_nie_zostawia_pustego_babelka() -> None:
    """Model przerwany przed pierwszym znakiem: dymek znika, nie wisi pusty."""
    log = ChatLog()
    log.start_assistant()

    assert log.finish() is None
    assert len(log) == 0


def test_zamkniecie_strumienia_moze_nadpisac_tresc() -> None:
    log = ChatLog()
    log.start_assistant()
    log.append_chunk("część")

    message = log.finish("pełna odpowiedź")

    assert message is not None and message.text == "pełna odpowiedź"


def test_dopisanie_bez_otwartego_strumienia_otwiera_go_samo() -> None:
    log = ChatLog()
    log.append_chunk("nagły fragment")

    assert len(log) == 1
    assert log.messages[0].role is ChatRole.ASSISTANT


def test_nowy_strumien_zamyka_poprzedni() -> None:
    log = ChatLog()
    log.start_assistant()
    log.append_chunk("pierwsza")
    log.start_assistant()
    log.append_chunk("druga")

    assert [message.text for message in log.messages] == ["pierwsza", "druga"]
    assert log.messages[0].streaming is False


def test_okno_rozmowy_nie_rosnie_bez_konca() -> None:
    log = ChatLog(max_messages=10)
    for index in range(40):
        log.add(ChatRole.USER, f"wiadomość {index}")

    assert len(log) == 10
    assert log.messages[-1].text == "wiadomość 39"


def test_czyszczenie_kasuje_takze_otwarty_strumien() -> None:
    log = ChatLog()
    log.start_assistant()
    log.clear()

    assert len(log) == 0 and not log.is_streaming


def test_kazda_rola_ma_wlasny_kolor_z_palety() -> None:
    log = ChatLog()
    kolory = {
        role: log.add(role, "tekst").bubble_colors(PALETTE)
        for role in (
            ChatRole.USER,
            ChatRole.ASSISTANT,
            ChatRole.TOOL,
            ChatRole.SYSTEM,
            ChatRole.ERROR,
        )
    }

    assert kolory[ChatRole.USER][0] == PALETTE.user_bubble
    assert kolory[ChatRole.ERROR][0] == PALETTE.error_bubble
    # Bąbelek użytkownika i asystenta muszą się różnić, inaczej rozmowa jest
    # ścianą tekstu bez podziału na strony.
    assert kolory[ChatRole.USER][0] != kolory[ChatRole.ASSISTANT][0]


def test_godzina_wiadomosci_w_czasie_lokalnym() -> None:
    log = ChatLog()
    message = log.add(ChatRole.USER, "cześć")
    message.created_at = datetime(2026, 8, 17, 13, 42, tzinfo=timezone.utc)

    assert ":" in message.time_label()
    assert len(message.time_label()) == 5


# --------------------------------------------------------------------------- #
# Wskaźnik nasłuchiwania
# --------------------------------------------------------------------------- #


def test_wskaznik_pulsuje_tylko_gdy_cos_sie_dzieje() -> None:
    assert ListeningState.LISTENING.is_active
    assert ListeningState.THINKING.is_active
    assert ListeningState.SPEAKING.is_active
    assert not ListeningState.IDLE.is_active
    assert not ListeningState.OFF.is_active
    assert not ListeningState.WAITING_WAKE.is_active


def test_podpis_wskaznika_mowi_o_frazie() -> None:
    caption = ListeningState.WAITING_WAKE.caption(wake_phrase="hej Aiko")
    assert "hej Aiko" in caption
    # Bez znanej frazy podpis nadal ma sens (nie „czekam na ""”).
    assert ListeningState.WAITING_WAKE.caption() == t("listening.waiting_wake_generic")


def test_kolor_wskaznika_pochodzi_z_palety() -> None:
    assert ListeningState.LISTENING.color(PALETTE) == PALETTE.listening_active
    assert ListeningState.THINKING.color(PALETTE) == PALETTE.listening_busy
    assert ListeningState.OFF.color(PALETTE) == PALETTE.listening_idle


def test_stan_mikrofonu_wynika_ze_stanu_nasluchu() -> None:
    assert ListeningState.WAITING_WAKE.is_microphone_on
    assert ListeningState.LISTENING.is_microphone_on
    assert not ListeningState.THINKING.is_microphone_on


# --------------------------------------------------------------------------- #
# Migawka stanu
# --------------------------------------------------------------------------- #


def test_migawka_startowa_nie_udaje_ze_wszystko_dziala() -> None:
    """Przed sprawdzeniem stan jest NIEZNANY, a nie „ok" — to różnica dla użytkownika."""
    for item in default_services():
        assert item.state is ServiceState.UNKNOWN


def test_zmiana_jednej_pozycji_nie_rusza_pozostalych() -> None:
    snapshot = StatusSnapshot()
    changed = snapshot.with_service("mic", ServiceState.OK, "wbudowany mikrofon")

    assert changed.service("mic") is not None
    assert changed.service("mic").state is ServiceState.OK  # type: ignore[union-attr]
    assert changed.service("ollama").state is ServiceState.UNKNOWN  # type: ignore[union-attr]
    # Migawka jest niezmienna — pierwotna zostaje bez zmian (podróżuje między wątkami).
    assert snapshot.service("mic").state is ServiceState.UNKNOWN  # type: ignore[union-attr]


def test_nieznana_pozycja_zostaje_dopisana() -> None:
    snapshot = StatusSnapshot().with_service("rvc", ServiceState.OFF, "wyłączone")
    assert snapshot.service("rvc") is not None


def test_opis_jezyka_rozroznia_ustawiony_od_automatu() -> None:
    forced = StatusSnapshot(language="en", language_forced=True)
    auto = StatusSnapshot(language="auto", language_forced=False)

    assert forced.language_label() == t("gui.status.language_forced", code="en")
    assert auto.language_label() == t("gui.status.language_auto")


def test_podsumowanie_stanu_nadaje_sie_do_logu() -> None:
    snapshot = StatusSnapshot(
        assistant_name="Aiko", model="qwen2.5", host="http://127.0.0.1:11434", language="pl"
    )
    lines = snapshot.summary_lines()

    assert any("Aiko" in line and "qwen2.5" in line for line in lines)
    # Podsumowanie mówi o języku odpowiedzi — w języku interfejsu, więc
    # porównujemy przez katalog, a nie przez polskie słowo.
    assert any(t("gui.status.language", language=snapshot.language_label()) == line for line in lines)


def test_zmiana_stanu_nasluchu_zachowuje_reszte_migawki() -> None:
    snapshot = StatusSnapshot(assistant_name="Aiko").with_service("mic", ServiceState.OK, "ok")
    changed = snapshot.with_listening(ListeningState.LISTENING)

    assert changed.listening is ListeningState.LISTENING
    assert changed.assistant_name == "Aiko"
    assert changed.service("mic").state is ServiceState.OK  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Raport zależności → panel stanu
# --------------------------------------------------------------------------- #


@dataclass
class FakeCheck:
    name: str
    ok: bool
    detail: str = ""
    required: bool = False


@dataclass
class FakeOllama:
    reachable: bool
    model_present: bool
    detail: str = ""


@dataclass
class FakeReport:
    checks: tuple[FakeCheck, ...] = ()
    ollama: FakeOllama | None = None


def test_stan_z_raportu_pokazuje_prawde_o_maszynie() -> None:
    report = FakeReport(
        checks=(
            FakeCheck(name=t("deps.mic.name"), ok=True, detail="wbudowany"),
            FakeCheck(name=t("deps.whisper.cache_name"), ok=False, detail="brak modelu"),
        ),
        ollama=FakeOllama(reachable=True, model_present=True, detail="działa"),
    )

    items = {item.key: item for item in services_from_report(report)}

    assert items["mic"].state is ServiceState.OK
    assert items["whisper"].state is ServiceState.OFF
    assert items["ollama"].state is ServiceState.OK


def test_nazwa_sprawdzenia_moze_byc_uszczegolowiona() -> None:
    """Raport dopisuje do nazwy szczegół — panel i tak ma ją znaleźć."""
    report = FakeReport(
        checks=(
            FakeCheck(name=t("deps.whisper.cache_name") + " (extra)", ok=True, detail="tiny"),
        )
    )

    items = {item.key: item for item in services_from_report(report)}

    assert items["whisper"].state is ServiceState.OK
    assert items["whisper"].detail == "tiny"


def test_ollama_bez_modelu_to_blad_a_nie_ostrzezenie() -> None:
    report = FakeReport(ollama=FakeOllama(reachable=True, model_present=False))
    items = {item.key: item for item in services_from_report(report)}

    assert items["ollama"].state is ServiceState.ERROR
    assert items["ollama"].detail == t("deps.ollama.no_model")


def test_niepelny_raport_nie_wywraca_panelu() -> None:
    """Część sprawdzeń rejestrują moduły, których na tej maszynie nie ma."""
    items = {item.key: item for item in services_from_report(FakeReport())}

    assert items["mic"].state is ServiceState.UNKNOWN
    assert "ollama" not in items
    # Zupełnie obcy obiekt też nie może niczego wywrócić.
    assert services_from_report(object()) is not None


def test_pozycja_statusu_ma_czytelna_linijke() -> None:
    assert ServiceStatus(key="mic", label="Mikrofon", detail="wbudowany").line() == (
        "Mikrofon: wbudowany"
    )
    assert ServiceStatus(key="mic", label="Mikrofon").line() == "Mikrofon"
