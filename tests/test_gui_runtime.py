"""Testy wątku roboczego GUI (Faza 10).

Wymaganie fazy brzmi „GUI nie może blokować głównego wątku", więc dokładnie to
jest tu sprawdzane — bez otwierania okna. Cały interfejs sprowadza się do dwóch
kolejek: komendy w jedną stronę, zdarzenia w drugą. Testy stają po stronie okna:
wołają metody, których używałby przycisk, i czekają na zdarzenia, które
narysowałby widget.

Nic tu nie dotyka Ollamy, mikrofonu ani karty dźwiękowej: model to scenariusz
odpowiedzi (``FakeToolLLM`` z ``conftest``), a mikrofon — atrapa podstawiona w
miejsce ``GuiListener``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeToolLLM, LLMStep, make_fake_tool

import brain.tool_router as tool_router_module
import gui.runtime as runtime_module
from config import Settings
from gui.runtime import AssistantRuntime, EventKind, RuntimeEvent
from gui.state import ChatRole, ListeningState
from i18n import t
from security.audit import AuditLog
from security.policy import SecurityPolicy
from security.risk import RiskLevel
from tools.registry import ToolRegistry

TIMEOUT = 10.0


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "memory_enabled": False,
        "embeddings_enabled": False,
        "tools_enabled": False,
        "tts_enabled": False,
        "mic_enabled": False,
        "language": "pl",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


class Collector:
    """Kolejka zdarzeń po stronie „okna" — z czekaniem zamiast sleepów."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []
        self._lock = threading.Lock()

    def __call__(self, event: RuntimeEvent) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> list[RuntimeEvent]:
        with self._lock:
            return list(self.events)

    def wait_for(
        self,
        kind: EventKind,
        *,
        timeout: float = TIMEOUT,
        where: Callable[[RuntimeEvent], bool] | None = None,
    ) -> RuntimeEvent:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.snapshot():
                if event.kind is kind and (where is None or where(event)):
                    return event
            time.sleep(0.01)
        raise AssertionError(
            f"nie doczekałam się {kind}; były: {[event.kind for event in self.snapshot()]}"
        )

    def texts(self, kind: EventKind) -> list[str]:
        return [event.text for event in self.snapshot() if event.kind is kind]

    def answer(self) -> str:
        """Odpowiedź modelu złożona z fragmentów — dokładnie to, co widzi okno."""
        return "".join(self.texts(EventKind.REPLY_CHUNK))


class FakeClient(FakeToolLLM):
    """Model z ``conftest`` uzupełniony o to, czego używa wątek roboczy."""

    def __init__(self, steps: Sequence[LLMStep], *, delay_s: float = 0.0) -> None:
        super().__init__(steps)
        self.delay_s = delay_s
        self.closed = False

    async def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
        async for chunk in super().stream_chat(messages, **kwargs):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield chunk

    async def list_models(self) -> list[str]:
        return ["model-a", "model-b"]

    async def aclose(self) -> None:
        self.closed = True


class FakeListener:
    """Mikrofon-atrapa: oddaje zaplanowane wypowiedzi, potem samą ciszę."""

    def __init__(self, settings: Any, publish: Any, *, on_state: Any = None) -> None:
        self.settings = settings
        self.publish = publish
        self.on_state = on_state
        self.active = False
        self.starts = 0
        self.stops = 0
        self.can_start = True
        self.windows_opened = 0
        self.transcripts: list[str] = []

    def start(self) -> bool:
        self.starts += 1
        self.active = self.can_start
        return self.can_start

    def open_window(self) -> None:
        # Kliknięcie „Słuchaj" otwiera okno rozmowy bez frazy wybudzającej.
        self.windows_opened += 1

    def listen_slice(self) -> str | None:
        if self.transcripts:
            text = self.transcripts.pop(0)
            if self.on_state is not None:
                self.on_state(ListeningState.LISTENING)
            return text
        time.sleep(0.02)  # atrapa nie może kręcić pętli na pełnych obrotach
        return None

    def stop(self) -> None:
        self.stops += 1
        self.active = False

    def close(self) -> None:
        self.active = False

    def describe(self) -> str:
        return "atrapa mikrofonu"

    def wake_phrase(self) -> str:
        return "hej testowo"

    def wake_detail(self) -> str:
        return "atrapa bramki"


@pytest.fixture
def fake_listener(monkeypatch: pytest.MonkeyPatch) -> list[FakeListener]:
    """Podstaw atrapę mikrofonu i oddaj listę utworzonych egzemplarzy."""
    created: list[FakeListener] = []

    def factory(settings: Any, publish: Any, *, on_state: Any = None) -> FakeListener:
        listener = FakeListener(settings, publish, on_state=on_state)
        created.append(listener)
        return listener

    monkeypatch.setattr(runtime_module, "GuiListener", factory)
    return created


def start_runtime(
    client: Any, *, settings: Settings | None = None, **kwargs: Any
) -> tuple[AssistantRuntime, Collector]:
    """Uruchom wątek roboczy z podstawionym „modelem"."""
    import brain.llm

    collector = Collector()
    active = settings or make_settings()

    original = brain.llm.OllamaClient
    brain.llm.OllamaClient = lambda *args, **values: client  # type: ignore[assignment, misc]
    try:
        runtime = AssistantRuntime(
            active, publish=collector, speech_enabled=False, **kwargs
        )
        runtime.start()
        collector.wait_for(EventKind.READY)
    finally:
        brain.llm.OllamaClient = original  # type: ignore[assignment]
    return runtime, collector


@pytest.fixture
def runtime_factory() -> Iterator[Callable[..., tuple[AssistantRuntime, Collector]]]:
    """Uruchamiaj wątki roboczo i domykaj je nawet po błędzie testu."""
    started: list[AssistantRuntime] = []

    def build(client: Any, **kwargs: Any) -> tuple[AssistantRuntime, Collector]:
        runtime, collector = start_runtime(client, **kwargs)
        started.append(runtime)
        return runtime, collector

    yield build

    for runtime in started:
        runtime.close(timeout=5.0)


# --------------------------------------------------------------------------- #
# Podstawowy przepływ tury
# --------------------------------------------------------------------------- #


def test_tura_trafia_do_okna_jako_zdarzenia(runtime_factory: Any) -> None:
    client = FakeClient([LLMStep(chunks=("Cześć", ", ", "jestem tutaj."))])
    runtime, events = runtime_factory(client)

    runtime.send("dzień dobry")
    events.wait_for(EventKind.REPLY_END)

    user = events.wait_for(EventKind.MESSAGE, where=lambda item: item.role is ChatRole.USER)
    assert user.text == "dzień dobry"
    assert events.answer() == "Cześć, jestem tutaj."


def test_wyslanie_wiadomosci_nie_blokuje_watku_okna(runtime_factory: Any) -> None:
    """Sedno wymagania: metoda wołana przez przycisk wraca natychmiast.

    Model „pisze" pół sekundy; gdyby cokolwiek liczyło się w wątku wołającym,
    poniższy pomiar by to pokazał.
    """
    client = FakeClient([LLMStep(chunks=tuple("abcdefghij"))], delay_s=0.05)
    runtime, events = runtime_factory(client)

    start = time.perf_counter()
    runtime.send("policz do dziesięciu")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"send() zablokowało wątek na {elapsed:.3f} s"
    # Dopiero teraz czekamy — odpowiedź powstaje w tle.
    events.wait_for(EventKind.REPLY_END)
    assert events.answer() == "abcdefghij"


def test_przerwanie_generowania_konczy_ture(runtime_factory: Any) -> None:
    client = FakeClient([LLMStep(chunks=tuple("x" * 200))], delay_s=0.05)
    runtime, events = runtime_factory(client)

    runtime.send("mów długo")
    events.wait_for(EventKind.REPLY_CHUNK)
    runtime.cancel()

    events.wait_for(
        EventKind.MESSAGE,
        where=lambda item: item.text == t("runtime.generation_interrupted"),
    )
    # Po przerwaniu asystent musi dalej przyjmować polecenia.
    runtime.send("jesteś tam?")
    events.wait_for(EventKind.REPLY_END, timeout=TIMEOUT)


def test_blad_modelu_jest_komunikatem_a_nie_koncem_pracy(runtime_factory: Any) -> None:
    from brain.llm import LLMConnectionError

    class BrokenClient(FakeClient):
        async def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
            if self.index == 0:
                self.index += 1
                raise LLMConnectionError("Ollama nie odpowiada", hint="uruchom `ollama serve`")
            async for chunk in FakeToolLLM.stream_chat(self, messages, **kwargs):
                yield chunk

    client = BrokenClient([LLMStep(chunks=("Jestem z powrotem.",))])
    runtime, events = runtime_factory(client)

    runtime.send("halo?")
    error = events.wait_for(EventKind.ERROR)
    assert "Ollama nie odpowiada" in error.text

    runtime.send("a teraz?")
    # Czekamy na treść, a nie na samo REPLY_END: błąd też domyka dymek
    # odpowiedzi (z ``detail="error"``), więc to zdarzenie już wystąpiło.
    events.wait_for(
        EventKind.REPLY_CHUNK, where=lambda item: "Jestem z powrotem." in item.text
    )


def test_nowa_rozmowa_nie_wywraca_watku(runtime_factory: Any) -> None:
    client = FakeClient([LLMStep(chunks=("ok",))])
    runtime, events = runtime_factory(client)

    runtime.new_conversation()
    runtime.send("pierwsze pytanie")

    events.wait_for(EventKind.REPLY_END)


# --------------------------------------------------------------------------- #
# Mikrofon
# --------------------------------------------------------------------------- #


def test_wypowiedz_z_mikrofonu_wchodzi_w_ture(
    runtime_factory: Any, fake_listener: list[FakeListener]
) -> None:
    client = FakeClient([LLMStep(chunks=("Słyszę cię.",))])
    runtime, events = runtime_factory(client)

    runtime.start_listening()
    deadline = time.monotonic() + TIMEOUT
    while not fake_listener and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_listener, "atrapa mikrofonu nie została utworzona"
    fake_listener[0].transcripts.append("jaka jest godzina")

    user = events.wait_for(
        EventKind.MESSAGE, where=lambda item: item.role is ChatRole.USER
    )
    assert user.text == "jaka jest godzina"
    assert user.detail == t("gui.detail.microphone")
    events.wait_for(EventKind.REPLY_END)


def test_wylaczenie_nasluchu_zwalnia_mikrofon(
    runtime_factory: Any, fake_listener: list[FakeListener]
) -> None:
    runtime, _events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))

    runtime.start_listening()
    # Czekamy, aż wątek roboczy FAKTYCZNIE otworzy mikrofon — inaczej test
    # sprawdzałby zwolnienie czegoś, czego nikt jeszcze nie zajął.
    deadline = time.monotonic() + TIMEOUT
    while (not fake_listener or fake_listener[0].starts == 0) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_listener[0].starts == 1
    assert runtime.is_listening
    # Kliknięcie „Słuchaj" jest jawnym zawołaniem — fraza wybudzająca nie jest
    # już potrzebna do pierwszej wypowiedzi.
    assert fake_listener[0].windows_opened == 1

    runtime.stop_listening()
    deadline = time.monotonic() + TIMEOUT
    while fake_listener[0].stops == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert fake_listener[0].stops >= 1
    assert not runtime.is_listening


def test_brak_mikrofonu_nie_zapetla_prob(
    runtime_factory: Any, fake_listener: list[FakeListener]
) -> None:
    """Maszyna bez mikrofonu: jedna próba, komunikat, koniec — nie co dwie sekundy."""
    runtime, _events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))

    deadline = time.monotonic() + TIMEOUT
    while not fake_listener and time.monotonic() < deadline:
        runtime.refresh()
        time.sleep(0.01)
    fake_listener[0].can_start = False

    runtime.start_listening()
    time.sleep(0.3)

    assert fake_listener[0].starts == 1
    assert not runtime.is_listening


# --------------------------------------------------------------------------- #
# Narzędzia i potwierdzenia
# --------------------------------------------------------------------------- #


def install_router(
    monkeypatch: pytest.MonkeyPatch, tool: Any, *, settings: Settings
) -> None:
    """Podstaw router z jednym narzędziem-atrapą, zachowując kanał potwierdzeń GUI."""

    def build(active: Any = None, **kwargs: Any) -> Any:
        used = active or settings
        return tool_router_module.ToolRouter(
            ToolRegistry([tool]),
            settings=used,
            policy=SecurityPolicy(used),
            broker=kwargs.get("broker"),
            audit=AuditLog(enabled=False),
        )

    monkeypatch.setattr(tool_router_module, "build_router", build)


def test_narzedzie_wysokiego_ryzyka_pyta_okno_o_zgode(
    runtime_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tools_enabled=True)
    tool = make_fake_tool(name="test.pisz", risk=RiskLevel.HIGH)
    install_router(monkeypatch, tool, settings=settings)

    client = FakeClient(
        [
            LLMStep(tool_calls=[{"function": {"name": "test.pisz", "arguments": {}}}]),
            LLMStep(chunks=("Zrobione.",)),
        ]
    )
    runtime, events = runtime_factory(client, settings=settings)

    runtime.send("wykonaj to zadanie")
    request = events.wait_for(EventKind.CONFIRM)

    # Dopóki nikt nie kliknął, narzędzie NIE zostało wykonane.
    assert tool.calls == []
    runtime.answer_confirmation(request.data.request_id, True)

    events.wait_for(EventKind.REPLY_END)
    assert len(tool.calls) == 1
    assert "Zrobione." in events.answer()
    assert events.texts(EventKind.TOOL)


def test_odmowa_w_oknie_nie_wykonuje_narzedzia(
    runtime_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tools_enabled=True)
    tool = make_fake_tool(name="test.pisz", risk=RiskLevel.HIGH)
    install_router(monkeypatch, tool, settings=settings)

    client = FakeClient(
        [
            LLMStep(tool_calls=[{"function": {"name": "test.pisz", "arguments": {}}}]),
            LLMStep(chunks=("Nie mogę tego zrobić.",)),
        ]
    )
    runtime, events = runtime_factory(client, settings=settings)

    runtime.send("wykonaj to zadanie")
    request = events.wait_for(EventKind.CONFIRM)
    runtime.answer_confirmation(request.data.request_id, False)

    events.wait_for(EventKind.REPLY_END)
    assert tool.calls == []
    # Model dostaje odmowę jako zwykły wynik i kończy turę zdaniem dla człowieka.
    assert "Nie mogę" in events.answer()


def test_zamkniecie_okna_w_trakcie_pytania_o_zgode_to_odmowa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brak odpowiedzi nie może znaczyć „wykonaj" ani zawiesić zamykania."""
    settings = make_settings(tools_enabled=True)
    tool = make_fake_tool(name="test.pisz", risk=RiskLevel.HIGH)
    install_router(monkeypatch, tool, settings=settings)

    client = FakeClient(
        [
            LLMStep(tool_calls=[{"function": {"name": "test.pisz", "arguments": {}}}]),
            LLMStep(chunks=("Trudno.",)),
        ]
    )
    runtime, events = start_runtime(client, settings=settings)
    try:
        runtime.send("wykonaj to zadanie")
        events.wait_for(EventKind.CONFIRM)
        start = time.perf_counter()
        runtime.close(timeout=8.0)
        elapsed = time.perf_counter() - start
    finally:
        runtime.close(timeout=5.0)

    assert tool.calls == []
    assert elapsed < 8.0, "zamykanie okna czekało na odpowiedź, której nikt nie da"


def test_potwierdzenie_niesie_opis_zbudowany_przez_narzedzie(
    runtime_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treść pytania pochodzi z narzędzia (Faza 7), nie z odpowiedzi modelu."""
    settings = make_settings(tools_enabled=True)
    tool = make_fake_tool(name="test.pisz", risk=RiskLevel.HIGH)
    install_router(monkeypatch, tool, settings=settings)

    client = FakeClient(
        [
            LLMStep(
                chunks=("nieszkodliwa operacja porządkowa",),
                tool_calls=[{"function": {"name": "test.pisz", "arguments": {}}}],
            ),
            LLMStep(chunks=("gotowe",)),
        ]
    )
    runtime, events = runtime_factory(client, settings=settings)

    runtime.send("zrób to")
    request = events.wait_for(EventKind.CONFIRM)

    assert request.data.tool == "test.pisz"
    assert "nieszkodliwa operacja" not in request.data.summary
    runtime.answer_confirmation(request.data.request_id, False)


# --------------------------------------------------------------------------- #
# Model, stan i ustawienia
# --------------------------------------------------------------------------- #


def test_zmiana_modelu_dotyczy_sesji_i_nie_pisze_do_pliku_ustawien(
    runtime_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model to ustawienie infrastruktury — okno mówi to wprost i nic nie zapisuje."""
    import config

    settings_file = tmp_path / "user_settings.json"
    settings_file.write_text(json.dumps({"assistant_name": "Miku"}), encoding="utf-8")
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(config, "_user_settings_cache", None, raising=False)

    runtime, events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))
    before = settings_file.read_bytes()

    runtime.set_model("inny-model")
    events.wait_for(EventKind.MESSAGE, where=lambda item: "inny-model" in item.text)
    status = events.wait_for(
        EventKind.STATUS,
        where=lambda item: item.snapshot is not None and item.snapshot.model == "inny-model",
    )

    assert status.snapshot is not None
    assert ".env" in events.texts(EventKind.MESSAGE)[-1]
    assert settings_file.read_bytes() == before


def test_lista_modeli_trafia_do_okna(runtime_factory: Any) -> None:
    runtime, events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))

    runtime.list_models()
    event = events.wait_for(EventKind.MODELS)

    assert "model-a" in (event.data or [])


def test_migawka_stanu_opisuje_uslugi(runtime_factory: Any) -> None:
    runtime, events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))

    runtime.refresh()
    event = events.wait_for(
        EventKind.STATUS,
        where=lambda item: item.snapshot is not None and bool(item.snapshot.services),
    )
    snapshot = event.snapshot
    assert snapshot is not None
    keys = {item.key for item in snapshot.services}

    assert {"mic", "speech", "tools"} <= keys
    assert snapshot.model


def test_przeladowanie_ustawien_wysyla_je_do_okna(
    runtime_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Po zapisie z panelu okno dostaje nowe imię i kolor — bez restartu."""
    import config

    settings_file = tmp_path / "user_settings.json"
    settings_file.write_text(
        json.dumps({"assistant_name": "Aiko", "ui_accent_color": "#FF6600"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", settings_file)
    monkeypatch.setattr(config, "_user_settings_cache", None, raising=False)
    monkeypatch.setattr(config, "_user_settings_mtime", None, raising=False)

    runtime, events = runtime_factory(FakeClient([LLMStep(chunks=("ok",))]))
    runtime.reload_settings()

    event = events.wait_for(EventKind.SETTINGS)
    assert event.text == "Aiko"
    assert event.detail == "#FF6600"


def test_zamkniecie_zwalnia_zasoby(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([LLMStep(chunks=("ok",))])
    runtime, events = start_runtime(client)

    runtime.send("cześć")
    events.wait_for(EventKind.REPLY_END)
    runtime.close(timeout=8.0)

    events.wait_for(EventKind.CLOSED, timeout=2.0)
    assert client.closed, "klient modelu nie został zamknięty"
