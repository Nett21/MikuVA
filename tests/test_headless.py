"""Tryb bezobsługowy (``python main.py --headless``): usługa bez klawiatury i okna.

Ten tryb istnieje po to, żeby asystenta dało się uruchomić z ``systemd --user``
i z Harmonogramu zadań Windows. Testy pilnują tego, co w usłudze łatwo zepsuć,
a czego nie widać w normalnym uruchomieniu:

* **nigdy nie wołamy ``input()``** — pod usługą ``stdin`` jest zamknięty, a
  ``EOFError`` w pętli oznacza proces kręcący się na 100% procesora,
* **brak mikrofonu kończy się kodem wyjścia**, a nie cichym czekaniem: usługa
  bez wejścia głosowego nie ma jak przyjąć polecenia,
* **potwierdzenia są odrzucane** — nie ma komu zadać pytania, więc HIGH i
  CRITICAL nie wykonują się nigdy,
* **SIGTERM zamyka pracę czysto** — inaczej każde ``systemctl --user stop``
  zostawiałoby stack trace w dzienniku i niezamkniętą bazę.

Cała warstwa sprzętowa (mikrofon, Whisper, Piper) i Ollama są atrapami — test
przechodzi na maszynie bez karty dźwiękowej i bez GPU.
"""

from __future__ import annotations

import signal
import threading
from typing import Any

import pytest

import main as main_module
from config import Settings
from security.confirm import AutoDenyBroker, ConfirmationRequest
from security.risk import RiskLevel

# --------------------------------------------------------------------------- #
# Atrapy
# --------------------------------------------------------------------------- #


class FakeVoiceInput:
    """Mikrofon oddający zaplanowane wypowiedzi; potem zwraca ``None`` (cisza)."""

    def __init__(self, phrases: list[str | None], *, can_enable: bool = True) -> None:
        self._phrases = list(phrases)
        self._can_enable = can_enable
        self.enabled = False
        self.closed = False
        self.enable_calls = 0
        self.listen_calls = 0

    def enable(self) -> bool:
        self.enable_calls += 1
        self.enabled = self._can_enable
        return self._can_enable

    def disable(self) -> None:
        self.enabled = False

    def close(self) -> None:
        self.closed = True
        self.enabled = False

    def listen(self, *, timeout_s: float | None = None, quiet: bool = False) -> str | None:
        self.listen_calls += 1
        if self._phrases:
            return self._phrases.pop(0)
        return None


class FakeVoiceOutput:
    def __init__(self, *, can_enable: bool = True) -> None:
        self._can_enable = can_enable
        self.enabled = False
        self.closed = False
        self.spoken: list[str] = []
        self._buffer = ""

    @property
    def status_text(self) -> str:
        return "atrapa"

    @property
    def is_unavailable(self) -> bool:
        return not self._can_enable

    def enable(self, *, quiet: bool = False) -> bool:
        self.enabled = self._can_enable
        return self._can_enable

    def begin(self, language: str | None = None) -> None:
        self._buffer = ""

    def feed(self, text: str) -> None:
        self._buffer += text

    def end(self) -> None:
        if self._buffer:
            self.spoken.append(self._buffer)
        self._buffer = ""

    def cancel(self) -> None:
        self._buffer = ""

    def close(self) -> None:
        self.closed = True
        self.enabled = False


class FakeClient:
    """Klient Ollamy: zawsze dostępny, zawsze odpowiada tym samym."""

    def __init__(self, settings: Settings, reply: str = "Zrobione.") -> None:
        self.settings = settings
        self.reply = reply
        self.closed = False
        self.turns = 0

    async def is_available(self) -> bool:
        return True

    async def stream_chat(self, messages: Any, **kwargs: Any) -> Any:
        self.turns += 1
        collect = kwargs.get("collect")
        if collect is not None:
            collect.content += self.reply
        yield self.reply

    async def chat(self, messages: Any, **kwargs: Any) -> str:
        return self.reply

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def headless_settings(tmp_path: Any) -> Settings:
    return Settings(
        _env_file=None,
        piper_voices_dir=str(tmp_path / "voices"),
        database_path=str(tmp_path / "pamiec.sqlite3"),
        memory_enabled=False,
        embeddings_enabled=False,
        tools_enabled=False,
        plugins_enabled=False,
        headless_ollama_wait_s=0.0,
        headless_greeting=False,
        web_enabled=False,
    )


@pytest.fixture
def report() -> Any:
    from config import detect_dependencies

    return detect_dependencies(Settings(_env_file=None))


def wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    voice: FakeVoiceInput,
    speaker: FakeVoiceOutput,
    client: FakeClient | None = None,
) -> FakeClient | None:
    """Podmień wejście, wyjście i klienta modelu na atrapy."""
    monkeypatch.setattr(main_module, "VoiceInput", lambda settings: voice)
    monkeypatch.setattr(main_module, "VoiceOutput", lambda settings: speaker)
    if client is not None:
        import brain.llm

        monkeypatch.setattr(brain.llm, "OllamaClient", lambda settings: client)
    return client


# --------------------------------------------------------------------------- #
# Wejście: mikrofon jest warunkiem, nie dodatkiem
# --------------------------------------------------------------------------- #


def test_brak_mikrofonu_konczy_sie_kodem_bledu(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = FakeVoiceInput([], can_enable=False)
    speaker = FakeVoiceOutput()
    wire(monkeypatch, voice=voice, speaker=speaker)

    code = main_module.run_headless(headless_settings, report, stop=threading.Event())

    assert code == main_module.EXIT_MISSING_DEPENDENCIES
    assert voice.closed  # nawet nieudany start musi po sobie posprzątać


def test_nigdy_nie_wola_input(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W usłudze ``stdin`` nie istnieje — ``input()`` to nieskończona pętla EOFError."""

    def wybuch(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("run_headless nie ma prawa czytać z klawiatury")

    monkeypatch.setattr("builtins.input", wybuch)
    voice = FakeVoiceInput(["która godzina", None, None])
    speaker = FakeVoiceOutput()
    client = FakeClient(headless_settings)
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    code = main_module.run_headless(
        headless_settings, report, stop=threading.Event(), max_turns=3
    )
    assert code == main_module.EXIT_OK


def test_cisza_nie_konczy_uslugi(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``listen()`` zwracające ``None`` to cisza, a nie powód do wyjścia."""
    voice = FakeVoiceInput([None, None, None, "cześć"])
    speaker = FakeVoiceOutput()
    client = FakeClient(headless_settings)
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    main_module.run_headless(headless_settings, report, stop=threading.Event(), max_turns=4)

    assert voice.listen_calls == 4
    assert client.turns == 1  # tylko jedna prawdziwa wypowiedź


def test_pusta_wypowiedz_nie_idzie_do_modelu(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = FakeVoiceInput(["   ", "\t", "pytanie"])
    speaker = FakeVoiceOutput()
    client = FakeClient(headless_settings)
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    main_module.run_headless(headless_settings, report, stop=threading.Event(), max_turns=3)
    assert client.turns == 1


def test_odzyskiwanie_nasluchu_po_awarii_mikrofonu(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odłączony mikrofon ma prowadzić do ponownej próby, a nie do śmierci usługi."""
    settings = headless_settings.model_copy(update={"headless_retry_s": 1.0})
    voice = FakeVoiceInput(["pytanie"])
    speaker = FakeVoiceOutput()
    client = FakeClient(settings)
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    # Pierwszy obrót: nasłuch działa. Potem symulujemy awarię.
    original_listen = voice.listen
    stan = {"obrot": 0}

    def listen(*, timeout_s: float | None = None, quiet: bool = False) -> str | None:
        stan["obrot"] += 1
        if stan["obrot"] == 1:
            voice.enabled = False  # awaria po pierwszym obrocie
            return original_listen()
        return original_listen()

    monkeypatch.setattr(voice, "listen", listen)

    main_module.run_headless(settings, report, stop=threading.Event(), max_turns=3)
    assert voice.enable_calls >= 2  # próbował wstać


# --------------------------------------------------------------------------- #
# Bezpieczeństwo: nie ma kogo pytać o zgodę
# --------------------------------------------------------------------------- #


def test_broker_uslugi_zawsze_odmawia() -> None:
    """``AutoDenyBroker`` nie ma wariantu „auto-zgoda" i nie da się go włączyć."""
    import asyncio

    broker = AutoDenyBroker(reason="usługa w tle")
    request = ConfirmationRequest.build(
        tool="shell.run",
        risk=RiskLevel.HIGH,
        summary="usuń wszystko",
    )
    outcome = asyncio.run(broker.ask(request))
    assert outcome.approved is False
    assert broker.available is False


def test_usluga_buduje_router_z_brokerem_odmawiajacym(
    tmp_path: Any, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprawdzamy realny przepływ: router usługi dostaje broker, który odmawia."""
    settings = Settings(
        _env_file=None,
        piper_voices_dir=str(tmp_path / "voices"),
        database_path=str(tmp_path / "pamiec.sqlite3"),
        memory_enabled=False,
        embeddings_enabled=False,
        tools_enabled=True,
        plugins_enabled=False,
        headless_ollama_wait_s=0.0,
        web_enabled=False,
    )
    przechwycone: dict[str, Any] = {}
    prawdziwy = main_module._build_tools

    def szpieg(*args: Any, **kwargs: Any) -> Any:
        przechwycone["broker"] = kwargs.get("broker")
        return prawdziwy(*args, **kwargs)

    monkeypatch.setattr(main_module, "_build_tools", szpieg)
    voice = FakeVoiceInput([])
    speaker = FakeVoiceOutput()
    wire(monkeypatch, voice=voice, speaker=speaker, client=FakeClient(settings))

    # Zatrzymanie przed pierwszym obrotem: interesuje nas sama budowa routera.
    stop = threading.Event()
    stop.set()
    main_module.run_headless(settings, report, stop=stop)

    assert isinstance(przechwycone.get("broker"), AutoDenyBroker)
    assert przechwycone["broker"].available is False


# --------------------------------------------------------------------------- #
# Zamykanie
# --------------------------------------------------------------------------- #


def test_zdarzenie_stop_konczy_petle_i_sprzata(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop = threading.Event()
    stop.set()  # zatrzymanie przed pierwszym obrotem
    voice = FakeVoiceInput(["nigdy nie usłyszane"])
    speaker = FakeVoiceOutput()
    client = FakeClient(headless_settings)
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    code = main_module.run_headless(headless_settings, report, stop=stop)

    assert code == main_module.EXIT_OK
    assert voice.listen_calls == 0
    assert voice.closed and speaker.closed and client.closed


def test_stop_w_trakcie_nasluchu_nie_wysyla_wypowiedzi_do_modelu(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wypowiedź, która przyszła równolegle z SIGTERM-em, nie zaczyna nowej tury.

    Bez tego sprawdzenia ``systemctl stop`` potrafiłby uruchomić narzędzie już
    po decyzji o zamknięciu — i zostawić po sobie akcję, o której nikt nie wie.
    """
    stop = threading.Event()
    voice = FakeVoiceInput([])
    speaker = FakeVoiceOutput()
    client = FakeClient(headless_settings)

    def listen(*, timeout_s: float | None = None, quiet: bool = False) -> str | None:
        stop.set()  # sygnał przychodzi w trakcie czekania na mikrofon
        return "zrób coś"

    voice.listen = listen  # type: ignore[method-assign]
    wire(monkeypatch, voice=voice, speaker=speaker, client=client)

    main_module.run_headless(headless_settings, report, stop=stop)
    assert client.turns == 0


def test_uchwyty_sygnalow_sa_przywracane() -> None:
    """Po ``run_headless`` proces ma mieć swoje poprzednie uchwyty sygnałów.

    Trwałe podmienienie SIGINT-u zabiłoby Ctrl+C w każdym późniejszym trybie
    uruchomionym w tym samym procesie (i w teście, który poleci po tym).
    """
    przed = signal.getsignal(signal.SIGINT)
    stop = threading.Event()
    restore = main_module._install_stop_handlers(stop)
    assert signal.getsignal(signal.SIGINT) is not przed
    restore()
    assert signal.getsignal(signal.SIGINT) is przed


def test_sygnal_ustawia_zdarzenie_zamiast_rzucac() -> None:
    """Handler nie może rzucać ani blokować — ma tylko postawić flagę."""
    stop = threading.Event()
    restore = main_module._install_stop_handlers(stop)
    try:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(int(signal.SIGTERM), None)
        assert stop.is_set()
    finally:
        restore()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_flaga_headless_jest_w_parserze() -> None:
    args = main_module.build_parser().parse_args(["--headless"])
    assert args.headless is True


def test_headless_wyklucza_sie_z_gui(capsys: pytest.CaptureFixture[str]) -> None:
    code = main_module.main(["--headless", "--gui"])
    assert code == main_module.EXIT_CONFIG_ERROR
    assert "--headless" in capsys.readouterr().out


def test_headless_wyklucza_sie_z_terminalem(capsys: pytest.CaptureFixture[str]) -> None:
    code = main_module.main(["--headless", "--terminal"])
    assert code == main_module.EXIT_CONFIG_ERROR


def test_headless_odrzuca_no_voice(capsys: pytest.CaptureFixture[str]) -> None:
    """Usługa nie ma innego wejścia niż mikrofon — ``--no-voice`` byłby sprzecznością."""
    code = main_module.main(["--headless", "--no-voice"])
    assert code == main_module.EXIT_CONFIG_ERROR


def test_pomoc_opisuje_tryb() -> None:
    tekst = main_module.build_parser().format_help()
    assert "--headless" in tekst


# --------------------------------------------------------------------------- #
# Plik jednostki systemd
# --------------------------------------------------------------------------- #


def test_jednostka_systemd_istnieje_i_jest_uzytkownika() -> None:
    from config import PROJECT_ROOT

    unit = PROJECT_ROOT / "scripts" / "systemd" / "miku-assistant.service"
    tresc = unit.read_text(encoding="utf-8")

    assert "[Service]" in tresc and "[Install]" in tresc
    assert "--headless" in tresc
    # Usługa UŻYTKOWNIKA: cel instalacji to default.target, nie multi-user.target.
    assert "WantedBy=default.target" in tresc
    assert "multi-user.target" not in tresc
    # Nic, co wymagałoby roota.
    assert "User=" not in tresc
    assert "Group=" not in tresc
    # Zamykanie po SIGTERM jest zadeklarowane wprost.
    assert "KillSignal=SIGTERM" in tresc
    # Restart, ale z limitem — inaczej brak mikrofonu kręci procesorem bez końca.
    assert "Restart=on-failure" in tresc
    assert "StartLimitBurst=" in tresc


def test_jednostka_nie_zawiera_sciezek_konkretnej_maszyny() -> None:
    """Wzorzec ma być przenośny: ``%h`` zamiast katalogu domowego autora."""
    import re

    from config import PROJECT_ROOT

    tresc = (PROJECT_ROOT / "scripts" / "systemd" / "miku-assistant.service").read_text(
        encoding="utf-8"
    )
    assert re.search(r"/home/[a-z]", tresc, re.IGNORECASE) is None
    assert "%h" in tresc


def test_jednostka_nie_odcina_katalogu_domowego() -> None:
    """``ProtectHome=yes`` odciąłby bazę pamięci i narzędzia plikowe.

    To nie jest przeoczenie w hardeningu, tylko świadoma decyzja — i musi taka
    zostać, bo inaczej usługa startuje, ale nic nie działa.
    """
    from config import PROJECT_ROOT

    tresc = (PROJECT_ROOT / "scripts" / "systemd" / "miku-assistant.service").read_text(
        encoding="utf-8"
    )
    aktywne = [
        line.strip()
        for line in tresc.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "ProtectHome=no" in aktywne
    assert "ProtectHome=yes" not in aktywne


# --------------------------------------------------------------------------- #
# Pętla zdarzeń: jedna na całą usługę
# --------------------------------------------------------------------------- #


def test_czekanie_na_ollame_uzywa_petli_wywolujacego() -> None:
    """Sprawdzenie dostępności i rozmowa muszą iść na TEJ SAMEJ pętli.

    ``httpx.AsyncClient`` wiąże pulę połączeń z pętlą, na której je otworzył.
    Wcześniejsza wersja tworzyła własną pętlę na czas czekania i kasowała
    ustawienie pętli bieżącej w ``finally`` — pierwsze pytanie po starcie
    kończyło się wtedy błędem „Event loop is closed".
    """
    import asyncio

    widziane: list[asyncio.AbstractEventLoop] = []

    class Sonda:
        async def is_available(self) -> bool:
            widziane.append(asyncio.get_running_loop())
            return True

    loop = asyncio.new_event_loop()
    try:
        wynik = main_module._headless_wait_for_ollama(  # noqa: SLF001
            Sonda(),  # type: ignore[arg-type]
            threading.Event(),
            loop=loop,
            timeout_s=5.0,
        )
        assert wynik is True
        assert widziane == [loop]
        # Pętla wywołującego zostaje otwarta i użyteczna — to ona prowadzi rozmowę.
        assert not loop.is_closed()
        assert loop.run_until_complete(asyncio.sleep(0)) is None
    finally:
        loop.close()


def test_czekanie_konczy_sie_po_limicie_bez_serwera() -> None:
    """Brak Ollamy nie może zablokować startu usługi na zawsze."""
    import asyncio

    class Martwy:
        async def is_available(self) -> bool:
            return False

    loop = asyncio.new_event_loop()
    try:
        wynik = main_module._headless_wait_for_ollama(  # noqa: SLF001
            Martwy(),  # type: ignore[arg-type]
            threading.Event(),
            loop=loop,
            timeout_s=0.01,
        )
        assert wynik is False
    finally:
        loop.close()


def test_stop_przerywa_czekanie_na_ollame() -> None:
    """SIGTERM w trakcie czekania na model ma zamknąć usługę od razu."""
    import asyncio

    stop = threading.Event()

    class Wolny:
        async def is_available(self) -> bool:
            stop.set()
            return False

    loop = asyncio.new_event_loop()
    try:
        assert (
            main_module._headless_wait_for_ollama(  # noqa: SLF001
                Wolny(),  # type: ignore[arg-type]
                stop,
                loop=loop,
                timeout_s=600.0,
            )
            is False
        )
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# Reakcja na SIGTERM: okna nasłuchu muszą być krótkie
# --------------------------------------------------------------------------- #


def test_usluga_nasluchuje_w_krotkich_oknach(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Między oknami nasłuchu pętla sprawdza ``stop`` — i tylko dzięki temu
    ``systemctl --user stop`` kończy się w sekundach, a nie po ``VAD_LISTEN_TIMEOUT_S``.

    Bez tego limitu systemd (``TimeoutStopSec=30``) dobijałby proces SIGKILL-em
    w środku sprzątania: baza zostawałaby z otwartą transakcją, a mikrofon zajęty.
    """
    settings = headless_settings.model_copy(update={"headless_listen_slice_s": 3.0})
    uzyte: list[Any] = []

    class Sledzacy(FakeVoiceInput):
        def listen(self, *, timeout_s: float | None = None, quiet: bool = False) -> str | None:
            uzyte.append(timeout_s)
            return None

    voice = Sledzacy([])
    speaker = FakeVoiceOutput()
    wire(monkeypatch, voice=voice, speaker=speaker, client=FakeClient(settings))

    main_module.run_headless(settings, report, stop=threading.Event(), max_turns=2)

    assert uzyte == [3.0, 3.0]
    # Okno nasłuchu musi być wyraźnie krótsze niż TimeoutStopSec z jednostki.
    assert settings.headless_listen_slice_s < 30.0


def test_domyslne_okno_nasluchu_miesci_sie_w_timeoutstopsec() -> None:
    """Wartość domyślna i plik jednostki muszą się zgadzać — inaczej usługa ginie od SIGKILL."""
    import re

    from config import PROJECT_ROOT

    settings = Settings(_env_file=None)
    tresc = (PROJECT_ROOT / "scripts" / "systemd" / "miku-assistant.service").read_text(
        encoding="utf-8"
    )
    dopasowanie = re.search(r"^TimeoutStopSec=(\d+)", tresc, re.MULTILINE)
    assert dopasowanie is not None
    assert settings.headless_listen_slice_s < int(dopasowanie.group(1))


def test_usluga_nie_zasypuje_dziennika_komunikatem_o_ciszy(
    headless_settings: Settings, report: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nasłuch wraca co kilka sekund — komunikat „nic nie usłyszałem" byłby spamem."""
    uzyte: list[bool] = []

    class Sledzacy(FakeVoiceInput):
        def listen(self, *, timeout_s: float | None = None, quiet: bool = False) -> str | None:
            uzyte.append(quiet)
            return None

    voice = Sledzacy([])
    speaker = FakeVoiceOutput()
    wire(monkeypatch, voice=voice, speaker=speaker, client=FakeClient(headless_settings))

    main_module.run_headless(headless_settings, report, stop=threading.Event(), max_turns=2)
    assert uzyte == [True, True]
