"""Testy widgetów GUI na prawdziwym oknie (Faza 10).

Te testy **wymagają Tk i sesji graficznej**, więc na serwerze bez pulpitu są
pomijane — reszta zestawu sprawdza tę samą logikę bez ekranu. Tutaj chodzi o
rzeczy, których nie da się sprawdzić inaczej: czy okno naprawdę bierze imię
asystenta z ustawień, czy zmiana koloru akcentu faktycznie przemalowuje widgety i
czy strumień odpowiedzi trafia do jednego dymka.

Wątek roboczy jest atrapą — testy nie uruchamiają modelu ani mikrofonu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import gui

_STATUS = gui.toolkit_status()
if not _STATUS.ok:
    # Świadomie NIE ``importorskip``: brak Tk to błąd importu w module ZALEŻNYM
    # (``customtkinter`` → ``tkinter`` → ``_tkinter``), a wtedy pytest go
    # przepuszcza dalej zamiast pominąć plik. Pytamy więc wprost tej samej
    # funkcji, której używa ``python main.py --gui``.
    pytest.skip(
        f"GUI niedostępne na tej maszynie: {_STATUS.detail}", allow_module_level=True
    )

import customtkinter as ctk  # noqa: E402 - dopiero po sprawdzeniu dostępności Tk

from config import Settings, UserSettings  # noqa: E402 - po sprawdzeniu dostępności Tk
from gui.chat import ChatView  # noqa: E402
from gui.state import ChatRole, ListeningState  # noqa: E402
from gui.status import StatusPanel  # noqa: E402
from gui.theme import Theme  # noqa: E402
from i18n import t  # noqa: E402


class StubRuntime:
    """Wątek roboczy-atrapa: notuje polecenia, niczego nie uruchamia."""

    def __init__(self, settings: Any, *, publish: Any, report: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.publish = publish
        self.commands: list[tuple[str, Any]] = []
        self.is_listening = False

    def __getattr__(self, name: str) -> Any:
        def record(*args: Any, **kwargs: Any) -> None:
            self.commands.append((name, args))

        return record

    def start(self) -> None:
        self.commands.append(("start", ()))

    def close(self, **kwargs: Any) -> None:
        self.commands.append(("close", ()))

    def toggle_listening(self) -> bool:
        self.is_listening = not self.is_listening
        return self.is_listening


@pytest.fixture
def user_settings_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import config

    target = tmp_path / "user_settings.json"
    target.write_text(
        json.dumps({"assistant_name": "Aiko", "ui_accent_color": "#FF6600"}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "USER_SETTINGS_FILE", target)
    monkeypatch.setattr(config, "_user_settings_cache", None, raising=False)
    monkeypatch.setattr(config, "_user_settings_mtime", None, raising=False)
    return target


@pytest.fixture
def window(user_settings_file: Path) -> Any:
    from gui.app import AssistantWindow

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        memory_enabled=False,
        embeddings_enabled=False,
        tools_enabled=False,
        tts_enabled=False,
        mic_enabled=False,
        gui_theme_mode="dark",
    )
    app = AssistantWindow(settings, speech_enabled=False, runtime_factory=StubRuntime)
    app.withdraw()  # test nie ma prawa migać oknem na ekranie użytkownika
    app.update_idletasks()
    try:
        yield app
    finally:
        app._closing = True
        app.destroy()


@pytest.fixture
def root() -> Any:
    app = ctk.CTk()
    app.withdraw()
    try:
        yield app
    finally:
        app.destroy()


# --------------------------------------------------------------------------- #
# Kolizje nazw z wnętrzem toolkitu
# --------------------------------------------------------------------------- #


def test_zaden_widget_nie_przykrywa_pola_toolkitu(window: Any, root: Any) -> None:
    """Własne pole o nazwie zajętej przez tkintera psuje widget w losowym miejscu.

    Regresja z prawdziwego uruchomienia: ``self._options`` przykryło
    ``tkinter.Misc._options()``, którego używa ``configure()`` — okno wywalało się
    przy pierwszej zmianie koloru. To samo dotyczy ``_draw`` z CustomTkintera.
    Test sprawdza WSZYSTKIE widgety naraz, bo takiej pomyłki nie widać w kodzie.
    """
    from gui.chat import ChatView
    from gui.settings_form import SettingsForm
    from gui.settings_panel import SettingsPanel
    from gui.status import ListeningIndicator, StatusPanel

    theme = Theme.build("#39C5BB")
    widgets = [
        window,
        window._chat,
        window._status,
        window._status.indicator,
        ChatView(root, theme=theme),
        StatusPanel(root, theme=theme),
        ListeningIndicator(root, theme=theme),
        SettingsPanel(root, theme=theme, form=SettingsForm(UserSettings())),
    ]

    # Pola, które tkinter/CustomTkinter zakłada sam na KAŻDYM widgecie, poznajemy
    # po czystym egzemplarzu klasy bazowej — inaczej test zgłaszałby ich własne
    # wnętrze jako nasz błąd.
    wlasne_toolkitu = set(vars(ctk.CTkFrame(root))) | set(vars(ctk.CTkLabel(root)))

    kolizje: list[str] = []
    for widget in widgets:
        base = type(widget).__mro__[1]
        for name in vars(widget):
            if not name.startswith("_") or name in wlasne_toolkitu:
                continue
            if hasattr(base, name):
                kolizje.append(f"{type(widget).__name__}.{name}")

    assert not kolizje, f"pola przykrywające wnętrze toolkitu: {sorted(set(kolizje))}"


# --------------------------------------------------------------------------- #
# Imię asystenta zamiast zaszytej nazwy
# --------------------------------------------------------------------------- #


def test_tytul_okna_i_naglowek_pokazuja_imie_z_ustawien(window: Any) -> None:
    assert "Aiko" in window.title()
    assert window._header.cget("text") == "Aiko"
    # Nigdzie nie ma zaszytego „Miku" — imię pochodzi wyłącznie z pliku ustawień.
    assert "Miku" not in window.title()


def test_zmiana_imienia_dziala_bez_restartu(window: Any) -> None:
    window._apply_user_settings(UserSettings(assistant_name="Zosia", ui_accent_color="#FF6600"))

    assert "Zosia" in window.title()
    assert window._header.cget("text") == "Zosia"


def test_podpis_wiadomosci_asystenta_zmienia_sie_razem_z_imieniem(window: Any) -> None:
    window._chat.add(ChatRole.ASSISTANT, "cześć")
    window._apply_user_settings(UserSettings(assistant_name="Zosia", ui_accent_color="#FF6600"))

    bubble = next(iter(window._chat._bubbles.values()))
    assert "Zosia" in bubble._caption.cget("text")


# --------------------------------------------------------------------------- #
# Kolor akcentu
# --------------------------------------------------------------------------- #


def test_okno_startuje_w_kolorze_z_ustawien(window: Any) -> None:
    assert window._theme.source_accent == "#ff6600"
    assert window._send_button.cget("fg_color") == window._theme.palette.accent


def test_zmiana_akcentu_przemalowuje_istniejace_babelki(window: Any) -> None:
    window._chat.add(ChatRole.USER, "pytanie")
    bubble = next(iter(window._chat._bubbles.values()))
    before = bubble.cget("fg_color")

    window._apply_user_settings(UserSettings(assistant_name="Aiko", ui_accent_color="#8844FF"))

    assert bubble.cget("fg_color") != before
    assert bubble.cget("fg_color") == window._theme.palette.user_bubble
    assert window._send_button.cget("fg_color") == window._theme.palette.accent


def test_podglad_koloru_dziala_przed_zapisem(window: Any) -> None:
    window._preview_accent("#00AAFF")
    assert window._theme.source_accent == "#00aaff"


# --------------------------------------------------------------------------- #
# Rozmowa
# --------------------------------------------------------------------------- #


def test_strumien_odpowiedzi_rysuje_jeden_dymek(root: Any) -> None:
    chat = ChatView(root, theme=Theme.build("#39C5BB"), assistant_name="Aiko")
    chat.pack(fill="both", expand=True)

    chat.start_reply()
    chat.append_chunk("Pierwsze ")
    chat.append_chunk("zdanie.")
    chat.finish_reply()
    root.update_idletasks()

    assert len(chat._bubbles) == 1
    bubble = next(iter(chat._bubbles.values()))
    assert bubble._body.cget("text") == "Pierwsze zdanie."


def test_pusta_odpowiedz_nie_zostawia_dymka(root: Any) -> None:
    chat = ChatView(root, theme=Theme.build("#39C5BB"))
    chat.pack(fill="both", expand=True)

    chat.start_reply()
    chat.finish_reply()
    root.update_idletasks()

    assert chat._bubbles == {}


def test_czyszczenie_rozmowy_usuwa_widgety(root: Any) -> None:
    chat = ChatView(root, theme=Theme.build("#39C5BB"))
    chat.pack(fill="both", expand=True)
    chat.add(ChatRole.USER, "raz")
    chat.add(ChatRole.ASSISTANT, "dwa")

    chat.clear()
    root.update_idletasks()

    assert chat._bubbles == {} and len(chat.log) == 0


# --------------------------------------------------------------------------- #
# Panel stanu i wskaźnik
# --------------------------------------------------------------------------- #


def test_wskaznik_pulsuje_tylko_w_trakcie_pracy(root: Any) -> None:
    theme = Theme.build("#39C5BB")
    panel = StatusPanel(root, theme=theme)
    panel.pack(fill="both", expand=True)

    panel.indicator.set_state(ListeningState.LISTENING)
    assert panel.indicator._job is not None

    panel.indicator.set_state(ListeningState.IDLE)
    root.update_idletasks()
    assert panel.indicator._job is None


def test_panel_stanu_pokazuje_migawke(root: Any) -> None:
    from gui.state import ServiceState, StatusSnapshot

    panel = StatusPanel(root, theme=Theme.build("#39C5BB"))
    panel.pack(fill="both", expand=True)

    snapshot = StatusSnapshot(assistant_name="Aiko", model="qwen2.5", host="http://x").with_service(
        "mic", ServiceState.OK, "wbudowany"
    )
    panel.update_snapshot(snapshot)
    root.update_idletasks()

    assert "Aiko" in panel._title.cget("text")
    assert "qwen2.5" in panel._model.cget("text")
    assert panel._rows["mic"]._detail.cget("text") == "wbudowany"


def test_przelacznik_mowy_pokazuje_stan_faktyczny(window: Any) -> None:
    """Na maszynie bez głosu przełącznik nie może udawać, że mowa działa."""
    from gui.state import ServiceState, StatusSnapshot

    window._on_status(
        StatusSnapshot().with_service("speech", ServiceState.OFF, "brak głosu Pipera")
    )
    assert window._speech_switch.get() == 0
    # Etykieta chowa się w trybie zwartym (wąskie okno), więc sprawdzamy stan,
    # który układ tylko *pokazuje*.
    assert window._speech_label == t("gui.speech_missing")

    window._on_status(StatusSnapshot().with_service("speech", ServiceState.OK, "piper"))
    assert window._speech_switch.get() == 1


def test_ekran_ustawien_przykrywa_pole_wpisywania(window: Any) -> None:
    """W trakcie ustawień nie ma sensu widzieć pola „napisz wiadomość”."""
    window._open_settings()
    info = window._settings_panel.grid_info()

    assert int(info["row"]) == 1 and int(info["rowspan"]) == 2


# --------------------------------------------------------------------------- #
# Ekran ustawień w oknie
# --------------------------------------------------------------------------- #


def test_ekran_ustawien_otwiera_sie_i_zamyka(window: Any) -> None:
    window._open_settings()
    assert window._settings_panel is not None

    window._close_overlays()
    assert window._settings_panel is None


def test_zapis_z_panelu_stosuje_kolor_i_imie_od_razu(
    window: Any, user_settings_file: Path
) -> None:
    window._open_settings()
    panel = window._settings_panel
    panel._widgets["assistant_name"].set("Zosia")
    panel._widgets["ui_accent_color"].set("#8844FF")

    panel._handle_save()

    saved = json.loads(user_settings_file.read_text(encoding="utf-8"))
    assert saved["assistant_name"] == "Zosia"
    assert "Zosia" in window.title()
    assert window._theme.source_accent == "#8844ff"


def test_przycisk_przegladaj_uzywa_systemowego_okna(
    window: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ścieżkę RVC wybiera się plikiem, nie z klawiatury."""
    from tkinter import filedialog

    chosen = tmp_path / "glos.pth"
    chosen.write_bytes(b"x")
    seen: dict[str, Any] = {}

    def fake_dialog(**kwargs: Any) -> str:
        seen.update(kwargs)
        return str(chosen)

    monkeypatch.setattr(filedialog, "askopenfilename", fake_dialog)
    window._open_settings()
    window._settings_panel._browse("rvc.model_path")

    assert seen["title"] == t(
        "settings.dialog_title", label=t("settings.field.rvc_model_path")
    )
    assert (t("settings.filter.rvc_model"), "*.pth") in seen["filetypes"]
    assert window._settings_panel._widgets["rvc.model_path"].get() == str(chosen.resolve())


# --------------------------------------------------------------------------- #
# Pytanie o zgodę
# --------------------------------------------------------------------------- #


def test_pytanie_o_zgode_dla_critical_wymaga_pelnej_frazy(window: Any) -> None:
    from security.confirm import ConfirmationRequest
    from security.risk import RiskLevel

    request = ConfirmationRequest.build(
        tool="shell.run",
        risk=RiskLevel.CRITICAL,
        summary="uruchomić polecenie powłoki",
        language="pl",
    )
    answers: list[bool] = []
    window._show_confirm(request)
    window._confirm_box._on_answer = lambda _request, approved: answers.append(approved)

    window._confirm_box._phrase.insert(0, "tak")
    window._confirm_box._approve_clicked()
    assert answers == [False]  # samo „tak" nie wystarcza dla CRITICAL

    window._confirm_box._phrase.delete(0, "end")
    window._confirm_box._phrase.insert(0, "tak, potwierdzam")
    window._confirm_box._approve_clicked()
    assert answers == [False, True]


def test_escape_przy_pytaniu_o_zgode_znaczy_odmowa(window: Any) -> None:
    from security.confirm import ConfirmationRequest
    from security.risk import RiskLevel

    request = ConfirmationRequest.build(
        tool="fs.delete", risk=RiskLevel.HIGH, summary="usunąć plik", language="pl"
    )
    window._show_confirm(request)
    window._close_overlays()

    commands = dict(
        (name, args) for name, args in window._runtime.commands if name == "answer_confirmation"
    )
    assert commands["answer_confirmation"] == (request.request_id, False)
    assert window._confirm_box is None
