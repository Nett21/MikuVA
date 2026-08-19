"""Okno asystenta w CustomTkinterze (Faza 10).

Układ: pasek górny (imię asystenta, model, przyciski), rozmowa w bąbelkach,
kolumna stanu po prawej, pole wpisywania u dołu. Ekran ustawień i pytanie o zgodę
na narzędzie pojawiają się **w tym samym oknie**, jako nakładki.

Podział pracy jest twardy i widoczny w kodzie:

* ten moduł działa wyłącznie w wątku interfejsu i **nigdy** nie woła modelu,
  mikrofonu ani dysku;
* :class:`gui.runtime.AssistantRuntime` robi całą resztę w swoim wątku;
* jedynym połączeniem jest kolejka zdarzeń opróżniana przez ``after()``.

Dlatego okno nie zamarza podczas generowania odpowiedzi: w tym czasie wątek
interfejsu ma do zrobienia tylko dopisywanie tekstu do dymka.

Nazwa asystenta i kolor akcentu pochodzą z ``config/user_settings.json`` i są
stosowane **od razu** po zapisaniu ustawień — bez restartu okna.
"""

from __future__ import annotations

import logging
import queue
from tkinter import font as tkfont
from typing import Any

import customtkinter as ctk

from config import APP_VERSION, Settings, UserSettings, get_user_settings
from gui.chat import ChatView
from gui.settings_form import ChoiceOptions, SaveResult, SettingsForm
from gui.settings_panel import SettingsPanel
from gui.state import ChatRole, ListeningState, ServiceState, StatusSnapshot
from gui.status import StatusPanel
from gui.theme import Metrics, Theme, ThemeMode, pick_font_family
from i18n import t
from security.confirm import ConfirmationRequest, interpret_answer

logger = logging.getLogger(__name__)

# Jak często wątek interfejsu zagląda do kolejki zdarzeń. 40 ms daje płynne
# dopisywanie tekstu (25 odświeżeń na sekundę) i zeruje obciążenie, gdy cicho.
POLL_MS: int = 40

# Ile fragmentów odpowiedzi łączymy w jedno odświeżenie. Model potrafi przysłać
# kilkadziesiąt fragmentów na sekundę — przemalowanie dymka przy każdym z nich
# byłoby czystą stratą, a tekst i tak pojawia się w tym samym momencie.
MAX_CHUNKS_PER_TICK: int = 120

# Rozmiar okna, gdy nic nie jest ustawione: mieści rozmowę i kolumnę stanu, ale
# NIGDY nie wychodzi za ekran — laptop z 1366x768 też musi je zmieścić.
_PREFERRED_SIZE: tuple[int, int] = (1180, 780)
_MIN_SIZE: tuple[int, int] = (560, 380)

# Progi układu. Okno MUSI działać przy dowolnym rozmiarze, bo na menedżerach
# kafelkowych (Hyprland, sway, i3) to kompozytor decyduje o rozmiarze i
# ``minsize()`` jest ignorowane — sprawdzone na żywo: żądane 700x700 dostało
# 820x700, a przy skalowaniu 1.4 okno urosło ponad ekran i ucięło dolny pasek.
_SIDEBAR_BREAKPOINT: int = 900  # poniżej: kolumna stanu chowa się sama
_COMPACT_BREAKPOINT: int = 760  # poniżej: krótkie napisy na przyciskach
_TINY_BREAKPOINT: int = 620  # poniżej: znika podtytuł i wybór modelu


def resolve_theme_mode(settings: Settings) -> ThemeMode:
    """Jasny czy ciemny motyw.

    ``GUI_THEME_MODE=system`` oddaje decyzję systemowi — ale nie zakładamy, że
    system w ogóle ma czym odpowiedzieć: na wielu pulpitach Linuksa nie ma
    standardowego sposobu odczytania preferencji, a wtedy CustomTkinter zwraca
    „Light". Wątpliwość rozstrzygamy na ciemny motyw, bo asystent bywa otwarty
    obok terminala.
    """
    wanted = str(getattr(settings, "gui_theme_mode", "system") or "system").strip().lower()
    if wanted in ("dark", "ciemny"):
        return "dark"
    if wanted in ("light", "jasny"):
        return "light"
    try:
        ctk.set_appearance_mode("system")
        detected = str(ctk.get_appearance_mode() or "").strip().lower()
    except Exception:  # pragma: no cover - zależne od systemu
        detected = ""
    return "light" if detected == "light" else "dark"


class AssistantWindow(ctk.CTk):
    """Główne okno. Buduje motyw z ustawień użytkownika i pilnuje wątku interfejsu."""

    def __init__(
        self,
        settings: Settings,
        *,
        report: object | None = None,
        speech_enabled: bool = True,
        start_in_voice_mode: bool = False,
        runtime_factory: Any = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._report = report
        self._user_settings: UserSettings = get_user_settings()
        self._events: queue.Queue[Any] = queue.Queue()
        self._settings_panel: SettingsPanel | None = None
        self._confirm_box: _ConfirmBox | None = None
        self._choices = ChoiceOptions()
        self._closing = False
        # ``None`` = o widoczności kolumny stanu decyduje szerokość okna;
        # True/False = użytkownik zdecydował sam i to jego wybór obowiązuje.
        self._sidebar_visible: bool | None = None
        self._layout_width = 0
        self._speech_label = t("gui.speech")

        mode = resolve_theme_mode(settings)
        try:
            ctk.set_appearance_mode(mode)
        except Exception:  # pragma: no cover - zależne od systemu
            logger.debug("Nie udało się ustawić trybu jasności", exc_info=True)
        self._apply_scaling()

        self._theme = Theme.build(
            self._user_settings.ui_accent_color,
            mode=mode,
            font_family=self._pick_font(),
            metrics=Metrics(),
        )

        self.title(self._window_title())
        self.configure(fg_color=self._theme.palette.background)
        self._apply_geometry()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        factory = runtime_factory
        if factory is None:
            from gui.runtime import AssistantRuntime

            factory = AssistantRuntime
        self._runtime = factory(
            settings,
            publish=self._events.put,
            report=report,
            speech_enabled=speech_enabled,
        )

        self._build_layout()
        self._runtime.start()
        self._runtime.list_models()
        if start_in_voice_mode:
            self._runtime.start_listening()
            self._sync_listen_button()
        self.after(POLL_MS, self._drain_events)
        self.bind("<Escape>", lambda _event: self._close_overlays())
        self.bind("<Configure>", self._on_window_resize)

    # --- ustawienia okna ---------------------------------------------------- #

    def _window_title(self) -> str:
        """Tytuł okna z imieniem asystenta — nigdy zaszytej nazwy."""
        return t(
            "gui.window_title",
            name=self._user_settings.assistant_name,
            version=APP_VERSION,
        )

    def _apply_scaling(self) -> None:
        """Skalowanie interfejsu — ustalone RAZ, przy starcie.

        CustomTkinter co 100 ms sprawdza DPI okna i przy każdej zmianie
        przeskalowuje wszystkie widgety. Na Linuksie ta ścieżka zostawia część z
        nich **niepomalowanych** — użytkownik widzi czarne prostokąty w miejscu
        przycisków, dopóki nie najedzie myszką (zgłoszone z pulpitu, odtworzone
        u siebie: przy skali 1.25 znikał cały dolny pasek).

        Dlatego odczytujemy DPI jeden raz, ustawiamy skalowanie na sztywno i
        wyłączamy automat. Kto pracuje na dwóch monitorach o różnym DPI, ustawia
        ``GUI_SCALING`` pod ten, którego używa do asystenta.

        Skalowania OKNA nie ruszamy nigdy: CustomTkinter mnoży przez nie żądaną
        geometrię, a na menedżerze kafelkowym rozmiar należy do kompozytora —
        efektem było okno wyższe niż ekran z uciętym paskiem wpisywania.
        """
        configured = float(getattr(self._settings, "gui_scaling", 0.0) or 0.0)
        detected = 1.0
        if configured <= 0:
            try:
                detected = float(ctk.ScalingTracker.get_window_dpi_scaling(self)) or 1.0
            except Exception:  # pragma: no cover - zależne od systemu
                detected = 1.0
        try:
            ctk.deactivate_automatic_dpi_awareness()
            ctk.set_window_scaling(1.0)
            ctk.set_widget_scaling(configured if configured > 0 else detected)
        except Exception:  # pragma: no cover - zależne od systemu
            logger.debug("Nie udało się ustawić skalowania", exc_info=True)

    def _pick_font(self) -> str:
        """Krój pisma dostępny na tej maszynie (albo nic — wtedy decyduje Tk)."""
        preferred = str(getattr(self._settings, "gui_font_family", "") or "")
        try:
            families = tkfont.families(self)
        except Exception:  # pragma: no cover - zależne od Tk
            families = ()
        return pick_font_family(families, preferred=preferred)

    def _apply_geometry(self) -> None:
        """Rozmiar okna dopasowany do ekranu, z ustawieniem jako nadpisaniem."""
        width, height = _PREFERRED_SIZE
        raw = str(getattr(self._settings, "gui_window_size", "") or "").strip().lower()
        if "x" in raw:
            first, _, second = raw.partition("x")
            if first.strip().isdigit() and second.strip().isdigit():
                width, height = int(first), int(second)
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            width = max(_MIN_SIZE[0], min(width, screen_width - 60))
            height = max(_MIN_SIZE[1], min(height, screen_height - 100))
        except Exception:  # pragma: no cover - zależne od systemu okien
            pass
        self.geometry(f"{width}x{height}")
        self.minsize(*_MIN_SIZE)

    # --- układ --------------------------------------------------------------- #

    def _build_layout(self) -> None:
        theme = self._theme
        pad = theme.metrics.pad

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        # Tylko wiersz rozmowy rośnie. Pasek górny i dolny mają zostać widoczne
        # ZAWSZE — to one dają dostęp do pisania i do przerwania odpowiedzi.
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1, minsize=60)
        self.grid_rowconfigure(2, weight=0)

        # --- pasek górny --- #
        self._top = ctk.CTkFrame(self, fg_color=theme.palette.surface, corner_radius=0)
        self._top.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._top.grid_columnconfigure(1, weight=1)

        self._header = ctk.CTkLabel(
            self._top,
            text=self._user_settings.assistant_name,
            anchor="w",
            font=ctk.CTkFont(
                family=theme.font_family or None,
                size=theme.metrics.font_size_title + 3,
                weight="bold",
            ),
            text_color=theme.palette.accent,
        )
        self._header.grid(row=0, column=0, sticky="w", padx=(pad, 6), pady=pad)

        self._subtitle = ctk.CTkLabel(
            self._top,
            text=t("gui.subtitle"),
            anchor="w",
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=theme.palette.text_faint,
        )
        self._subtitle.grid(row=0, column=1, sticky="w", pady=pad)

        self._model_menu = ctk.CTkOptionMenu(
            self._top,
            values=[self._settings.ollama_model],
            command=self._on_model_change,
            width=210,
            fg_color=theme.palette.surface_alt,
            button_color=theme.palette.accent,
            button_hover_color=theme.palette.accent_hover,
            text_color=theme.palette.text,
            dropdown_fg_color=theme.palette.surface,
            dropdown_hover_color=theme.palette.accent_soft,
            dropdown_text_color=theme.palette.text,
        )
        self._model_menu.set(self._settings.ollama_model)
        self._model_menu.grid(row=0, column=2, padx=6, pady=pad)

        self._speech_switch = ctk.CTkSwitch(
            self._top,
            text=t("gui.speech"),
            command=self._on_speech_toggle,
            progress_color=theme.palette.accent,
            button_color=theme.palette.surface_alt,
            button_hover_color=theme.palette.accent_hover,
            text_color=theme.palette.text,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
        )
        self._speech_switch.select()
        self._speech_switch.grid(row=0, column=3, padx=6, pady=pad)

        self._sidebar_button = ctk.CTkButton(
            self._top,
            text=t("gui.status_hide"),
            width=110,
            command=self._toggle_sidebar,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        self._sidebar_button.grid(row=0, column=4, padx=6, pady=pad)

        self._new_button = ctk.CTkButton(
            self._top,
            text=t("gui.new_conversation"),
            width=130,
            command=self._on_new_conversation,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        self._new_button.grid(row=0, column=5, padx=6, pady=pad)

        self._settings_button = ctk.CTkButton(
            self._top,
            text=t("gui.settings"),
            width=120,
            command=self._open_settings,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        self._settings_button.grid(row=0, column=6, padx=(6, pad), pady=pad)

        # --- rozmowa i stan --- #
        self._chat = ChatView(
            self,
            theme=theme,
            assistant_name=self._user_settings.assistant_name,
        )
        self._chat.grid(row=1, column=0, sticky="nsew")

        self._status = StatusPanel(self, theme=theme)
        self._status.grid(row=1, column=1, sticky="nsew")

        # Bez tego przewijana rozmowa ŻĄDA tyle wysokości, ile ma treści, a Tk
        # przydziela ją kosztem ostatniego wiersza — i dolny pasek znika poza
        # oknem. Ustalony (mały) rozmiar własny + waga wiersza = rozmowa zajmuje
        # to, co zostanie, a pole wpisywania zostaje na ekranie.
        for panel in (self._chat, self._status):
            panel.configure(width=240, height=120)
            panel.grid_propagate(False)

        # Paski górny i dolny zostają przy propagacji rozmiaru: zamrożenie ich
        # (grid_propagate(False)) sprawiało, że po zmianie skalowania malowały
        # się tylko do swojej ZAPAMIĘTANEJ szerokości — obok pola wpisywania
        # widać było tło rozmowy zamiast przycisków (sprawdzone pikselami zrzutu).
        # Minimalną szerokość okna trzymają w ryzach progi układu: przy wąskim
        # oknie napisy skracają się, a kolumna stanu znika.

        # --- pole wpisywania --- #
        self._bottom = ctk.CTkFrame(self, fg_color=theme.palette.surface, corner_radius=0)
        self._bottom.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._bottom.grid_columnconfigure(0, weight=1)

        self._entry = ctk.CTkEntry(
            self._bottom,
            placeholder_text=t("gui.input_placeholder"),
            fg_color=theme.palette.surface_alt,
            border_color=theme.palette.border,
            text_color=theme.palette.text,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            height=38,
        )
        self._entry.grid(row=0, column=0, sticky="ew", padx=(pad, 6), pady=pad)
        self._entry.bind("<Return>", self._on_send)
        self._entry.focus_set()

        self._listen_button = ctk.CTkButton(
            self._bottom,
            text=t("gui.listen"),
            width=120,
            command=self._on_listen_toggle,
            fg_color=theme.palette.accent_soft,
            hover_color=theme.palette.accent_hover,
            text_color=theme.palette.text,
        )
        self._listen_button.grid(row=0, column=1, padx=6, pady=pad)

        self._send_button = ctk.CTkButton(
            self._bottom,
            text=t("gui.send"),
            width=110,
            command=self._on_send,
            fg_color=theme.palette.accent,
            hover_color=theme.palette.accent_hover,
            text_color=theme.palette.accent_text,
        )
        self._send_button.grid(row=0, column=2, padx=6, pady=pad)

        self._stop_button = ctk.CTkButton(
            self._bottom,
            text=t("gui.interrupt"),
            width=110,
            command=self._on_cancel,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text_muted,
        )
        self._stop_button.grid(row=0, column=3, padx=(6, pad), pady=pad)

        # Pole wpisywania ma pierwszeństwo przy zwężaniu — przyciski są przy nim
        # wąskie, ale zostają klikalne.
        self._bottom.grid_columnconfigure(0, weight=1, minsize=120)
        # Pierwsze dopasowanie do rozmiaru, jaki faktycznie dostaniemy od WM.
        self.after_idle(lambda: self._apply_layout(max(self.winfo_width(), 1)))

    # --- układ zależny od rozmiaru ------------------------------------------- #

    def _on_window_resize(self, event: Any) -> None:
        """Dopasuj układ do rozmiaru, jaki DAŁ nam menedżer okien.

        Na kafelkowych menedżerach nie decydujemy o rozmiarze okna — dostajemy
        go. Zamiast prosić o minimum (co i tak jest ignorowane), układ chowa
        rzeczy w kolejności ważności: najpierw kolumna stanu, potem długie
        napisy, na końcu podtytuł i wybór modelu. Pole wpisywania i przyciski
        rozmowy zostają zawsze.
        """
        if getattr(event, "widget", None) is not self:
            return  # zdarzenia z widgetów wewnątrz nas nie zmieniają układu okna
        width = int(getattr(event, "width", 0) or 0)
        if width <= 1 or width == self._layout_width:
            return
        self._layout_width = width
        self._apply_layout(width)

    def _widget_scaling(self) -> float:
        """Ile razy CustomTkinter powiększa widgety na tym ekranie.

        Progi układu MUSZĄ być liczone w pikselach logicznych: przy skalowaniu
        1.4 okno o 784 pikselach mieści tyle, co 560 przy skali 1.0 — a bez tego
        przeliczenia pasek górny zostawał w wersji pełnej i wychodził poza
        krawędź (odtworzone na żywo).
        """
        try:
            return float(ctk.ScalingTracker.get_widget_scaling(self)) or 1.0
        except Exception:  # pragma: no cover - zależne od wersji biblioteki
            return 1.0

    def _force_redraw(self, widget: Any = None) -> None:
        """Przerysuj całe drzewo widgetów.

        CustomTkinter rysuje każdy widget na własnym płótnie i po zmianie
        skalowania (np. po przeniesieniu okna na monitor o innym DPI) część z
        nich zostaje niepomalowana — użytkownik widzi czarny prostokąt do czasu,
        aż najedzie myszką i wymusi odrysowanie. Zdarzenia myszy nie są planem
        naprawczym, więc robimy to sami.

        ``_draw`` jest polem wewnętrznym biblioteki, dlatego sięgamy po nie
        defensywnie: gdy przyszła wersja je przemianuje, zostanie zachowanie
        sprzed poprawki, a nie wyjątek w oknie.
        """
        target = self if widget is None else widget
        draw = getattr(target, "_draw", None)
        if callable(draw):
            try:
                draw(no_color_updates=False)
            except Exception:  # pragma: no cover - zależne od wersji biblioteki
                logger.debug("Nie udało się przerysować %r", target, exc_info=True)
        try:
            children = target.winfo_children()
        except Exception:  # pragma: no cover - widget w trakcie niszczenia
            return
        for child in children:
            self._force_redraw(child)

    def _apply_layout(self, width: int) -> None:
        logical = width / self._widget_scaling()
        compact = logical < _COMPACT_BREAKPOINT
        tiny = logical < _TINY_BREAKPOINT

        self._sync_sidebar(width)

        # Krótkie napisy zamiast pełnych — przycisk ma zostać klikalny, a nie
        # zniknąć poza krawędź okna.
        self._new_button.configure(
            text=t("gui.new_short") if compact else t("gui.new_conversation"),
            width=44 if compact else 130,
        )
        self._settings_button.configure(
            text=t("gui.settings_short") if compact else t("gui.settings"),
            width=44 if compact else 120,
        )
        self._sidebar_button.configure(width=44 if compact else 110)
        self._speech_switch.configure(text="" if compact else self._speech_label)
        self._send_button.configure(width=64 if compact else 110)
        self._listen_button.configure(width=72 if compact else 120)
        self._stop_button.configure(width=64 if compact else 110)

        if tiny:
            self._subtitle.grid_remove()
            self._model_menu.grid_remove()
        else:
            self._subtitle.grid()
            self._model_menu.grid()
            self._model_menu.configure(width=140 if compact else 210)

        # Po zmianie układu (a zwłaszcza skalowania) część widgetów zostaje
        # niepomalowana — patrz :meth:`_force_redraw`. Dwa podejścia: zaraz po
        # przeliczeniu układu i chwilę później, bo CustomTkinter kończy własne
        # przeskalowanie dopiero w kolejnych obrotach pętli zdarzeń.
        self.after_idle(self._force_redraw)
        self.after(120, self._force_redraw)

    def _sync_sidebar(self, width: int | None = None) -> None:
        """Pokaż albo schowaj kolumnę stanu — automatycznie albo na życzenie."""
        available = self.winfo_width() if width is None else width
        wanted = self._sidebar_visible
        if wanted is None:  # brak decyzji użytkownika → decyduje szerokość
            wanted = available / self._widget_scaling() >= _SIDEBAR_BREAKPOINT
        if wanted:
            self._status.grid()
        else:
            self._status.grid_remove()
        self._sidebar_button.configure(
            text=t("gui.status_hide") if wanted else t("gui.status_show")
        )

    def _toggle_sidebar(self) -> None:
        current = self._status.winfo_ismapped()
        self._sidebar_visible = not current
        self._sync_sidebar()

    # --- pompa zdarzeń ------------------------------------------------------- #

    def _drain_events(self) -> None:
        """Przepisz zdarzenia z wątku roboczego na ekran.

        Wołane co :data:`POLL_MS` w wątku interfejsu — to JEDYNE miejsce, w którym
        stan z wątku roboczego trafia do widgetów. Fragmenty odpowiedzi są
        scalane, żeby jedno odświeżenie obsłużyło całą porcję tekstu.
        """
        from gui.runtime import EventKind

        pending_chunks: list[str] = []
        processed = 0
        try:
            while processed < MAX_CHUNKS_PER_TICK:
                try:
                    event = self._events.get_nowait()
                except queue.Empty:
                    break
                processed += 1
                if event.kind is EventKind.REPLY_CHUNK:
                    pending_chunks.append(event.text)
                    continue
                if pending_chunks:
                    self._chat.append_chunk("".join(pending_chunks))
                    pending_chunks = []
                self._handle_event(event)
        finally:
            if pending_chunks:
                self._chat.append_chunk("".join(pending_chunks))
            if not self._closing:
                self.after(POLL_MS, self._drain_events)

    def _handle_event(self, event: Any) -> None:
        from gui.runtime import EventKind

        if event.kind is EventKind.STATUS and event.snapshot is not None:
            self._on_status(event.snapshot)
        elif event.kind is EventKind.MESSAGE:
            self._chat.add(event.role or ChatRole.SYSTEM, event.text, detail=event.detail)
        elif event.kind is EventKind.READY:
            if isinstance(event.data, UserSettings):
                self._apply_user_settings(event.data)
            self._chat.add(ChatRole.ASSISTANT, event.text, detail=event.detail)
        elif event.kind is EventKind.REPLY_START:
            self._chat.start_reply()
        elif event.kind is EventKind.REPLY_END:
            self._chat.finish_reply()
        elif event.kind is EventKind.TOOL:
            self._chat.add(ChatRole.TOOL, event.text, detail=event.detail)
        elif event.kind is EventKind.ERROR:
            self._chat.add(ChatRole.ERROR, event.text, detail=event.detail)
        elif event.kind is EventKind.CONFIRM and isinstance(event.data, ConfirmationRequest):
            self._show_confirm(event.data)
        elif event.kind is EventKind.CONFIRM_CLOSED:
            self._close_confirm()
        elif event.kind is EventKind.MODELS:
            self._on_models(event.data, event.detail)
        elif event.kind is EventKind.VOICES:
            self._on_voices(event.data)
        elif event.kind is EventKind.SETTINGS and isinstance(event.data, UserSettings):
            self._apply_user_settings(event.data)
        elif event.kind is EventKind.CLOSED:
            self._closing = True

    def _on_status(self, snapshot: StatusSnapshot) -> None:
        self._status.update_snapshot(snapshot)
        if snapshot.model and snapshot.model != self._model_menu.get():
            self._model_menu.set(snapshot.model)
        self._sync_listen_button()
        busy = snapshot.listening in (ListeningState.THINKING, ListeningState.SPEAKING)
        self._stop_button.configure(state="normal" if busy else "disabled")

        # Przełącznik mowy pokazuje STAN FAKTYCZNY, a nie życzenie: na maszynie
        # bez głosu Pipera „Mowa: włączona" byłaby po prostu nieprawdą.
        speech = snapshot.service("speech")
        if speech is not None:
            if speech.state is ServiceState.OK:
                self._speech_switch.select()
            else:
                self._speech_switch.deselect()
            self._speech_label = (
                t("gui.speech")
                if speech.state is ServiceState.OK
                else t("gui.speech_missing")
            )
            if self._layout_width == 0 or (
                self._layout_width / self._widget_scaling() >= _COMPACT_BREAKPOINT
            ):
                self._speech_switch.configure(text=self._speech_label)

    def _on_models(self, models: Any, detail: str = "") -> None:
        names = [str(item) for item in (models or []) if str(item).strip()]
        if not names:
            return
        if self._settings.ollama_model not in names:
            names.insert(0, self._settings.ollama_model)
        self._model_menu.configure(values=names)
        self._model_menu.set(self._settings.ollama_model)
        if detail:
            self._chat.add(ChatRole.SYSTEM, detail)

    def _on_voices(self, data: Any) -> None:
        try:
            voices, engines = data
        except (TypeError, ValueError):  # pragma: no cover - obrona
            return
        self._choices = ChoiceOptions(
            piper_voices=tuple(str(item) for item in voices),
            tts_engines=tuple(str(item) for item in engines),
        )
        if self._settings_panel is not None:
            self._settings_panel.set_options(self._choices)

    # --- reakcje na kliknięcia ---------------------------------------------- #

    def _on_send(self, _event: Any = None) -> None:
        text = self._entry.get().strip()
        if not text:
            return
        self._entry.delete(0, "end")
        self._runtime.send(text)

    def _on_cancel(self) -> None:
        self._runtime.cancel()

    def _on_listen_toggle(self) -> None:
        self._runtime.toggle_listening()
        self._sync_listen_button()

    def _sync_listen_button(self) -> None:
        listening = bool(getattr(self._runtime, "is_listening", False))
        palette = self._theme.palette
        self._listen_button.configure(
            text=t("gui.stop_listening") if listening else t("gui.listen"),
            fg_color=palette.accent if listening else palette.accent_soft,
            text_color=palette.accent_text if listening else palette.text,
        )

    def _on_speech_toggle(self) -> None:
        self._runtime.set_muted(self._speech_switch.get() == 0)

    def _on_new_conversation(self) -> None:
        self._runtime.new_conversation()
        self._chat.clear()
        self._chat.add(ChatRole.SYSTEM, t("gui.new_conversation_notice"))

    def _on_model_change(self, name: str) -> None:
        self._runtime.set_model(name)

    # --- ekran ustawień ------------------------------------------------------ #

    def _open_settings(self) -> None:
        if self._settings_panel is not None:
            return
        form = SettingsForm(get_user_settings())
        self._settings_panel = SettingsPanel(
            self,
            theme=self._theme,
            form=form,
            options=self._choices,
            on_saved=self._on_settings_saved,
            on_preview_accent=self._preview_accent,
            on_close=self._close_settings,
            on_voice_test=self._runtime.say_sample,
        )
        # Nakładka zajmuje obszar rozmowy i kolumny stanu; pasek górny zostaje,
        # więc widać, w czyich ustawieniach się siedzi.
        # rowspan=2: ekran ustawień przykrywa też pasek wpisywania — inaczej
        # w trakcie zmiany ustawień widać pole „napisz wiadomość”, które do
        # niczego wtedy nie służy.
        self._settings_panel.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew")
        self._settings_panel.tkraise()

    def _close_settings(self) -> None:
        panel, self._settings_panel = self._settings_panel, None
        if panel is not None:
            panel.destroy()
        # Podglądowy akcent mógł się różnić od zapisanego — wracamy do prawdy.
        self._apply_user_settings(get_user_settings())
        self._entry.focus_set()

    def _on_settings_saved(self, result: SaveResult) -> None:
        """Zastosuj zapisane ustawienia od razu (imię, kolor, cechy charakteru)."""
        if result.settings is not None:
            self._apply_user_settings(result.settings)
        self._runtime.reload_settings()
        if result.needs_tts_reload:
            self._runtime.reload_speech()
        if result.changed:
            self._chat.add(ChatRole.SYSTEM, result.message())

    def _preview_accent(self, accent: str) -> None:
        """Podgląd koloru przed zapisem — ta sama droga co po zapisie."""
        self._set_theme(self._theme.with_accent(accent))

    def _apply_user_settings(self, user_settings: UserSettings) -> None:
        """Imię i kolor z pliku ustawień → tytuł okna, nagłówki i cała paleta."""
        self._user_settings = user_settings
        self.title(self._window_title())
        self._header.configure(text=user_settings.assistant_name)
        self._chat.set_assistant_name(user_settings.assistant_name)
        # Porównujemy z kolorem WPISANYM, nie narysowanym — patrz Theme.source_accent.
        if not self._theme.wants_accent(user_settings.ui_accent_color):
            self._set_theme(self._theme.with_accent(user_settings.ui_accent_color))

    def _set_theme(self, theme: Theme) -> None:
        """Przemaluj całe okno nową paletą — bez restartu i bez utraty rozmowy."""
        self._theme = theme
        palette = theme.palette
        self.configure(fg_color=palette.background)
        for frame in (self._top, self._bottom):
            frame.configure(fg_color=palette.surface)
        self._header.configure(text_color=palette.accent)
        self._subtitle.configure(text_color=palette.text_faint)
        self._model_menu.configure(
            fg_color=palette.surface_alt,
            button_color=palette.accent,
            button_hover_color=palette.accent_hover,
            text_color=palette.text,
            dropdown_fg_color=palette.surface,
            dropdown_hover_color=palette.accent_soft,
            dropdown_text_color=palette.text,
        )
        self._speech_switch.configure(
            progress_color=palette.accent,
            button_color=palette.surface_alt,
            button_hover_color=palette.accent_hover,
            text_color=palette.text,
        )
        for button in (self._new_button, self._settings_button, self._sidebar_button):
            button.configure(
                fg_color=palette.surface_alt,
                hover_color=palette.accent_soft,
                text_color=palette.text,
            )
        self._entry.configure(
            fg_color=palette.surface_alt,
            border_color=palette.border,
            text_color=palette.text,
        )
        self._send_button.configure(
            fg_color=palette.accent,
            hover_color=palette.accent_hover,
            text_color=palette.accent_text,
        )
        self._stop_button.configure(
            fg_color=palette.surface_alt,
            hover_color=palette.accent_soft,
            text_color=palette.text_muted,
        )
        self._sync_listen_button()
        self._chat.apply_theme(theme)
        self._status.apply_theme(theme)
        if self._settings_panel is not None:
            self._settings_panel.apply_theme(theme)
        if self._confirm_box is not None:
            self._confirm_box.apply_theme(theme)
        self.after_idle(self._force_redraw)

    # --- pytanie o zgodę ----------------------------------------------------- #

    def _show_confirm(self, request: ConfirmationRequest) -> None:
        self._close_confirm()
        self._confirm_box = _ConfirmBox(
            self,
            theme=self._theme,
            request=request,
            on_answer=self._answer_confirm,
        )
        self._confirm_box.grid(row=1, column=0, columnspan=2, rowspan=2, sticky="nsew")
        self._confirm_box.tkraise()

    def _answer_confirm(self, request: ConfirmationRequest, approved: bool) -> None:
        self._runtime.answer_confirmation(request.request_id, approved)
        self._close_confirm()

    def _close_confirm(self) -> None:
        box, self._confirm_box = self._confirm_box, None
        if box is not None:
            box.destroy()
        if self._settings_panel is not None:
            self._settings_panel.tkraise()

    def _close_overlays(self) -> None:
        """Escape: zamknij ustawienia. Pytania o zgodę zamyka tylko odpowiedź.

        Zgoda musi być świadoma, więc klawisz ucieczki nie może jej „przypadkiem"
        udzielić — ale nie może też zostawić routera czekającego bez końca.
        Dlatego Escape przy pytaniu o zgodę znaczy ODMOWA.
        """
        if self._confirm_box is not None:
            request = self._confirm_box.request
            self._answer_confirm(request, False)
            return
        if self._settings_panel is not None:
            self._close_settings()

    # --- zamykanie ----------------------------------------------------------- #

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._chat.add(ChatRole.SYSTEM, t("gui.closing"))
        self.update_idletasks()
        try:
            self._runtime.close()
        except Exception:  # pragma: no cover - zamykanie nie może rzucać
            logger.debug("Błąd przy zamykaniu wątku roboczego", exc_info=True)
        self.destroy()


class _ConfirmBox(ctk.CTkFrame):
    """Pytanie o zgodę na narzędzie HIGH/CRITICAL — nakładka w oknie.

    Treść pytania buduje NARZĘDZIE (Faza 7), nie model językowy, więc panel tylko
    ją wypisuje. Dla poziomu CRITICAL wymagane jest wpisanie pełnej frazy —
    sprawdza ją ta sama funkcja co w terminalu (:func:`interpret_answer`), żeby
    reguła zgody istniała w jednym miejscu.
    """

    def __init__(
        self,
        master: Any,
        *,
        theme: Theme,
        request: ConfirmationRequest,
        on_answer: Any,
    ) -> None:
        super().__init__(master, fg_color=theme.palette.background, corner_radius=0)
        self.request = request
        self._theme = theme
        self._on_answer = on_answer
        pad = theme.metrics.pad

        self._card = ctk.CTkFrame(
            self, fg_color=theme.palette.surface, corner_radius=theme.metrics.radius
        )
        self._card.grid(row=0, column=0, padx=pad * 2, pady=pad * 2, sticky="new")
        self._card.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._title = ctk.CTkLabel(
            self._card,
            text="\n".join(request.render_lines()[:1]),
            anchor="w",
            justify="left",
            wraplength=680,
            font=ctk.CTkFont(
                family=theme.font_family or None, size=theme.metrics.font_size + 1, weight="bold"
            ),
            text_color=theme.palette.text,
        )
        self._title.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, 4))

        body = "\n".join(request.render_lines()[1:])
        self._body = ctk.CTkLabel(
            self._card,
            text=body,
            anchor="w",
            justify="left",
            wraplength=680,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            text_color=theme.palette.text_muted,
        )
        self._body.grid(row=1, column=0, sticky="ew", padx=pad)

        self._phrase: ctk.CTkEntry | None = None
        row = 2
        if request.requires_phrase:
            self._phrase = ctk.CTkEntry(
                self._card,
                placeholder_text=request.prompt().strip(": "),
                fg_color=theme.palette.surface_alt,
                border_color=theme.palette.state_error,
                text_color=theme.palette.text,
            )
            self._phrase.grid(row=row, column=0, sticky="ew", padx=pad, pady=(pad, 0))
            row += 1

        buttons = ctk.CTkFrame(self._card, fg_color="transparent")
        buttons.grid(row=row, column=0, sticky="e", padx=pad, pady=pad)

        self._deny = ctk.CTkButton(
            buttons,
            text=t("gui.confirm.cancel"),
            width=120,
            command=lambda: self._on_answer(self.request, False),
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        self._deny.grid(row=0, column=0, padx=(0, 8))

        self._approve = ctk.CTkButton(
            buttons,
            text=t("gui.confirm.allow"),
            width=140,
            command=self._approve_clicked,
            fg_color=theme.palette.accent,
            hover_color=theme.palette.accent_hover,
            text_color=theme.palette.accent_text,
        )
        self._approve.grid(row=0, column=1)

    def _approve_clicked(self) -> None:
        if self._phrase is None:
            self._on_answer(self.request, True)
            return
        outcome = interpret_answer(self._phrase.get(), self.request, channel="gui")
        self._on_answer(self.request, outcome.approved)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        palette = theme.palette
        self.configure(fg_color=palette.background)
        self._card.configure(fg_color=palette.surface)
        self._title.configure(text_color=palette.text)
        self._body.configure(text_color=palette.text_muted)
        self._deny.configure(
            fg_color=palette.surface_alt,
            hover_color=palette.accent_soft,
            text_color=palette.text,
        )
        self._approve.configure(
            fg_color=palette.accent,
            hover_color=palette.accent_hover,
            text_color=palette.accent_text,
        )
        if self._phrase is not None:
            self._phrase.configure(
                fg_color=palette.surface_alt,
                border_color=palette.state_error,
                text_color=palette.text,
            )


def run_window(
    settings: Settings,
    report: object | None = None,
    *,
    speech_enabled: bool = True,
    start_in_voice_mode: bool = False,
) -> int:
    """Zbuduj okno i oddaj sterowanie pętli interfejsu. Zwraca kod wyjścia.

    Wyjątek ``tkinter.TclError`` znaczy zwykle jedno: nie ma z czym rysować (brak
    serwera X, sesja bez pulpitu, zdalny terminal). Nie jest to awaria programu —
    wołający zamienia to na komunikat i propozycję trybu terminalowego.
    """
    window = AssistantWindow(
        settings,
        report=report,
        speech_enabled=speech_enabled,
        start_in_voice_mode=start_in_voice_mode,
    )
    try:
        window.mainloop()
    except KeyboardInterrupt:  # pragma: no cover - przerwanie z terminala
        with_closing = getattr(window, "_on_close", None)
        if callable(with_closing):
            with_closing()
    return 0


__all__ = [
    "MAX_CHUNKS_PER_TICK",
    "POLL_MS",
    "AssistantWindow",
    "resolve_theme_mode",
    "run_window",
]
