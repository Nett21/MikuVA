"""Widok rozmowy: bąbelki czatu w CustomTkinterze (Faza 10).

Ten moduł tylko **rysuje** to, co opisuje :mod:`gui.state`. Nie zna modelu
językowego, nie wie o wątkach i nie decyduje, co się w rozmowie pojawia — dostaje
gotowe wiadomości i maluje je kolorami z :mod:`gui.theme`.

Wszystkie kolory pochodzą z palety, a paleta z ``ui_accent_color``. Dlatego
:meth:`ChatView.apply_theme` potrafi przemalować całą historię rozmowy w miejscu:
zmiana koloru akcentu w panelu ustawień jest widoczna natychmiast, bez restartu i
bez czyszczenia okna.

Dopisywanie strumienia odpowiedzi idzie do **istniejącego** bąbelka
(:meth:`append_chunk`), a nie tworzy nowego widgetu na każdy fragment — inaczej
przy dłuższej odpowiedzi okno tworzyłoby setki widgetów na sekundę.
"""

from __future__ import annotations

import logging
from typing import Any

import customtkinter as ctk

from gui.state import ChatLog, ChatMessage, ChatRole
from gui.theme import Theme
from i18n import t

logger = logging.getLogger(__name__)

# Jaką część szerokości okna może zająć bąbelek. Wartość względna, bo szerokość
# okna zależy od ekranu, a nie od naszych założeń.
_BUBBLE_WIDTH_RATIO: float = 0.74
_MIN_WRAP: int = 240

# Podpisy roli biorą się z katalogu tekstów (język interfejsu). Imię asystenta
# jest wstawiane dynamicznie — w kodzie nie ma żadnego imienia na sztywno
# (patrz ``ChatView.set_assistant_name``).
_ROLE_KEYS: dict[ChatRole, str] = {
    ChatRole.USER: "gui.role.user",
    ChatRole.SYSTEM: "gui.role.system",
    ChatRole.TOOL: "gui.role.tool",
    ChatRole.ERROR: "gui.role.error",
}


class _Bubble(ctk.CTkFrame):
    """Jeden dymek rozmowy: nagłówek (kto, kiedy) i treść."""

    def __init__(
        self,
        master: Any,
        message: ChatMessage,
        *,
        theme: Theme,
        caption: str,
        wraplength: int,
    ) -> None:
        background, foreground = message.bubble_colors(theme.palette)
        super().__init__(
            master,
            fg_color=background,
            # Kolor POD zaokrąglonym rogiem. Domyślnie CustomTkinter wykrywa go z
            # rodzica przy tworzeniu i gubi przy zmianie skalowania — zostawał
            # czarny narożnik do czasu najechania myszką.
            bg_color=theme.palette.background,
            corner_radius=theme.metrics.bubble_radius,
        )
        self.message = message
        self._theme = theme

        self._caption = ctk.CTkLabel(
            self,
            text=caption,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=foreground,
            # Tło WPROST, a nie „transparent": przezroczysta etykieta rysuje się
            # kolorem wykrytym z rodzica przy tworzeniu, a po zmianie skalowania
            # CustomTkinter gubi ten kolor i zostaje czarny prostokąt do czasu
            # najechania myszką. Regresja zgłoszona z prawdziwego pulpitu.
            fg_color=background,
            corner_radius=theme.metrics.bubble_radius,
        )
        self._caption.grid(row=0, column=0, sticky="w", padx=theme.metrics.pad, pady=(6, 0))

        self._body = ctk.CTkLabel(
            self,
            text=message.text,
            anchor="w",
            justify="left",
            wraplength=wraplength,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            text_color=foreground,
            fg_color=background,
            corner_radius=theme.metrics.bubble_radius,
        )
        self._body.grid(row=1, column=0, sticky="w", padx=theme.metrics.pad, pady=(0, 8))
        self.grid_columnconfigure(0, weight=1)

    def set_text(self, text: str) -> None:
        self._body.configure(text=text)

    def set_caption(self, caption: str) -> None:
        self._caption.configure(text=caption)

    def set_wraplength(self, wraplength: int) -> None:
        self._body.configure(wraplength=wraplength)

    def apply_theme(self, theme: Theme) -> None:
        """Przemaluj dymek na nową paletę (zmiana akcentu bez restartu)."""
        self._theme = theme
        background, foreground = self.message.bubble_colors(theme.palette)
        self.configure(
            fg_color=background,
            bg_color=theme.palette.background,
            corner_radius=theme.metrics.bubble_radius,
        )
        self._caption.configure(
            text_color=foreground,
            fg_color=background,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
        )
        self._body.configure(
            text_color=foreground,
            fg_color=background,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
        )


class ChatView(ctk.CTkFrame):
    """Przewijana historia rozmowy z bąbelkami po obu stronach."""

    def __init__(
        self,
        master: Any,
        *,
        theme: Theme,
        assistant_name: str = "",
        log: ChatLog | None = None,
    ) -> None:
        super().__init__(master, fg_color=theme.palette.background, corner_radius=0)
        self._theme = theme
        self._assistant_name = assistant_name or t("gui.role.assistant_fallback")
        self._log = log if log is not None else ChatLog()
        self._bubbles: dict[int, _Bubble] = {}
        self._wraplength = 520

        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=theme.palette.background,
            scrollbar_button_color=theme.palette.border,
            scrollbar_button_hover_color=theme.palette.accent,
            corner_radius=0,
        )
        self._scroll.grid(row=0, column=0, sticky="nsew")
        self._scroll.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.bind("<Configure>", self._on_resize)

    # --- właściwości ------------------------------------------------------- #

    @property
    def log(self) -> ChatLog:
        return self._log

    def set_assistant_name(self, name: str) -> None:
        """Nowe imię asystenta — także w podpisach wiadomości już widocznych."""
        cleaned = (name or "").strip() or t("gui.role.assistant_fallback")
        if cleaned == self._assistant_name:
            return
        self._assistant_name = cleaned
        for bubble in self._bubbles.values():
            bubble.set_caption(self._caption_for(bubble.message))

    # --- dodawanie treści --------------------------------------------------- #

    def add(self, role: ChatRole, text: str, *, detail: str = "") -> ChatMessage:
        message = self._log.add(role, text, detail=detail)
        self._draw_message(message)
        self._scroll_to_end()
        return message

    def start_reply(self) -> ChatMessage:
        """Otwórz dymek asystenta, do którego popłynie strumień odpowiedzi."""
        self._drop_empty_streaming()
        message = self._log.start_assistant()
        self._draw_message(message)
        self._scroll_to_end()
        return message

    def append_chunk(self, text: str) -> None:
        """Dopisz fragment odpowiedzi do otwartego dymka."""
        if not text:
            return
        was_streaming = self._log.is_streaming
        message = self._log.append_chunk(text)
        if not was_streaming:
            self._draw_message(message)
        bubble = self._bubbles.get(id(message))
        if bubble is not None:
            bubble.set_text(message.text)
        self._scroll_to_end()

    def finish_reply(self, text: str | None = None) -> None:
        """Zamknij strumień; puste odpowiedzi nie zostawiają pustego dymka."""
        pending = self._log.streaming_message
        message = self._log.finish(text)
        if message is None:
            if pending is not None:
                self._remove(pending)
            return
        bubble = self._bubbles.get(id(message))
        if bubble is not None:
            bubble.set_text(message.text)
        self._scroll_to_end()

    def clear(self) -> None:
        for bubble in list(self._bubbles.values()):
            bubble.destroy()
        self._bubbles.clear()
        self._log.clear()

    # --- motyw i układ ------------------------------------------------------ #

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.configure(fg_color=theme.palette.background)
        self._scroll.configure(
            fg_color=theme.palette.background,
            scrollbar_button_color=theme.palette.border,
            scrollbar_button_hover_color=theme.palette.accent,
        )
        for bubble in self._bubbles.values():
            bubble.apply_theme(theme)

    def _on_resize(self, event: Any) -> None:
        width = int(getattr(event, "width", 0) or 0)
        if width <= 1:
            return
        wraplength = max(_MIN_WRAP, int(width * _BUBBLE_WIDTH_RATIO))
        if abs(wraplength - self._wraplength) < 12:
            return
        self._wraplength = wraplength
        for bubble in self._bubbles.values():
            bubble.set_wraplength(wraplength)

    # --- rysowanie --------------------------------------------------------- #

    def _caption_for(self, message: ChatMessage) -> str:
        role_key = _ROLE_KEYS.get(message.role)
        who = self._assistant_name if role_key is None else t(role_key)
        parts = [who]
        moment = message.time_label()
        if moment:
            parts.append(moment)
        if message.detail:
            parts.append(message.detail)
        return "  ·  ".join(parts)

    def _draw_message(self, message: ChatMessage) -> None:
        row = len(self._bubbles)
        bubble = _Bubble(
            self._scroll,
            message,
            theme=self._theme,
            caption=self._caption_for(message),
            wraplength=self._wraplength,
        )
        # Wiadomość użytkownika po prawej, reszta po lewej — bez tego rozmowa
        # czyta się jak jednolita ściana tekstu.
        sticky = "e" if message.role is ChatRole.USER else "w"
        bubble.grid(
            row=row,
            column=0,
            sticky=sticky,
            padx=self._theme.metrics.gap,
            pady=self._theme.metrics.gap // 2,
        )
        self._bubbles[id(message)] = bubble

    def _remove(self, message: ChatMessage) -> None:
        bubble = self._bubbles.pop(id(message), None)
        if bubble is not None:
            bubble.destroy()

    def _drop_empty_streaming(self) -> None:
        pending = self._log.streaming_message
        if pending is not None and pending.is_empty:
            self._log.finish()
            self._remove(pending)

    def _scroll_to_end(self) -> None:
        """Przewiń na dół.

        CustomTkinter nie ma na to publicznej metody, więc sięgamy po jego pole
        wewnętrzne — ale **defensywnie**: gdy przyszła wersja biblioteki je
        przemianuje, rozmowa po prostu przestanie się sama przewijać, a nie
        wywali okno. Kolejność: najpierw pozwól policzyć układ, potem przewiń.
        """
        canvas = getattr(self._scroll, "_parent_canvas", None)
        if canvas is None:
            return
        try:
            self._scroll.update_idletasks()
            canvas.yview_moveto(1.0)
        except Exception:  # pragma: no cover - zależne od wersji biblioteki
            logger.debug("Nie udało się przewinąć rozmowy", exc_info=True)


__all__ = ["ChatView"]
