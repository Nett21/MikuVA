"""Panel statusu i wskaźnik nasłuchiwania (Faza 10).

Panel odpowiada na pytania, które przy asystencie głosowym pada się najczęściej:
czy on mnie w ogóle słyszy, jakim modelem myśli, w jakim języku odpowie i co z
tego wszystkiego akurat nie działa. Dane pochodzą z migawki
:class:`gui.state.StatusSnapshot` — panel nie sprawdza niczego sam, bo sprawdzanie
sprzętu w wątku interfejsu to najprostszy sposób na zamrożone okno.

Wskaźnik nasłuchiwania **pulsuje** kolorem akcentu, gdy asystent słucha, myśli
albo mówi. Puls chodzi po ``after()`` w wątku interfejsu (tanie, kilka odświeżeń
na sekundę) i zatrzymuje się sam, gdy nie ma czego pokazywać — animacja kręcąca
się w tle na laptopie na baterii to nie funkcja.
"""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from gui.state import ListeningState, ServiceState, StatusSnapshot
from gui.theme import Theme, mix, parse_color, to_hex
from i18n import t

# Co ile milisekund zmienia się jasność wskaźnika. Wartość zauważalna, ale nie
# nerwowa; przy 4 krokach daje pełny oddech w niecałą sekundę.
_PULSE_MS: int = 220
_PULSE_STEPS: tuple[float, ...] = (0.0, 0.35, 0.7, 0.35)


class ListeningIndicator(ctk.CTkFrame):
    """Kropka „słucham/myślę/mówię" z podpisem, w kolorze akcentu."""

    def __init__(self, master: Any, *, theme: Theme) -> None:
        super().__init__(master, fg_color=theme.palette.surface, bg_color=theme.palette.surface)
        self._theme = theme
        self._listening_state = ListeningState.OFF
        self._wake_phrase = ""
        self._step = 0
        self._job: str | None = None

        size = theme.metrics.indicator_size
        self._dot = ctk.CTkFrame(
            self,
            width=size,
            height=size,
            corner_radius=size // 2,
            fg_color=theme.palette.listening_idle,
        )
        self._dot.grid(row=0, column=0, padx=(0, 8))
        self._dot.grid_propagate(False)

        self._caption = ctk.CTkLabel(
            self,
            text=self._listening_state.caption(),
            anchor="w",
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            text_color=theme.palette.text_muted,
            fg_color=theme.palette.surface,
        )
        self._caption.grid(row=0, column=1, sticky="w")
        self.grid_columnconfigure(1, weight=1)

    def set_state(self, state: ListeningState, *, wake_phrase: str = "") -> None:
        self._wake_phrase = wake_phrase or self._wake_phrase
        changed = state is not self._listening_state
        self._listening_state = state
        self._caption.configure(text=state.caption(wake_phrase=self._wake_phrase))
        if changed:
            self._step = 0
        self._repaint()
        if state.is_active:
            self._start_pulse()
        else:
            self._stop_pulse()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        size = theme.metrics.indicator_size
        self._dot.configure(width=size, height=size, corner_radius=size // 2)
        self._caption.configure(
            text_color=theme.palette.text_muted,
            fg_color=theme.palette.surface,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
        )
        self._repaint()

    def destroy(self) -> None:  # pragma: no cover - zamykanie okna
        self._stop_pulse()
        super().destroy()

    # --- puls --------------------------------------------------------------- #

    def _repaint(self) -> None:
        base = self._listening_state.color(self._theme.palette)
        if not self._listening_state.is_active:
            self._safe_color(base)
            return
        # Puls to domieszka koloru tła — kropka gaśnie i rozjaśnia się,
        # zachowując odcień akcentu.
        weight = _PULSE_STEPS[self._step % len(_PULSE_STEPS)]
        color = parse_color(base)
        ground = parse_color(self._theme.palette.surface)
        if color is None or ground is None:  # pragma: no cover - obrona
            self._safe_color(base)
            return
        self._safe_color(to_hex(mix(color, ground, weight)))

    def _safe_color(self, color: str) -> None:
        try:
            self._dot.configure(fg_color=color)
        except Exception:  # pragma: no cover - widget w trakcie niszczenia
            pass

    def _start_pulse(self) -> None:
        if self._job is not None:
            return
        self._tick()

    def _tick(self) -> None:
        if not self.winfo_exists():  # pragma: no cover - okno zamknięte
            self._job = None
            return
        self._step += 1
        self._repaint()
        if self._listening_state.is_active:
            self._job = self.after(_PULSE_MS, self._tick)
        else:
            self._job = None

    def _stop_pulse(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:  # pragma: no cover - zależne od stanu okna
                pass


class _StatusRow(ctk.CTkFrame):
    """Jedna pozycja: kropka stanu, etykieta i szczegół."""

    def __init__(self, master: Any, *, theme: Theme, label: str) -> None:
        super().__init__(master, fg_color=theme.palette.surface, bg_color=theme.palette.surface)
        self._theme = theme
        self._dot = ctk.CTkFrame(
            self, width=9, height=9, corner_radius=5, fg_color=theme.palette.state_off
        )
        self._dot.grid(row=0, column=0, padx=(0, 8), pady=(4, 0), sticky="n")
        self._dot.grid_propagate(False)

        self._name_label = ctk.CTkLabel(
            self,
            text=label,
            anchor="w",
            font=ctk.CTkFont(
                family=theme.font_family or None, size=theme.metrics.font_size_small, weight="bold"
            ),
            text_color=theme.palette.text,
            fg_color=theme.palette.surface,
        )
        self._name_label.grid(row=0, column=1, sticky="w")

        self._detail = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            wraplength=theme.metrics.sidebar_width - 60,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=theme.palette.text_muted,
            fg_color=theme.palette.surface,
        )
        self._detail.grid(row=1, column=1, sticky="w", pady=(0, 6))
        self.grid_columnconfigure(1, weight=1)

    def update_status(self, state: ServiceState, detail: str) -> None:
        self._dot.configure(fg_color=state.color(self._theme.palette))
        self._detail.configure(text=detail or t("common.dash"))

    def apply_theme(self, theme: Theme, state: ServiceState) -> None:
        self._theme = theme
        self.configure(fg_color=theme.palette.surface, bg_color=theme.palette.surface)
        self._dot.configure(fg_color=state.color(theme.palette))
        self._name_label.configure(text_color=theme.palette.text, fg_color=theme.palette.surface)
        self._detail.configure(
            text_color=theme.palette.text_muted,
            fg_color=theme.palette.surface,
            wraplength=theme.metrics.sidebar_width - 60,
        )


class StatusPanel(ctk.CTkFrame):
    """Kolumna stanu: model, język, wskaźnik nasłuchu i lista usług."""

    def __init__(
        self, master: Any, *, theme: Theme, snapshot: StatusSnapshot | None = None
    ) -> None:
        super().__init__(
            master,
            fg_color=theme.palette.surface,
            corner_radius=0,
            width=theme.metrics.sidebar_width,
        )
        self._theme = theme
        self._snapshot = snapshot or StatusSnapshot()
        self._rows: dict[str, _StatusRow] = {}

        pad = theme.metrics.pad

        self._title = ctk.CTkLabel(
            self,
            text=t("gui.status.title", name=""),
            anchor="w",
            font=ctk.CTkFont(
                family=theme.font_family or None, size=theme.metrics.font_size_title, weight="bold"
            ),
            text_color=theme.palette.text,
        )
        self._title.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, 2))

        self.indicator = ListeningIndicator(self, theme=theme)
        self.indicator.grid(row=1, column=0, sticky="ew", padx=pad, pady=(4, pad))

        self._model = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            wraplength=theme.metrics.sidebar_width - 2 * pad,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=theme.palette.text_muted,
        )
        self._model.grid(row=2, column=0, sticky="ew", padx=pad)

        self._language = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            wraplength=theme.metrics.sidebar_width - 2 * pad,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=theme.palette.text_muted,
        )
        self._language.grid(row=3, column=0, sticky="ew", padx=pad, pady=(0, pad))

        self._rows_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=theme.palette.border,
            scrollbar_button_hover_color=theme.palette.accent,
        )
        self._rows_frame.grid(row=4, column=0, sticky="nsew", padx=pad // 2)
        self._rows_frame.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.update_snapshot(self._snapshot)

    # --- odświeżanie --------------------------------------------------------- #

    def update_snapshot(self, snapshot: StatusSnapshot) -> None:
        """Przepisz migawkę na ekran. Wołane wyłącznie z wątku interfejsu."""
        self._snapshot = snapshot
        self._title.configure(text=t("gui.status.title", name=snapshot.assistant_name))
        model = snapshot.model or t("gui.status.no_model")
        self._model.configure(
            text=f"{t('gui.status.model', model=model)}\n{snapshot.host or ''}".strip()
        )
        self._language.configure(
            text=t("gui.status.language", language=snapshot.language_label())
        )
        self.indicator.set_state(snapshot.listening, wake_phrase=snapshot.wake_phrase)

        for index, item in enumerate(snapshot.services):
            row = self._rows.get(item.key)
            if row is None:
                row = _StatusRow(self._rows_frame, theme=self._theme, label=item.label)
                row.grid(row=index, column=0, sticky="ew", padx=self._theme.metrics.pad // 2)
                self._rows[item.key] = row
            row.update_status(item.state, item.detail)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        pad = theme.metrics.pad
        self.configure(fg_color=theme.palette.surface, width=theme.metrics.sidebar_width)
        self._title.configure(
            text_color=theme.palette.text,
            font=ctk.CTkFont(
                family=theme.font_family or None, size=theme.metrics.font_size_title, weight="bold"
            ),
        )
        for label in (self._model, self._language):
            label.configure(
                text_color=theme.palette.text_muted,
                wraplength=theme.metrics.sidebar_width - 2 * pad,
                font=ctk.CTkFont(
                    family=theme.font_family or None, size=theme.metrics.font_size_small
                ),
            )
        self._rows_frame.configure(
            scrollbar_button_color=theme.palette.border,
            scrollbar_button_hover_color=theme.palette.accent,
        )
        self.indicator.apply_theme(theme)
        for item in self._snapshot.services:
            row = self._rows.get(item.key)
            if row is not None:
                row.apply_theme(theme, item.state)


__all__ = ["ListeningIndicator", "StatusPanel"]
