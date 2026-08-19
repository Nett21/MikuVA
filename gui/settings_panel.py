"""Ekran ustawień w oknie asystenta (Faza 10).

Panel edytuje **wyłącznie** ``config/user_settings.json`` (przez
:func:`config.save_user_settings`) — nigdy pliku ``.env``. Widgety nie znają
struktury tego pliku: dostają listę pól z :mod:`gui.settings_form`, a ten moduł
zamienia je na zapis i pilnuje, żeby pola nieobecne na ekranie zostały
nietknięte.

Ścieżki do plików RVC wybiera się **systemowym oknem wyboru pliku**, nie z
klawiatury. Nie chodzi o wygodę: literówka w ścieżce daje „mowa działa, tylko
brzmi zwyczajnie", czyli błąd, którego nie widać. Wpisywanie ręczne zostaje
możliwe (pole jest edytowalne), ale nie jest domyślną drogą.

Panel jest **ramką w oknie**, a nie osobnym okienkiem. To decyzja przenośności:
dodatkowe okno na tilingowych menedżerach (Wayland, i3, Hyprland) bywa układane
jako kolejny kafelek albo traci powiązanie z rodzicem, a modalność zachowuje się
inaczej na każdej platformie.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from gui.settings_form import (
    ChoiceOptions,
    FieldSpec,
    SaveResult,
    SettingsForm,
    file_request,
    section_title,
    sections,
)
from gui.theme import Theme
from i18n import t

logger = logging.getLogger(__name__)


class SettingsPanel(ctk.CTkFrame):
    """Ekran ustawień: imię, kolor akcentu, cechy charakteru, mowa i RVC."""

    def __init__(
        self,
        master: Any,
        *,
        theme: Theme,
        form: SettingsForm,
        options: ChoiceOptions | None = None,
        on_saved: Callable[[SaveResult], None] | None = None,
        on_preview_accent: Callable[[str], None] | None = None,
        on_close: Callable[[], None] | None = None,
        on_voice_test: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color=theme.palette.background, corner_radius=0)
        self._theme = theme
        self._form = form
        self._choices = options or ChoiceOptions()
        self._on_saved = on_saved
        self._on_preview_accent = on_preview_accent
        self._on_close = on_close
        self._on_voice_test = on_voice_test

        self._widgets: dict[str, Any] = {}
        # Listy wyboru trzymamy osobno, żeby podmiana wartości nie musiała
        # szukać widgetu po wnętrzu CustomTkintera.
        self._menus: dict[str, Any] = {}
        self._value_labels: dict[str, ctk.CTkLabel] = {}
        self._swatch: ctk.CTkFrame | None = None
        self._themed: list[tuple[Any, str]] = []

        self._build()
        self._load_values()

    # --- budowa układu ------------------------------------------------------ #

    def _build(self) -> None:
        theme = self._theme
        pad = theme.metrics.pad

        header = ctk.CTkFrame(self, fg_color=theme.palette.surface, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(
            header,
            text=t("settings.title"),
            anchor="w",
            font=ctk.CTkFont(
                family=theme.font_family or None,
                size=theme.metrics.font_size_title,
                weight="bold",
            ),
            text_color=theme.palette.text,
        )
        title.grid(row=0, column=0, sticky="w", padx=pad, pady=pad)
        self._themed.append((title, "title"))

        close = ctk.CTkButton(
            header,
            text=t("settings.back"),
            width=150,
            command=self._handle_close,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        close.grid(row=0, column=2, sticky="e", padx=pad, pady=pad)
        self._themed.append((close, "secondary"))

        self._body = ctk.CTkScrollableFrame(
            self,
            fg_color=theme.palette.background,
            scrollbar_button_color=theme.palette.border,
            scrollbar_button_hover_color=theme.palette.accent,
        )
        self._body.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(pad, 0))
        self._body.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        row = 0
        for section, specs in sections():
            heading = ctk.CTkLabel(
                self._body,
                text=section_title(section),
                anchor="w",
                font=ctk.CTkFont(
                    family=theme.font_family or None,
                    size=theme.metrics.font_size + 1,
                    weight="bold",
                ),
                text_color=theme.palette.accent,
            )
            heading.grid(row=row, column=0, sticky="ew", pady=(pad, 4))
            self._themed.append((heading, "heading"))
            row += 1
            for spec in specs:
                card = self._build_field(spec)
                card.grid(row=row, column=0, sticky="ew", pady=4)
                row += 1

        footer = ctk.CTkFrame(self, fg_color=theme.palette.surface, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        self._message = ctk.CTkLabel(
            footer,
            text="",
            anchor="w",
            justify="left",
            wraplength=520,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size_small),
            text_color=theme.palette.text_muted,
        )
        self._message.grid(row=0, column=0, sticky="ew", padx=pad, pady=pad)
        self._themed.append((self._message, "muted"))

        revert = ctk.CTkButton(
            footer,
            text=t("settings.revert"),
            width=110,
            command=self._handle_revert,
            fg_color=theme.palette.surface_alt,
            hover_color=theme.palette.accent_soft,
            text_color=theme.palette.text,
        )
        revert.grid(row=0, column=1, padx=(0, 8), pady=pad)
        self._themed.append((revert, "secondary"))

        save = ctk.CTkButton(
            footer,
            text=t("settings.save"),
            width=130,
            command=self._handle_save,
            fg_color=theme.palette.accent,
            hover_color=theme.palette.accent_hover,
            text_color=theme.palette.accent_text,
        )
        save.grid(row=0, column=2, padx=(0, pad), pady=pad)
        self._themed.append((save, "primary"))

    def _build_field(self, spec: FieldSpec) -> ctk.CTkFrame:
        theme = self._theme
        pad = theme.metrics.pad
        card = ctk.CTkFrame(
            self._body, fg_color=theme.palette.surface, corner_radius=theme.metrics.radius
        )
        card.grid_columnconfigure(0, weight=1)
        self._themed.append((card, "card"))

        label = ctk.CTkLabel(
            card,
            text=spec.label,
            anchor="w",
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            text_color=theme.palette.text,
        )
        label.grid(row=0, column=0, sticky="w", padx=pad, pady=(pad // 2, 0))
        self._themed.append((label, "label"))

        control = self._build_control(card, spec)
        control.grid(row=1, column=0, sticky="ew", padx=pad, pady=4)

        if spec.help:
            note = ctk.CTkLabel(
                card,
                text=spec.help + ("" if spec.live else t("settings.needs_reload_suffix")),
                anchor="w",
                justify="left",
                wraplength=620,
                font=ctk.CTkFont(
                    family=theme.font_family or None, size=theme.metrics.font_size_small
                ),
                text_color=theme.palette.text_faint,
            )
            note.grid(row=2, column=0, sticky="ew", padx=pad, pady=(0, pad // 2))
            self._themed.append((note, "faint"))
        return card

    def _build_control(self, parent: Any, spec: FieldSpec) -> ctk.CTkFrame:
        theme = self._theme
        holder = ctk.CTkFrame(parent, fg_color="transparent")
        holder.grid_columnconfigure(0, weight=1)

        if spec.kind == "multiline":
            widget = ctk.CTkTextbox(
                holder,
                height=90,
                wrap="word",
                fg_color=theme.palette.surface_alt,
                border_color=theme.palette.border,
                text_color=theme.palette.text,
                font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
            )
            widget.grid(row=0, column=0, sticky="ew")
            self._widgets[spec.key] = widget
            self._themed.append((widget, "textbox"))
            return holder

        if spec.kind == "switch":
            variable = ctk.BooleanVar(value=False)
            widget = ctk.CTkSwitch(
                holder,
                text="",
                variable=variable,
                progress_color=theme.palette.accent,
                button_color=theme.palette.surface_alt,
                button_hover_color=theme.palette.accent_hover,
            )
            widget.grid(row=0, column=0, sticky="w")
            self._widgets[spec.key] = variable
            self._themed.append((widget, "switch"))
            return holder

        if spec.kind == "choice":
            values = list(self._choices.values_for(spec)) or [""]
            variable = ctk.StringVar(value=values[0])
            widget = ctk.CTkOptionMenu(
                holder,
                variable=variable,
                values=[self._display_choice(value) for value in values],
                fg_color=theme.palette.surface_alt,
                button_color=theme.palette.accent,
                button_hover_color=theme.palette.accent_hover,
                text_color=theme.palette.text,
                dropdown_fg_color=theme.palette.surface,
                dropdown_hover_color=theme.palette.accent_soft,
                dropdown_text_color=theme.palette.text,
            )
            widget.grid(row=0, column=0, sticky="w")
            self._widgets[spec.key] = variable
            self._menus[spec.key] = widget
            self._themed.append((widget, "option"))
            if spec.choices_source == "piper_voices" and self._on_voice_test is not None:
                test = ctk.CTkButton(
                    holder,
                    text=t("settings.listen_sample"),
                    width=110,
                    command=self._on_voice_test,
                    fg_color=theme.palette.surface_alt,
                    hover_color=theme.palette.accent_soft,
                    text_color=theme.palette.text,
                )
                test.grid(row=0, column=1, padx=(8, 0))
                self._themed.append((test, "secondary"))
            return holder

        if spec.kind in ("int", "float"):
            minimum = float(spec.minimum if spec.minimum is not None else 0.0)
            maximum = float(spec.maximum if spec.maximum is not None else 1.0)
            step = float(spec.step or (1.0 if spec.kind == "int" else 0.05))
            steps = max(1, int(round((maximum - minimum) / step)))
            variable = ctk.DoubleVar(value=minimum)
            widget = ctk.CTkSlider(
                holder,
                from_=minimum,
                to=maximum,
                number_of_steps=steps,
                variable=variable,
                command=lambda value, key=spec.key: self._on_slider(key, value),
                progress_color=theme.palette.accent,
                button_color=theme.palette.accent,
                button_hover_color=theme.palette.accent_hover,
                fg_color=theme.palette.surface_alt,
            )
            widget.grid(row=0, column=0, sticky="ew")
            value_label = ctk.CTkLabel(
                holder,
                text="",
                width=60,
                font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
                text_color=theme.palette.text_muted,
            )
            value_label.grid(row=0, column=1, padx=(8, 0))
            self._widgets[spec.key] = variable
            self._value_labels[spec.key] = value_label
            self._themed.append((widget, "slider"))
            self._themed.append((value_label, "muted"))
            return holder

        # text / color / path — wszystkie na polu tekstowym, różnią się dodatkami.
        variable = ctk.StringVar(value="")
        entry = ctk.CTkEntry(
            holder,
            textvariable=variable,
            placeholder_text=spec.placeholder,
            fg_color=theme.palette.surface_alt,
            border_color=theme.palette.border,
            text_color=theme.palette.text,
            font=ctk.CTkFont(family=theme.font_family or None, size=theme.metrics.font_size),
        )
        entry.grid(row=0, column=0, sticky="ew")
        self._widgets[spec.key] = variable
        self._themed.append((entry, "entry"))

        if spec.kind == "color":
            self._swatch = ctk.CTkFrame(
                holder,
                width=34,
                height=28,
                corner_radius=8,
                fg_color=self._form.preview_accent(),
            )
            self._swatch.grid(row=0, column=1, padx=(8, 0))
            self._swatch.grid_propagate(False)
            # Podglądu nie odpalamy na każdym znaku: „#3" nie jest kolorem, a
            # przemalowanie okna przy każdym naciśnięciu klawisza to migotanie.
            entry.bind("<Return>", lambda _event: self._preview_accent())
            entry.bind("<FocusOut>", lambda _event: self._preview_accent())

        if spec.kind == "path":
            browse = ctk.CTkButton(
                holder,
                text=t("settings.browse"),
                width=150,
                command=lambda key=spec.key: self._browse(key),
                fg_color=theme.palette.accent_soft,
                hover_color=theme.palette.accent_hover,
                text_color=theme.palette.text,
            )
            browse.grid(row=0, column=1, padx=(8, 0))
            self._themed.append((browse, "secondary"))
            clear = ctk.CTkButton(
                holder,
                text=t("settings.clear"),
                width=90,
                command=lambda key=spec.key: self._widgets[key].set(""),
                fg_color=theme.palette.surface_alt,
                hover_color=theme.palette.accent_soft,
                text_color=theme.palette.text_muted,
            )
            clear.grid(row=0, column=2, padx=(8, 0))
            self._themed.append((clear, "secondary"))
        return holder

    # --- wartości ------------------------------------------------------------ #

    def _display_choice(self, value: str) -> str:
        return self._choices.label_for(value)

    def _choice_value(self, spec: FieldSpec, shown: str) -> str:
        for value in self._choices.values_for(spec):
            if self._display_choice(value) == shown:
                return value
        return shown

    def _load_values(self) -> None:
        """Przepisz wartości z formularza do widgetów."""
        for _, specs in sections():
            for spec in specs:
                widget = self._widgets.get(spec.key)
                if widget is None:
                    continue
                value = self._form.value(spec.key)
                if spec.kind == "multiline":
                    widget.delete("1.0", "end")
                    widget.insert("1.0", str(value or ""))
                elif spec.kind == "switch":
                    widget.set(bool(value))
                elif spec.kind in ("int", "float"):
                    widget.set(float(value or 0))
                    self._update_value_label(spec.key, float(value or 0))
                elif spec.kind == "choice":
                    widget.set(self._display_choice(str(value or "")))
                else:
                    widget.set(str(value or ""))
        self._update_swatch()

    def _collect(self) -> None:
        """Przepisz wartości z widgetów do formularza (z normalizacją)."""
        for _, specs in sections():
            for spec in specs:
                widget = self._widgets.get(spec.key)
                if widget is None:
                    continue
                if spec.kind == "multiline":
                    raw: Any = widget.get("1.0", "end")
                elif spec.kind == "choice":
                    raw = self._choice_value(spec, str(widget.get()))
                else:
                    raw = widget.get()
                self._form.set(spec.key, raw)

    def set_options(self, options: ChoiceOptions) -> None:
        """Podmień listy wyboru (głosy Pipera znalezione przez wątek roboczy)."""
        self._choices = options
        for _, specs in sections():
            for spec in specs:
                if spec.kind != "choice":
                    continue
                widget = self._widgets.get(spec.key)
                if widget is None:
                    continue
                current = self._form.value(spec.key)
                values = [self._display_choice(value) for value in options.values_for(spec)]
                menu = self._menus.get(spec.key)
                if menu is not None and values:
                    menu.configure(values=values)
                widget.set(self._display_choice(str(current or "")))

    # --- reakcje na kliknięcia ---------------------------------------------- #

    def _on_slider(self, key: str, value: float) -> None:
        self._update_value_label(key, value)

    def _update_value_label(self, key: str, value: float) -> None:
        label = self._value_labels.get(key)
        if label is None:
            return
        whole = abs(value) >= 1 or float(value).is_integer()
        text = f"{int(round(value)):+d}" if whole else f"{value:.2f}"
        if key.endswith("index_rate"):
            text = f"{value:.2f}"
        label.configure(text=text)

    def _update_swatch(self) -> None:
        if self._swatch is None:
            return
        with_accent = self._form.preview_accent()
        try:
            self._swatch.configure(fg_color=with_accent)
        except Exception:  # pragma: no cover - widget zamknięty
            pass

    def _preview_accent(self) -> None:
        """Pokaż wpisany kolor od razu, jeszcze przed zapisem."""
        self._collect()
        self._update_swatch()
        if self._on_preview_accent is not None:
            self._on_preview_accent(self._form.preview_accent())

    def _browse(self, key: str) -> None:
        """Systemowe okno wyboru pliku dla pola ścieżki."""
        variable = self._widgets.get(key)
        if variable is None:
            return
        request = file_request(key, str(variable.get() or ""))
        try:
            chosen = filedialog.askopenfilename(
                parent=self.winfo_toplevel(), **request.as_dialog_kwargs()
            )
        except Exception as exc:  # pragma: no cover - brak wsparcia dla okna
            logger.warning("Nie udało się otworzyć okna wyboru pliku: %s", exc)
            self._show_message(t("settings.dialog_unavailable"), kind="warning")
            return
        if not chosen:
            return
        variable.set(self._form.set(key, chosen))

    def _handle_revert(self) -> None:
        self._form.revert()
        self._load_values()
        self._show_message(t("settings.reverted"))
        if self._on_preview_accent is not None:
            self._on_preview_accent(self._form.preview_accent())

    def _handle_close(self) -> None:
        if self._on_close is not None:
            self._on_close()

    def _handle_save(self) -> None:
        self._collect()
        result = self._form.save()
        if not result.ok:
            self._show_message(result.message(), kind="error")
            return
        message = result.message()
        if result.warnings:
            message += "  " + "  ".join(result.warnings)
        self._show_message(message, kind="warning" if result.warnings else "ok")
        self._load_values()
        if self._on_saved is not None:
            self._on_saved(result)

    def _show_message(self, text: str, *, kind: str = "ok") -> None:
        colors = {
            "ok": self._theme.palette.text_muted,
            "warning": self._theme.palette.state_busy,
            "error": self._theme.palette.state_error,
        }
        self._message.configure(text=text, text_color=colors.get(kind, self._theme.palette.text))

    # --- motyw --------------------------------------------------------------- #

    def apply_theme(self, theme: Theme) -> None:
        """Przemaluj panel — także wtedy, gdy jest otwarty w chwili zmiany akcentu."""
        self._theme = theme
        palette = theme.palette
        self.configure(fg_color=palette.background)
        self._body.configure(
            fg_color=palette.background,
            scrollbar_button_color=palette.border,
            scrollbar_button_hover_color=palette.accent,
        )
        for widget, role in self._themed:
            try:
                self._apply_role(widget, role, theme)
            except Exception:  # pragma: no cover - zamykane widgety
                continue
        self._update_swatch()

    @staticmethod
    def _apply_role(widget: Any, role: str, theme: Theme) -> None:
        palette = theme.palette
        if role in ("title", "heading", "label"):
            widget.configure(
                text_color=palette.accent if role == "heading" else palette.text
            )
        elif role == "muted":
            widget.configure(text_color=palette.text_muted)
        elif role == "faint":
            widget.configure(text_color=palette.text_faint)
        elif role == "card":
            widget.configure(fg_color=palette.surface)
        elif role == "primary":
            widget.configure(
                fg_color=palette.accent,
                hover_color=palette.accent_hover,
                text_color=palette.accent_text,
            )
        elif role == "secondary":
            widget.configure(
                fg_color=palette.surface_alt,
                hover_color=palette.accent_soft,
                text_color=palette.text,
            )
        elif role in ("entry", "textbox"):
            widget.configure(
                fg_color=palette.surface_alt,
                border_color=palette.border,
                text_color=palette.text,
            )
        elif role == "option":
            widget.configure(
                fg_color=palette.surface_alt,
                button_color=palette.accent,
                button_hover_color=palette.accent_hover,
                text_color=palette.text,
                dropdown_fg_color=palette.surface,
                dropdown_hover_color=palette.accent_soft,
                dropdown_text_color=palette.text,
            )
        elif role == "slider":
            widget.configure(
                progress_color=palette.accent,
                button_color=palette.accent,
                button_hover_color=palette.accent_hover,
                fg_color=palette.surface_alt,
            )
        elif role == "switch":
            widget.configure(
                progress_color=palette.accent,
                button_color=palette.surface_alt,
                button_hover_color=palette.accent_hover,
            )


__all__ = ["SettingsPanel"]
