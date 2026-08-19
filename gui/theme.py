"""Motyw graficzny liczony z JEDNEGO pola: ``ui_accent_color`` (Faza 10).

W kodzie GUI nie ma ani jednego koloru wpisanego „bo tak wygląda dobrze". Każdy
odcień — tło, bąbelki czatu, obramowania, wskaźnik nasłuchiwania — powstaje z
akcentu podanego przez użytkownika w ``config/user_settings.json``. Dzięki temu
zmiana tego jednego pola realnie zmienia wygląd całego okna, bez zaglądania w
kod, i nie da się „zapomnieć" o jakimś elemencie: nie istnieje miejsce, które
trzymałoby własną kopię koloru.

Moduł jest **czysty**: nie importuje tkintera ani CustomTkintera, nie dotyka
plików i nie zależy od systemu. Da się go w całości przetestować bez ekranu, a
paleta jest deterministyczna — ten sam akcent zawsze daje ten sam wynik.

Dwa tryby jasności (``light``/``dark``) są liczone osobno, bo czytelność tekstu
zależy od tła. Kolor tekstu na akcencie nie jest zgadywany: wybieramy czarny albo
biały według **kontrastu** (WCAG), więc jasnożółty akcent dostaje czarny napis, a
granatowy — biały.
"""

from __future__ import annotations

import colorsys
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

logger = logging.getLogger(__name__)

# Akcent używany, gdy pole użytkownika jest puste albo nie da się go odczytać.
# Ta sama wartość co domyślna w ``config.UserSettings.ui_accent_color`` — ale
# GUI nigdy nie zakłada, że dostanie poprawny kolor: pole bywa edytowane ręcznie.
DEFAULT_ACCENT: Final[str] = "#39C5BB"

ThemeMode = Literal["light", "dark"]

_HEX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Skrajne punkty jasności. Świadomie NIE czysta czerń i biel: pełny kontrast
# tła męczy oczy przy dłuższej rozmowie, a bąbelki przestają być widoczne.
_DARK_BASE: Final[tuple[int, int, int]] = (18, 20, 24)
_LIGHT_BASE: Final[tuple[int, int, int]] = (248, 249, 251)

# Próg kontrastu dla tekstu na kolorowym tle (WCAG AA dla dużego tekstu to 3.0,
# dla zwykłego 4.5). Wybieramy wariant o większym kontraście, a próg służy
# tylko do zalogowania ostrzeżenia — nie do odrzucenia koloru użytkownika.
_MIN_CONTRAST: Final[float] = 3.0

RGB = tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Operacje na kolorach
# --------------------------------------------------------------------------- #


def parse_color(value: str | None) -> RGB | None:
    """``"#39C5BB"`` / ``"39c5bb"`` / ``"#abc"`` → ``(r, g, b)``. ``None`` = nie kolor.

    Zapis trzyznakowy jest rozwijany tak jak w CSS (``#abc`` = ``#aabbcc``), bo
    użytkownik wpisujący kolor ręcznie ma prawo użyć krótszej formy.
    """
    if not value:
        return None
    match = _HEX_PATTERN.match(value.strip())
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))


def to_hex(color: RGB) -> str:
    """``(57, 197, 187)`` → ``"#39c5bb"`` (zapis, który rozumie każdy toolkit)."""
    red, green, blue = (max(0, min(255, int(round(channel)))) for channel in color)
    return f"#{red:02x}{green:02x}{blue:02x}"


def normalize_accent(value: str | None) -> str:
    """Akcent użytkownika sprowadzony do zapisu ``#rrggbb``; przy błędzie domyślny."""
    parsed = parse_color(value)
    if parsed is None:
        if value:
            logger.info(
                "ui_accent_color=%r nie jest kolorem szesnastkowym — używam %s.",
                value,
                DEFAULT_ACCENT,
            )
        return DEFAULT_ACCENT.lower()
    return to_hex(parsed)


def mix(first: RGB, second: RGB, weight: float) -> RGB:
    """Zmieszaj dwa kolory. ``weight=0`` → pierwszy, ``1`` → drugi."""
    ratio = max(0.0, min(1.0, weight))
    return (
        round(first[0] + (second[0] - first[0]) * ratio),
        round(first[1] + (second[1] - first[1]) * ratio),
        round(first[2] + (second[2] - first[2]) * ratio),
    )


def _with_lightness(color: RGB, *, delta: float) -> RGB:
    """Rozjaśnij (``delta>0``) albo przyciemnij kolor, zachowując odcień."""
    red, green, blue = (channel / 255 for channel in color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness = max(0.0, min(1.0, lightness + delta))
    result = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (round(result[0] * 255), round(result[1] * 255), round(result[2] * 255))


def lighten(color: RGB, amount: float) -> RGB:
    return _with_lightness(color, delta=abs(amount))


def darken(color: RGB, amount: float) -> RGB:
    return _with_lightness(color, delta=-abs(amount))


def saturate(color: RGB, factor: float) -> RGB:
    """Wzmocnij (``factor>1``) albo zgaś nasycenie — odcień zostaje ten sam."""
    red, green, blue = (channel / 255 for channel in color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    saturation = max(0.0, min(1.0, saturation * factor))
    result = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (round(result[0] * 255), round(result[1] * 255), round(result[2] * 255))


def relative_luminance(color: RGB) -> float:
    """Luminancja względna wg WCAG (0 = czerń, 1 = biel)."""
    channels = []
    for raw in color:
        value = raw / 255
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(first: RGB, second: RGB) -> float:
    """Stosunek kontrastu dwóch kolorów wg WCAG (1.0 – 21.0)."""
    lighter = max(relative_luminance(first), relative_luminance(second))
    darker = min(relative_luminance(first), relative_luminance(second))
    return (lighter + 0.05) / (darker + 0.05)


def readable_text_on(background: RGB) -> RGB:
    """Czarny albo biały napis na danym tle — ten o większym kontraście.

    Nie zgadujemy „ciemny kolor = biały napis": jasnozielony i ciemnożółty łamią
    taką regułę. Liczymy kontrast i wybieramy wariant lepszy dla oka.
    """
    white: RGB = (255, 255, 255)
    black: RGB = (24, 24, 27)
    on_white = contrast_ratio(background, white)
    on_black = contrast_ratio(background, black)
    best = white if on_white >= on_black else black
    if max(on_white, on_black) < _MIN_CONTRAST:  # pragma: no cover - skrajne kolory
        logger.info(
            "Kolor %s daje niski kontrast tekstu (%.2f) — napis może być słabo czytelny.",
            to_hex(background),
            max(on_white, on_black),
        )
    return best


# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Palette:
    """Wszystkie kolory interfejsu — gotowe łańcuchy ``#rrggbb``.

    Każde pole jest policzone z akcentu. Widgety dostają wyłącznie te wartości,
    więc nie istnieje element, który po zmianie akcentu zostałby w starym kolorze.
    """

    mode: ThemeMode
    accent: str
    accent_hover: str
    accent_soft: str
    accent_text: str

    background: str
    surface: str
    surface_alt: str
    border: str

    text: str
    text_muted: str
    text_faint: str

    user_bubble: str
    user_text: str
    assistant_bubble: str
    assistant_text: str
    tool_bubble: str
    tool_text: str
    system_bubble: str
    system_text: str
    error_bubble: str
    error_text: str

    # Wskaźnik nasłuchiwania: spoczynek, nasłuch, przetwarzanie mowy.
    listening_idle: str
    listening_active: str
    listening_busy: str

    # Kropki stanu w panelu statusu.
    state_ok: str
    state_busy: str
    state_off: str
    state_error: str

    @property
    def is_dark(self) -> bool:
        return self.mode == "dark"

    def as_dict(self) -> dict[str, str]:
        """Paleta jako słownik — wygodne w testach i w logu diagnostycznym."""
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
            if field != "mode"
        }


def build_palette(accent: str | None = None, mode: ThemeMode = "dark") -> Palette:
    """Zbuduj pełną paletę z akcentu użytkownika.

    Tła są **przybarwione akcentem** (kilka procent domieszki), a nie neutralnie
    szare — inaczej zmiana akcentu byłaby widoczna tylko na przyciskach, a
    użytkownik, który wpisał fioletowy, dostałby fioletowy guzik w szarym oknie.
    """
    base = parse_color(accent) or parse_color(DEFAULT_ACCENT) or (57, 197, 187)
    dark = mode == "dark"
    ground: RGB = _DARK_BASE if dark else _LIGHT_BASE

    # Akcent na ciemnym tle bywa zbyt ciemny, na jasnym — zbyt jaskrawy.
    # Dociągamy go do tła, zamiast wymagać od użytkownika dwóch kolorów.
    accent_rgb = lighten(base, 0.06) if dark and relative_luminance(base) < 0.18 else base
    accent_hover = lighten(accent_rgb, 0.08) if dark else darken(accent_rgb, 0.08)
    accent_soft = mix(ground, accent_rgb, 0.22 if dark else 0.16)

    background = mix(ground, accent_rgb, 0.04)
    surface = mix(ground, accent_rgb, 0.08) if dark else mix(ground, accent_rgb, 0.05)
    surface_alt = mix(ground, accent_rgb, 0.14) if dark else mix(ground, accent_rgb, 0.10)
    border = mix(surface, accent_rgb, 0.28)

    text_base: RGB = (238, 240, 244) if dark else (28, 30, 34)
    text = mix(text_base, accent_rgb, 0.05)
    # Tekst drugoplanowy gaśnie w stronę TŁA, a tło jest już przybarwione
    # akcentem — dzięki temu również napisy pomocnicze idą za kolorem
    # użytkownika, a nie zostają szare przy każdym akcencie.
    text_muted = mix(text_base, background, 0.35)
    text_faint = mix(text_base, background, 0.55)

    # Bąbelek użytkownika nosi pełny akcent, asystenta — przybarwioną
    # powierzchnię. To najczęściej oglądany element interfejsu, więc właśnie na
    # nim zmiana akcentu musi być widoczna od razu.
    user_bubble = accent_rgb
    assistant_bubble = surface_alt
    tool_bubble = mix(surface_alt, accent_rgb, 0.10)
    system_bubble = mix(ground, accent_rgb, 0.06)
    error_base: RGB = (196, 62, 62)
    error_bubble = mix(surface_alt, error_base, 0.35 if dark else 0.18)

    return Palette(
        mode=mode,
        accent=to_hex(accent_rgb),
        accent_hover=to_hex(accent_hover),
        accent_soft=to_hex(accent_soft),
        accent_text=to_hex(readable_text_on(accent_rgb)),
        background=to_hex(background),
        surface=to_hex(surface),
        surface_alt=to_hex(surface_alt),
        border=to_hex(border),
        text=to_hex(text),
        text_muted=to_hex(text_muted),
        text_faint=to_hex(text_faint),
        user_bubble=to_hex(user_bubble),
        user_text=to_hex(readable_text_on(user_bubble)),
        assistant_bubble=to_hex(assistant_bubble),
        assistant_text=to_hex(readable_text_on(assistant_bubble)),
        tool_bubble=to_hex(tool_bubble),
        tool_text=to_hex(mix(readable_text_on(tool_bubble), accent_rgb, 0.35)),
        system_bubble=to_hex(system_bubble),
        system_text=to_hex(text_muted),
        error_bubble=to_hex(error_bubble),
        error_text=to_hex(readable_text_on(error_bubble)),
        listening_idle=to_hex(mix(surface_alt, accent_rgb, 0.25)),
        listening_active=to_hex(saturate(accent_rgb, 1.25)),
        listening_busy=to_hex(mix(accent_rgb, (240, 190, 60), 0.55)),
        state_ok=to_hex(mix(accent_rgb, (110, 210, 140), 0.45)),
        state_busy=to_hex(mix(accent_rgb, (240, 190, 60), 0.45)),
        state_off=to_hex(text_faint),
        state_error=to_hex(mix(error_base, background, 0.15)),
    )


# --------------------------------------------------------------------------- #
# Czcionki i miary
# --------------------------------------------------------------------------- #

# Czcionki są **propozycjami**, nie wymaganiami. Nazwa kroju istnieje tylko na
# części systemów („Segoe UI" na Windowsie, „SF Pro" na macOS-ie), a wpisanie
# nieistniejącej nazwy w Tk kończy się cichym zastępnikiem o innych metrykach.
# Dlatego wybieramy z listy TYLKO to, co system faktycznie zgłasza, a gdy nic nie
# pasuje — oddajemy pusty łańcuch i decyzję zostawiamy toolkitowi.
FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "Inter",
    "Segoe UI Variable",
    "Segoe UI",
    "SF Pro Text",
    "Noto Sans",
    "Cantarell",
    "Ubuntu",
    "DejaVu Sans",
    "Liberation Sans",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
)

MONO_CANDIDATES: Final[tuple[str, ...]] = (
    "JetBrains Mono",
    "Cascadia Mono",
    "Consolas",
    "SF Mono",
    "Noto Sans Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Menlo",
    "Courier New",
)


def pick_font_family(
    available: Iterable[object] = (),
    *,
    preferred: str = "",
    candidates: Iterable[object] | None = None,
) -> str:
    """Wybierz krój dostępny na TEJ maszynie. ``""`` = niech zdecyduje toolkit.

    ``available`` to zwykle wynik ``tkinter.font.families()``. Porównanie ignoruje
    wielkość liter, bo ta sama rodzina bywa raportowana różnie („DejaVu Sans"
    kontra „dejavu sans"). Wpisana przez użytkownika nazwa ma pierwszeństwo, ale
    tylko wtedy, gdy system ją zna — inaczej wracamy do listy kandydatów.
    """
    try:
        families = {str(name).strip().lower(): str(name).strip() for name in available}
    except TypeError:  # pragma: no cover - obrona przed nietypowym argumentem
        families = {}

    wanted = (preferred or "").strip()
    if wanted:
        if not families or wanted.lower() in families:
            return wanted
        logger.info(
            "Czcionka %r nie jest zainstalowana na tej maszynie — wybieram automatycznie.",
            wanted,
        )

    pool = candidates if candidates is not None else FONT_CANDIDATES
    for name in pool:
        match = families.get(str(name).strip().lower())
        if match:
            return match
    return ""


@dataclass(frozen=True, slots=True)
class Metrics:
    """Miary układu w pikselach logicznych (skalowanie robi toolkit)."""

    radius: int = 12
    bubble_radius: int = 14
    gap: int = 10
    pad: int = 14
    sidebar_width: int = 280
    font_size: int = 13
    font_size_small: int = 11
    font_size_title: int = 17
    indicator_size: int = 14


@dataclass(frozen=True, slots=True)
class Theme:
    """Paleta + czcionki + miary. Jedyne źródło wyglądu dla widgetów."""

    palette: Palette
    # Kolor WPISANY przez użytkownika (znormalizowany), a nie ten narysowany.
    # Te dwie wartości potrafią się różnić: bardzo ciemny akcent jest na ciemnym
    # tle rozjaśniany, żeby w ogóle było go widać. Do pytania „czy użytkownik
    # zmienił kolor?" służy właśnie ta wartość — porównywanie z ``palette.accent``
    # dawałoby „zmienił" przy każdym sprawdzeniu.
    source_accent: str = DEFAULT_ACCENT
    font_family: str = ""
    mono_family: str = ""
    metrics: Metrics = Metrics()

    @property
    def mode(self) -> ThemeMode:
        return self.palette.mode

    @property
    def accent(self) -> str:
        """Akcent faktycznie rysowany (po dopasowaniu do jasności tła)."""
        return self.palette.accent

    @classmethod
    def build(
        cls,
        accent: str | None = None,
        *,
        mode: ThemeMode = "dark",
        font_family: str = "",
        mono_family: str = "",
        metrics: Metrics | None = None,
    ) -> Theme:
        return cls(
            palette=build_palette(accent, mode),
            source_accent=normalize_accent(accent),
            font_family=font_family,
            mono_family=mono_family,
            metrics=metrics or Metrics(),
        )

    def with_accent(self, accent: str | None) -> Theme:
        """Nowy motyw z innym akcentem — do natychmiastowego zastosowania zmiany."""
        return Theme(
            palette=build_palette(accent, self.palette.mode),
            source_accent=normalize_accent(accent),
            font_family=self.font_family,
            mono_family=self.mono_family,
            metrics=self.metrics,
        )

    def wants_accent(self, accent: str | None) -> bool:
        """Czy ten motyw jest już zbudowany z podanego koloru?"""
        return normalize_accent(accent) == self.source_accent

    def with_mode(self, mode: ThemeMode) -> Theme:
        return Theme(
            palette=build_palette(self.source_accent, mode),
            source_accent=self.source_accent,
            font_family=self.font_family,
            mono_family=self.mono_family,
            metrics=self.metrics,
        )


__all__ = [
    "DEFAULT_ACCENT",
    "FONT_CANDIDATES",
    "MONO_CANDIDATES",
    "Metrics",
    "Palette",
    "Theme",
    "ThemeMode",
    "build_palette",
    "contrast_ratio",
    "darken",
    "lighten",
    "mix",
    "normalize_accent",
    "parse_color",
    "pick_font_family",
    "readable_text_on",
    "relative_luminance",
    "saturate",
    "to_hex",
]
