"""Panel ustawień — logika bez ani jednego widgetu (Faza 10).

Co tu jest: definicja pól, walidacja, budowa zapisu do
``config/user_settings.json`` przez :func:`config.save_user_settings`, rozpoznanie
zmian wymagających przeładowania mowy oraz opis okna wyboru pliku (co pokazać,
od jakiego katalogu zacząć). Czego tu nie ma: rysowania. Dzięki temu najbardziej
awaryjna część panelu — „czy zapis nie zgubi pól, których ten ekran nie edytuje" —
jest sprawdzalna testem bez ekranu.

Dwie zasady, których pilnuje ten moduł:

* **zapis idzie do ``user_settings.json``, nigdy do ``.env``** — plik ``.env``
  opisuje infrastrukturę (adresy, urządzenia, limity), a nie preferencje;
* **scalanie zamiast nadpisywania** — panel zapisuje wyłącznie pola, które
  faktycznie zmieniono, więc ustawienia nieobecne na ekranie (wake_word,
  piper_voices, speech_language...) zostają nietknięte.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from config import (
    MODELS_DIR,
    PROJECT_ROOT,
    ConfigError,
    UserSettings,
    get_user_settings,
    home_directory,
    save_user_settings,
)
from gui.theme import normalize_accent, parse_color
from i18n import t

logger = logging.getLogger(__name__)

FieldKind = Literal["text", "color", "multiline", "choice", "path", "int", "float", "switch"]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Opis jednego pola panelu ustawień.

    ``key`` jest ścieżką w ``user_settings.json`` z kropką dla zagnieżdżeń
    (``rvc.model_path``) — ta sama nazwa służy do odczytu, walidacji i zapisu, więc
    nie ma jak pomylić pola z ekranu z polem w pliku.

    Napisy trzymamy jako **klucze katalogu tekstów**, nie jako gotowe zdania:
    panel ustawień pokazuje je w języku interfejsu, a ten bywa zmieniony między
    uruchomieniami. ``label`` i ``help`` tłumaczą się w chwili odczytu.
    """

    key: str
    label_key: str
    kind: FieldKind
    section: str = "assistant"
    help_key: str = ""
    # Czy zmiana działa od razu, czy wymaga przeładowania silnika mowy. To nie
    # kosmetyka: użytkownik musi wiedzieć, dlaczego nowy głos jeszcze nie brzmi.
    live: bool = True
    reload_tts: bool = False
    choices_source: str = ""
    # Filtry okna wyboru pliku: (klucz nazwy w katalogu, wzorzec).
    file_filter_keys: tuple[tuple[str, str], ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    placeholder_key: str = ""

    @property
    def label(self) -> str:
        return t(self.label_key)

    @property
    def help(self) -> str:
        return t(self.help_key) if self.help_key else ""

    @property
    def placeholder(self) -> str:
        return t(self.placeholder_key) if self.placeholder_key else ""

    @property
    def file_filter(self) -> tuple[tuple[str, str], ...]:
        """Filtry w języku interfejsu — nazwy tłumaczone, wzorce bez zmian."""
        return tuple((t(name_key), pattern) for name_key, pattern in self.file_filter_keys)

    @property
    def is_nested(self) -> bool:
        return "." in self.key

    @property
    def is_file(self) -> bool:
        return self.kind == "path"


# Kolejność pól = kolejność na ekranie. „Filtry" plików są PODPOWIEDZIĄ okna
# systemowego, nie warunkiem: różne widelce RVC używają różnych rozszerzeń, więc
# zawsze zostaje pozycja „wszystkie pliki".
FORM_FIELDS: Final[tuple[FieldSpec, ...]] = (
    FieldSpec(
        key="assistant_name",
        label_key="settings.field.assistant_name",
        kind="text",
        section="assistant",
        help_key="settings.help.assistant_name",
        placeholder_key="settings.placeholder.assistant_name",
    ),
    FieldSpec(
        key="ui_accent_color",
        label_key="settings.field.ui_accent_color",
        kind="color",
        section="assistant",
        help_key="settings.help.ui_accent_color",
    ),
    FieldSpec(
        key="personality_traits",
        label_key="settings.field.personality_traits",
        kind="multiline",
        section="assistant",
        help_key="settings.help.personality_traits",
        placeholder_key="settings.placeholder.personality_traits",
    ),
    FieldSpec(
        key="voice_engine",
        label_key="settings.field.voice_engine",
        kind="choice",
        section="voice",
        choices_source="tts_engines",
        help_key="settings.help.voice_engine",
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        key="piper_model",
        label_key="settings.field.piper_model",
        kind="choice",
        section="voice",
        choices_source="piper_voices",
        help_key="settings.help.piper_model",
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        # Język mowy bywa INNY niż język odpowiedzi: ktoś woli angielski
        # interfejs, a mówi po polsku. Bez tego pola jedyną drogą było ręczne
        # dopisanie klucza do pliku ustawień — czyli w praktyce nikt tego nie robił.
        key="speech_language",
        label_key="settings.field.speech_language",
        kind="text",
        section="voice",
        help_key="settings.help.speech_language",
        placeholder_key="settings.placeholder.speech_language",
    ),
    FieldSpec(
        key="rvc.enabled",
        label_key="settings.field.rvc_enabled",
        kind="switch",
        section="rvc",
        help_key="settings.help.rvc_enabled",
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        key="rvc.model_path",
        label_key="settings.field.rvc_model_path",
        kind="path",
        section="rvc",
        help_key="settings.help.rvc_model_path",
        file_filter_keys=(
            ("settings.filter.rvc_model", "*.pth"),
            ("settings.filter.all_files", "*"),
        ),
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        key="rvc.index_path",
        label_key="settings.field.rvc_index_path",
        kind="path",
        section="rvc",
        help_key="settings.help.rvc_index_path",
        file_filter_keys=(
            ("settings.filter.rvc_index", "*.index"),
            ("settings.filter.all_files", "*"),
        ),
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        key="rvc.pitch_shift",
        label_key="settings.field.rvc_pitch_shift",
        kind="int",
        section="rvc",
        minimum=-24,
        maximum=24,
        step=1,
        help_key="settings.help.rvc_pitch_shift",
        live=False,
        reload_tts=True,
    ),
    FieldSpec(
        key="rvc.index_rate",
        label_key="settings.field.rvc_index_rate",
        kind="float",
        section="rvc",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        help_key="settings.help.rvc_index_rate",
        live=False,
        reload_tts=True,
    ),
)


def section_title(section: str) -> str:
    """Nagłówek sekcji w języku interfejsu."""
    return t(f"settings.section.{section}")

# Pola, których zmiana działa natychmiast — bez restartu i bez przeładowania
# czegokolwiek. Zgodne z wymaganiem Fazy 10 i sprawdzane testem.
LIVE_KEYS: Final[frozenset[str]] = frozenset(
    # Język mowy jest czytany przy KAŻDEJ transkrypcji, więc zmiana działa od
    # następnej wypowiedzi — bez restartu i bez przeładowania mowy.
    {"assistant_name", "ui_accent_color", "personality_traits", "speech_language"}
)


def field_by_key(key: str) -> FieldSpec | None:
    for spec in FORM_FIELDS:
        if spec.key == key:
            return spec
    return None


def sections() -> tuple[tuple[str, tuple[FieldSpec, ...]], ...]:
    """Pola pogrupowane w sekcje, w kolejności z :data:`FORM_FIELDS`."""
    order: list[str] = []
    grouped: dict[str, list[FieldSpec]] = {}
    for spec in FORM_FIELDS:
        if spec.section not in grouped:
            grouped[spec.section] = []
            order.append(spec.section)
        grouped[spec.section].append(spec)
    return tuple((name, tuple(grouped[name])) for name in order)


# --------------------------------------------------------------------------- #
# Odczyt i zapis wartości
# --------------------------------------------------------------------------- #


def read_value(user_settings: UserSettings, key: str) -> Any:
    """Wartość pola z ustawień — obsługuje zagnieżdżenia (``rvc.pitch_shift``)."""
    target: Any = user_settings
    for part in key.split("."):
        target = getattr(target, part, None)
        if target is None and part != key.rsplit(".", maxsplit=1)[-1]:
            return None
    return target


def current_values(user_settings: UserSettings | None = None) -> dict[str, Any]:
    """Bieżące wartości wszystkich pól panelu."""
    settings = user_settings if user_settings is not None else get_user_settings()
    return {spec.key: read_value(settings, spec.key) for spec in FORM_FIELDS}


def build_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    """Zamień płaskie ``{"rvc.pitch_shift": 2}`` na zagnieżdżone ``{"rvc": {...}}``.

    Zapis zawiera **tylko** przekazane klucze — resztę pliku zachowuje
    :func:`config.save_user_settings` (scala, nie nadpisuje).
    """
    payload: dict[str, Any] = {}
    for key, value in values.items():
        head, _, tail = key.partition(".")
        if tail:
            nested = payload.setdefault(head, {})
            if isinstance(nested, dict):
                nested[tail] = value
        else:
            payload[key] = value
    return payload


# --------------------------------------------------------------------------- #
# Normalizacja tego, co przyszło z widgetów
# --------------------------------------------------------------------------- #


def relativize_path(raw: str, *, root: Path | None = None) -> str:
    """Ścieżka pliku zapisana tak, żeby przetrwała przeniesienie projektu.

    Plik leżący W projekcie zapisujemy ścieżką względną z ukośnikami w przód
    (``models/rvc/glos.pth``): ten sam plik po skopiowaniu katalogu na inny
    komputer — także na inny system — nadal się znajduje. Plik spoza projektu
    zostaje ścieżką bezwzględną, bo nic innego nie miałoby sensu.
    """
    text = (raw or "").strip().strip('"')
    if not text:
        return ""
    candidate = Path(text).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - zależne od systemu plików
        return text
    try:
        inside = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(resolved)
    return inside.as_posix()


def coerce(spec: FieldSpec, raw: Any) -> Any:
    """Wartość z widgetu → wartość do zapisu. Nie rzuca: błędy zgłasza walidacja."""
    if spec.kind == "switch":
        return bool(raw)
    if spec.kind == "int":
        try:
            whole = round(float(str(raw).replace(",", ".").strip() or 0))
        except (TypeError, ValueError):
            return raw
        return int(_clamp(float(whole), spec))
    if spec.kind == "float":
        try:
            fraction = float(str(raw).replace(",", ".").strip() or 0.0)
        except (TypeError, ValueError):
            return raw
        return round(_clamp(fraction, spec), 4)
    if spec.kind == "path":
        return relativize_path(str(raw or ""))
    if spec.kind == "color":
        text = str(raw or "").strip()
        # Kolor bez kratki („39C5BB") to najczęstsza pomyłka przy wklejaniu —
        # przyjmujemy, bo intencja jest jednoznaczna.
        if text and not text.startswith("#") and parse_color(text) is not None:
            text = f"#{text}"
        return text
    if spec.kind == "multiline":
        return str(raw or "").strip()
    return str(raw or "").strip()


def _clamp(value: float, spec: FieldSpec) -> float:
    if spec.minimum is not None:
        value = max(spec.minimum, value)
    if spec.maximum is not None:
        value = min(spec.maximum, value)
    return value


# --------------------------------------------------------------------------- #
# Walidacja i ostrzeżenia
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FieldProblem:
    """Problem z jednym polem. ``blocking=False`` znaczy „zapiszę, ale uprzedzam"."""

    key: str
    message: str
    blocking: bool = True


def resolve_declared_path(raw: str) -> Path | None:
    """Ścieżka z pliku ustawień → miejsce na dysku (tak samo jak w ``config.py``)."""
    text = (raw or "").strip()
    if not text:
        return None
    import os

    candidate = Path(os.path.expandvars(text)).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def validate(
    values: Mapping[str, Any], *, base: UserSettings | None = None
) -> list[FieldProblem]:
    """Sprawdź wartości panelu. Lista pusta = można zapisywać.

    Walidację merytoryczną prowadzi ten sam model pydantic, który czyta plik
    ustawień — panel nie ma własnego zestawu reguł, który mógłby się z nim
    rozjechać. Do tego dochodzą ostrzeżenia niedostępne modelowi: „wskazanego
    pliku nie ma na dysku" jest prawdą tylko na TEJ maszynie.
    """
    problems: list[FieldProblem] = []
    settings = base if base is not None else get_user_settings()

    merged = settings.model_dump()
    for key, value in build_payload(values).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    try:
        UserSettings.model_validate(merged)
    except Exception as exc:  # pydantic ValidationError (albo dowolny inny)
        for key, message in _field_errors(exc):
            problems.append(FieldProblem(key=key, message=message))
        if not problems:
            problems.append(FieldProblem(key="", message=str(exc)))

    # Kolor: pydantic sprawdza wzorzec, ale komunikat („string does not match
    # regex") nie mówi człowiekowi nic. Podmieniamy go na zdanie po polsku.
    accent = values.get("ui_accent_color")
    if accent is not None and parse_color(str(accent)) is None:
        problems = [item for item in problems if item.key != "ui_accent_color"]
        problems.append(
            FieldProblem(key="ui_accent_color", message=t("settings.problem.color"))
        )

    for spec in FORM_FIELDS:
        if not spec.is_file or spec.key not in values:
            continue
        raw = str(values.get(spec.key) or "").strip()
        if not raw:
            continue
        path = resolve_declared_path(raw)
        if path is None or not path.is_file():
            problems.append(
                FieldProblem(
                    key=spec.key,
                    message=t("settings.problem.missing_file", path=path or raw),
                    blocking=False,
                )
            )
            continue
        expected = _expected_suffix(spec)
        if expected and path.suffix.lower() != expected:
            problems.append(
                FieldProblem(
                    key=spec.key,
                    message=t("settings.problem.wrong_suffix", suffix=expected),
                    blocking=False,
                )
            )

    if bool(values.get("rvc.enabled")) and not str(values.get("rvc.model_path") or "").strip():
        problems.append(
            FieldProblem(
                key="rvc.model_path",
                message=t("settings.problem.rvc_without_model"),
                blocking=False,
            )
        )
    return problems


def _expected_suffix(spec: FieldSpec) -> str:
    for _, pattern in spec.file_filter_keys:
        if pattern.startswith("*.") and pattern != "*":
            return pattern[1:].lower()
    return ""


def _field_errors(exc: Exception) -> list[tuple[str, str]]:
    """Wyciągnij (pole, komunikat) z błędu walidacji pydantica."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return []
    result: list[tuple[str, str]] = []
    try:
        for item in errors():
            location = ".".join(str(part) for part in item.get("loc", ()))
            message = str(item.get("msg", "nieprawidłowa wartość"))
            result.append((location, message))
    except Exception:  # pragma: no cover - nietypowy wyjątek
        return []
    return result


# --------------------------------------------------------------------------- #
# Wynik zapisu
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Co się stało po naciśnięciu „Zapisz"."""

    ok: bool
    settings: UserSettings | None = None
    changed: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    problems: tuple[FieldProblem, ...] = ()
    error: str = ""

    @property
    def needs_tts_reload(self) -> bool:
        """Czy trzeba zbudować silnik mowy od nowa, żeby zmiana zabrzmiała?"""
        return any(
            (spec := field_by_key(key)) is not None and spec.reload_tts for key in self.changed
        )

    @property
    def applied_live(self) -> tuple[str, ...]:
        """Zmiany, które zadziałały od razu (imię, kolor, cechy charakteru)."""
        return tuple(key for key in self.changed if key in LIVE_KEYS)

    def message(self) -> str:
        """Jedno zdanie dla użytkownika — dokładnie to, co się stało."""
        if not self.ok:
            return self.error or t("settings.result.not_saved")
        if not self.changed:
            return t("settings.result.nothing")
        labels = [
            (spec.label if (spec := field_by_key(key)) is not None else key)
            for key in self.changed
        ]
        listing = ", ".join(labels)
        if self.needs_tts_reload:
            return t("settings.result.saved_reload", fields=listing)
        return t("settings.result.saved", fields=listing)


class SettingsForm:
    """Stan panelu ustawień: wartości wyjściowe, zmiany i zapis.

    Formularz **nie zapisuje niczego samoczynnie** — dopóki nie padnie
    :meth:`save`, plik ustawień pozostaje nietknięty. Panel może więc pokazywać
    ostrzeżenia w trakcie edycji bez ryzyka, że w połowie wpisywania koloru
    zapisze się „#39C".
    """

    def __init__(
        self,
        user_settings: UserSettings | None = None,
        *,
        path: Path | None = None,
    ) -> None:
        self._base = user_settings if user_settings is not None else get_user_settings()
        self._path = path
        self._values: dict[str, Any] = current_values(self._base)

    @property
    def base(self) -> UserSettings:
        return self._base

    @property
    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def value(self, key: str) -> Any:
        return self._values.get(key)

    def set(self, key: str, raw: Any) -> Any:
        """Zapamiętaj wartość z widgetu (po normalizacji). Zwraca to, co zapamiętano."""
        spec = field_by_key(key)
        value = coerce(spec, raw) if spec is not None else raw
        self._values[key] = value
        return value

    def update(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            self.set(key, value)

    def changed_keys(self) -> tuple[str, ...]:
        """Pola różniące się od stanu wyjściowego — tylko te pójdą do zapisu."""
        original = current_values(self._base)
        changed: list[str] = []
        for spec in FORM_FIELDS:
            before = original.get(spec.key)
            after = self._values.get(spec.key)
            if isinstance(before, float) or isinstance(after, float):
                try:
                    if abs(float(before or 0.0) - float(after or 0.0)) > 1e-6:
                        changed.append(spec.key)
                    continue
                except (TypeError, ValueError):
                    pass
            if str(before if before is not None else "") != str(
                after if after is not None else ""
            ):
                changed.append(spec.key)
        return tuple(changed)

    def problems(self) -> tuple[FieldProblem, ...]:
        return tuple(validate(self._values, base=self._base))

    def preview_accent(self) -> str:
        """Kolor akcentu do natychmiastowego podglądu (zawsze poprawny)."""
        return normalize_accent(str(self._values.get("ui_accent_color") or ""))

    def revert(self) -> None:
        self._values = current_values(self._base)

    def save(self) -> SaveResult:
        """Zapisz zmienione pola do ``config/user_settings.json``.

        Zapis idzie przez :func:`config.save_user_settings`, więc pola nieedytowane
        na tym ekranie zostają w pliku bez zmian — razem z kluczami, których model
        nawet nie zna.
        """
        problems = self.problems()
        blocking = tuple(item for item in problems if item.blocking)
        warnings = tuple(item.message for item in problems if not item.blocking)
        if blocking:
            return SaveResult(ok=False, problems=problems, error=blocking[0].message)

        changed = self.changed_keys()
        if not changed:
            return SaveResult(ok=True, settings=self._base, changed=(), warnings=warnings)

        payload = build_payload({key: self._values[key] for key in changed})
        try:
            saved = save_user_settings(payload, self._path)
        except ConfigError as exc:
            logger.warning("Nie zapisano ustawień z panelu GUI: %s", exc.message)
            return SaveResult(ok=False, problems=problems, error=exc.user_message)
        except OSError as exc:  # pragma: no cover - zależne od uprawnień
            return SaveResult(
                ok=False, problems=problems, error=t("settings.problem.write_failed", error=exc)
            )

        self._base = saved
        self._values = current_values(saved)
        return SaveResult(
            ok=True, settings=saved, changed=changed, warnings=warnings, problems=problems
        )


# --------------------------------------------------------------------------- #
# Okno wyboru pliku
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FileRequest:
    """Czego panel chce od systemowego okna wyboru pliku."""

    key: str
    title: str
    filetypes: tuple[tuple[str, str], ...] = ()
    initial_dir: str = ""
    initial_file: str = ""

    def as_dialog_kwargs(self) -> dict[str, Any]:
        """Argumenty dla ``tkinter.filedialog.askopenfilename``.

        Puste wartości są POMIJANE, a nie przekazywane jako ``""``: okno bez
        ``initialdir`` startuje w miejscu wybranym przez system, co na obcej
        maszynie jest lepszym punktem wyjścia niż katalog, którego tam nie ma.
        """
        kwargs: dict[str, Any] = {"title": self.title}
        if self.filetypes:
            kwargs["filetypes"] = [tuple(item) for item in self.filetypes]
        if self.initial_dir:
            kwargs["initialdir"] = self.initial_dir
        if self.initial_file:
            kwargs["initialfile"] = self.initial_file
        return kwargs


def file_request(key: str, current: str = "") -> FileRequest:
    """Zbuduj opis okna wyboru pliku dla danego pola.

    Katalog startowy wybieramy po kolei: katalog obecnie wskazanego pliku, potem
    ``models/`` w projekcie, potem katalog domowy — i tylko jeśli takie miejsce
    naprawdę istnieje. Żadna ścieżka nie jest wpisana w kod: wszystkie pochodzą z
    ``config.py``, który zna tę maszynę.
    """
    spec = field_by_key(key)
    label = spec.label if spec is not None else key
    filters = spec.file_filter if spec is not None else ()

    initial_dir = ""
    initial_file = ""
    existing = resolve_declared_path(current)
    if existing is not None:
        initial_file = existing.name
        parent = existing.parent
        if parent.is_dir():
            initial_dir = str(parent)

    if not initial_dir:
        for candidate in _default_directories():
            if candidate is not None and candidate.is_dir():
                initial_dir = str(candidate)
                break

    return FileRequest(
        key=key,
        title=t("settings.dialog_title", label=label),
        filetypes=filters,
        initial_dir=initial_dir,
        initial_file=initial_file,
    )


def _default_directories() -> Sequence[Path | None]:
    # models/rvc jest tylko SUGESTIĄ miejsca na modele głosu — katalog nie musi
    # istnieć i nic go nie tworzy na siłę.
    return (MODELS_DIR / "rvc", MODELS_DIR, home_directory())


@dataclass(frozen=True, slots=True)
class ChoiceOptions:
    """Listy wyboru wypełniane przez panel (głosy Pipera, silniki mowy).

    Panel dostaje je z warstwy audio dopiero w chwili otwarcia ekranu — ten moduł
    nie importuje niczego, co ładuje modele.
    """

    piper_voices: tuple[str, ...] = ()
    tts_engines: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)

    def values_for(self, spec: FieldSpec) -> tuple[str, ...]:
        if spec.choices_source == "piper_voices":
            # Puste znaczy „dobierz do języka" — to poprawny wybór, więc musi być
            # na liście, a nie tylko możliwy przez wyczyszczenie pola.
            return ("", *self.piper_voices)
        if spec.choices_source == "tts_engines":
            return self.tts_engines or ("piper", "none")
        return ()

    def label_for(self, value: str) -> str:
        if not value:
            return t("settings.auto_voice")
        return str(self.labels.get(value, value))


__all__ = [
    "FORM_FIELDS",
    "LIVE_KEYS",
    "ChoiceOptions",
    "FieldKind",
    "FieldProblem",
    "FieldSpec",
    "FileRequest",
    "SaveResult",
    "SettingsForm",
    "build_payload",
    "coerce",
    "current_values",
    "field_by_key",
    "file_request",
    "read_value",
    "relativize_path",
    "resolve_declared_path",
    "section_title",
    "sections",
    "validate",
]
