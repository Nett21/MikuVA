"""Repozytorium nadaje się do opublikowania.

Ten plik pilnuje rzeczy, których nie da się cofnąć. Sekret wypchnięty do
publicznego repozytorium zostaje w historii gita **na zawsze** — usunięcie go
kolejnym commitem niczego nie naprawia, a przepisanie historii nie dotyczy
kopii, które ktoś zdążył sklonować, ani cache'u GitHuba. Jedyną skuteczną
reakcją jest unieważnienie klucza.

Dlatego testy sprawdzają nie „czy plik z sekretem jest ignorowany", ale
**czy w plikach przeznaczonych do publikacji nie ma czego szukać**.

Reguły ``.gitignore`` interpretujemy tu w uproszczeniu (nazwy katalogów
i rozszerzenia), bo git nie musi być zainstalowany na maszynie, na której lecą
testy. Uproszczenie działa na korzyść bezpieczeństwa: skanujemy WIĘCEJ plików,
niż trafiłoby do repozytorium, nigdy mniej.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from config import PROJECT_ROOT

GITIGNORE = PROJECT_ROOT / ".gitignore"

# Katalogi, których zawartość nigdy nie trafia do repozytorium.
POMIJANE_KATALOGI = frozenset({
    ".git", ".venv", "venv", "env", "ENV", "__pycache__", "models", "vendor",
    "logs", "data", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
    ".vscode", "build", "dist", "node_modules",
})

# Pliki lokalne — istnieją na maszynie, ale są w .gitignore.
POMIJANE_PLIKI = frozenset({
    ".env",
    "config/user_settings.json",
    "config/dependency_status.json",
    "config/offline_bundle.json",
    ".claude/settings.local.json",
})

POMIJANE_ROZSZERZENIA = frozenset({
    ".pyc", ".pyo", ".log", ".db", ".sqlite", ".sqlite3",
    ".onnx", ".pth", ".gguf", ".safetensors", ".ckpt",
})


def pliki_do_publikacji() -> Iterator[tuple[str, Path]]:
    """Pliki, które trafiłyby do repozytorium (z zapasem — patrz docstring modułu)."""
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in POMIJANE_KATALOGI for part in rel.parts):
            continue
        nazwa = str(rel).replace("\\", "/")
        if nazwa in POMIJANE_PLIKI or path.suffix in POMIJANE_ROZSZERZENIA:
            continue
        yield nazwa, path


def tekst(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None


@pytest.fixture(scope="module")
def publikowane() -> list[tuple[str, str]]:
    wynik = []
    for nazwa, path in pliki_do_publikacji():
        zawartosc = tekst(path)
        if zawartosc is not None:
            wynik.append((nazwa, zawartosc))
    assert len(wynik) > 50, "skan nie znalazł plików projektu — sprawdź reguły pomijania"
    return wynik


# --------------------------------------------------------------------------- #
# .gitignore
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "wzorzec",
    [
        ".env",                        # klucze API i ustawienia maszyny
        "config/user_settings.json",   # imię, głos, ścieżki do modeli RVC
        ".claude/settings.local.json",  # osobiste ustawienia narzędzi
        "models/",                     # setki MB do kilku GB
        "logs/",
        "__pycache__/",
        ".venv/",
        "venv/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "*.py[cod]",
        "*.sqlite3",
        "*.db",
        "*.log",
        "*.tmp",
        "*.onnx",                      # gdyby model trafił poza models/
        "*.pth",
    ],
)
def test_gitignore_blokuje(wzorzec: str) -> None:
    linie = {
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert wzorzec in linie, f".gitignore nie blokuje {wzorzec}"


def test_pliki_wzorcowe_nie_sa_ignorowane() -> None:
    """`*.example` MUSZĄ trafić do repozytorium — bez nich nikt nie skonfiguruje projektu."""
    linie = {
        line.strip()
        for line in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for potrzebny in (".env.example", "config/user_settings.example.json"):
        assert potrzebny not in linie
        assert (PROJECT_ROOT / potrzebny).is_file(), f"brak pliku {potrzebny}"


# --------------------------------------------------------------------------- #
# Sekrety
# --------------------------------------------------------------------------- #

# Wzorce kluczy o rozpoznawalnym kształcie. Nie łapią wszystkiego — łapią to,
# co realnie wycieka: klucze skopiowane z panelu dostawcy prosto do kodu.
WZORCE_KLUCZY = (
    ("klucz OpenAI/Anthropic", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("token GitHuba", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}")),
    ("token GitHuba (fine-grained)", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}")),
    ("token Slacka", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("klucz Google", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("klucz AWS", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("klucz prywatny", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
)


def test_zaden_publikowany_plik_nie_zawiera_klucza(publikowane: list[tuple[str, str]]) -> None:
    trafienia = []
    for nazwa, zawartosc in publikowane:
        if nazwa == "tests/test_repository.py":
            continue  # ten plik zawiera same WZORCE, nie klucze
        for opis, wzorzec in WZORCE_KLUCZY:
            dopasowanie = wzorzec.search(zawartosc)
            if dopasowanie:
                trafienia.append(f"{nazwa}: {opis} ({dopasowanie.group(0)[:24]}…)")
    assert not trafienia, "znaleziono sekrety w plikach do publikacji:\n" + "\n".join(trafienia)


def test_zadne_pole_klucza_nie_ma_wpisanej_wartosci(publikowane: list[tuple[str, str]]) -> None:
    """`API_KEY = "cokolwiek"` w kodzie to sekret, nawet jeśli akurat nie wygląda na klucz."""
    przypisanie = re.compile(
        r"(?i)\b(api_?key|access_?token|auth_?token|client_?secret|password)\s*[=:]\s*"
        r"['\"]([^'\"\s]{10,})['\"]"
    )
    # Wartości oczywiście udawane — używają ich testy, żeby sprawdzić zamazywanie.
    udawane = re.compile(r"(?i)sekret|fake|dummy|test|przyklad|przykład|example|xxx|token-|<.*>")

    trafienia = []
    for nazwa, zawartosc in publikowane:
        if nazwa == "tests/test_repository.py":
            continue
        for dopasowanie in przypisanie.finditer(zawartosc):
            wartosc = dopasowanie.group(2)
            if udawane.search(wartosc):
                continue
            trafienia.append(f"{nazwa}: {dopasowanie.group(1)}={wartosc[:20]}…")
    assert not trafienia, "wygląda na wpisane sekrety:\n" + "\n".join(trafienia)


# --------------------------------------------------------------------------- #
# Prywatne ścieżki
# --------------------------------------------------------------------------- #


def test_zaden_publikowany_plik_nie_zawiera_katalogu_domowego_autora(
    publikowane: list[tuple[str, str]],
) -> None:
    """Ścieżka z nazwą konta autora nie zadziała u nikogo innego — i mówi, kim jest autor.

    Dopuszczone są nazwy oczywiście zastępcze (`/home/uzytkownik`, `/Users/you`),
    bo pojawiają się w przykładach i w danych testowych.
    """
    zastepcze = frozenset({
        "uzytkownik", "użytkownik", "user", "users", "you", "ktos", "ktoś",
        "twoj", "twój", "nazwa", "username", "downloads", "me",
    })
    wzorzec = re.compile(r"/(?:home|Users)/([A-Za-z][A-Za-z0-9_.-]*)")

    trafienia = []
    for nazwa, zawartosc in publikowane:
        if nazwa == "tests/test_repository.py":
            continue
        for dopasowanie in wzorzec.finditer(zawartosc):
            konto = dopasowanie.group(1)
            if konto.lower() in zastepcze:
                continue
            trafienia.append(f"{nazwa}: {dopasowanie.group(0)}")
    assert not trafienia, (
        "ścieżki z konkretnej maszyny w plikach do publikacji:\n"
        + "\n".join(sorted(set(trafienia)))
    )


def test_zadna_sciezka_do_modelu_rvc_nie_jest_zaszyta(publikowane: list[tuple[str, str]]) -> None:
    """Modele RVC są prywatne i prawnie obciążone (patrz sekcja Ograniczeń w README).

    W repozytorium wolno mieć wyłącznie FILTRY rozszerzeń (`*.pth`) i puste pola
    konfiguracji — nigdy ścieżki do konkretnego pliku.
    """
    wzorzec = re.compile(r"['\"]([^'\"\n]*/[^'\"\n]*\.(?:pth|index))['\"]")
    trafienia = []
    for nazwa, zawartosc in publikowane:
        if nazwa.startswith("tests/") or nazwa == "tests/test_repository.py":
            continue  # testy używają ścieżek tymczasowych z tmp_path
        for dopasowanie in wzorzec.finditer(zawartosc):
            trafienia.append(f"{nazwa}: {dopasowanie.group(1)}")
    assert not trafienia, "zaszyte ścieżki do modeli:\n" + "\n".join(trafienia)


# --------------------------------------------------------------------------- #
# Pliki, których publikacja wymaga
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "nazwa",
    [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        ".gitignore",
        ".env.example",
        "config/user_settings.example.json",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".github/workflows/tests.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
    ],
)
def test_plik_wymagany_do_publikacji_istnieje(nazwa: str) -> None:
    path = PROJECT_ROOT / nazwa
    assert path.is_file(), f"brak {nazwa}"
    assert path.stat().st_size > 0, f"{nazwa} jest pusty"
