"""Dokumentacja musi opisywać ten kod, a nie poprzedni.

README podaje konkretne wartości domyślne, nazwy ustawień i nazwy narzędzi.
Każda z nich to obietnica, którą łatwo złamać jedną zmianą w ``config.py`` —
a rozjazd między dokumentacją a kodem jest gorszy niż brak dokumentacji, bo
czytelnik nie ma jak się zorientować, że został wprowadzony w błąd.

Te testy nie sprawdzają stylu ani kompletności. Sprawdzają wyłącznie rzeczy
weryfikowalne maszynowo: czy wartość podana w tekście naprawdę jest domyślna,
czy opisane ustawienie istnieje, czy odnośnik prowadzi do istniejącego miejsca.
"""

from __future__ import annotations

import re

import pytest

from config import PROJECT_ROOT, Settings

README = PROJECT_ROOT / "README.md"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def defaults() -> Settings:
    return Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# Wartości domyślne wypisane w README
# --------------------------------------------------------------------------- #

# Pole ustawień → wartość, którą README podaje jako domyślną.
DOCUMENTED: dict[str, object] = {
    "ollama_model": "qwen2.5:7b-instruct",
    "ollama_num_ctx": 8192,
    "ollama_temperature": 0.7,
    "ollama_max_tokens": 1024,
    "ollama_keep_alive": "10m",
    "ollama_read_timeout": 120.0,
    "whisper_model": "small",
    "whisper_beam_size": 5,
    "whisper_idle_unload_s": 300.0,
    "wake_similarity": 0.72,
    "wake_window_s": 30.0,
    "wake_whisper_model": "base",
    "history_max_messages": 40,
    "history_max_chars": 12_000,
    "llm_history_max_messages": 16,
    "llm_history_max_chars": 6_000,
    "memory_trim_ratio": 0.75,
    "memory_recall_limit": 5,
    "memory_recall_min_score": 0.35,
    "memory_retention_days": 0,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "tools_max_calls_per_turn": 6,
    "tool_result_max_chars": 4_000,
    "security_require_confirm_from": "HIGH",
    "security_allow_critical": False,
    "security_confirm_timeout_s": 60.0,
    "web_max_chars": 6_000,
    "web_max_redirects": 3,
    "web_allow_private_hosts": False,
    "headless_ollama_wait_s": 60.0,
    "headless_retry_s": 15.0,
    "headless_listen_slice_s": 5.0,
    "headless_greeting": False,
    "tts_min_sentence_chars": 24,
    "tts_max_sentence_chars": 320,
    "vad_listen_timeout_s": 30.0,
    "vad_preroll_ms": 300,
}


@pytest.mark.parametrize(("pole", "wartosc"), sorted(DOCUMENTED.items()))
def test_wartosc_z_readme_jest_faktycznie_domyslna(
    defaults: Settings, pole: str, wartosc: object
) -> None:
    assert hasattr(defaults, pole), f"README opisuje nieistniejące ustawienie {pole}"
    assert getattr(defaults, pole) == wartosc, (
        f"README podaje dla {pole.upper()} wartość {wartosc!r}, a kod ma "
        f"{getattr(defaults, pole)!r} — popraw jedno albo drugie"
    )


def test_narzedzia_domyslnie_wylaczone_naprawde_sa_puste(defaults: Settings) -> None:
    """README obiecuje, że ``shell.run`` i dostęp do dysku są domyślnie zamknięte."""
    assert defaults.shell_allowed_binaries == ""
    assert defaults.fs_allowed_roots == ""


# --------------------------------------------------------------------------- #
# Nazwy ustawień wymienione w README i w .env.example
# --------------------------------------------------------------------------- #


def _env_keys(text: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]{2,})=", text, re.MULTILINE)}


def test_kazdy_klucz_z_env_example_istnieje_w_ustawieniach(defaults: Settings) -> None:
    """Wpis w ``.env.example``, którego nie zna ``Settings``, nic nie robi.

    Pydantic-settings po cichu ignoruje nieznane zmienne, więc literówka
    w nazwie kończy się ustawieniem, które wygląda na działające i nie działa.
    """
    znane = set(type(defaults).model_fields)
    nieznane = sorted(k for k in _env_keys(ENV_EXAMPLE.read_text(encoding="utf-8"))
                      if k.lower() not in znane)
    assert not nieznane, f".env.example opisuje nieistniejące ustawienia: {nieznane}"


def test_nowe_ustawienia_sa_opisane_w_env_example() -> None:
    """Ustawienie bez wpisu w ``.env.example`` jest praktycznie nie do znalezienia."""
    tekst = ENV_EXAMPLE.read_text(encoding="utf-8")
    for klucz in (
        "WHISPER_IDLE_UNLOAD_S",
        "LLM_HISTORY_MAX_MESSAGES",
        "LLM_HISTORY_MAX_CHARS",
        "HEADLESS_OLLAMA_WAIT_S",
        "HEADLESS_RETRY_S",
        "HEADLESS_LISTEN_SLICE_S",
        "HEADLESS_GREETING",
    ):
        assert f"{klucz}=" in tekst, f"brak {klucz} w .env.example"


def test_env_example_daje_sie_wczytac() -> None:
    """Plik wzorcowy musi być poprawną konfiguracją, a nie tylko komentarzem."""
    settings = Settings(_env_file=str(ENV_EXAMPLE))
    assert settings.whisper_idle_unload_s == 300.0
    assert settings.llm_history_max_messages == 16
    assert settings.headless_listen_slice_s == 5.0


# --------------------------------------------------------------------------- #
# Odnośniki
# --------------------------------------------------------------------------- #


def _slug(heading: str) -> str:
    """Kotwica tak, jak liczy ją GitHub: małe litery, bez interpunkcji, spacje → myślniki.

    Myślnik em (—) i ukośnik są USUWANE, a nie zamieniane na myślnik — dlatego
    „Instalacja — Windows 11" daje ``instalacja--windows-11`` z dwoma myślnikami.
    """
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def _headings(text: str) -> set[str]:
    found: set[str] = set()
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = re.match(r"^#{1,6}\s+(.*)$", line)
        if match:
            found.add(_slug(match.group(1)))
    return found


def test_kotwice_wewnetrzne_prowadza_do_istniejacych_naglowkow(readme: str) -> None:
    naglowki = _headings(readme)
    linki = re.findall(r"\]\(#([^)]+)\)", readme)
    assert linki, "spis treści zniknął — to prawie na pewno pomyłka"
    martwe = sorted({link for link in linki if link not in naglowki})
    assert not martwe, f"martwe odnośniki w README: {martwe}"


def test_odnosniki_do_plikow_prowadza_do_istniejacych_plikow(readme: str) -> None:
    sciezki = re.findall(r"\]\((?!#)(?!https?:)([^)]+)\)", readme)
    brakujace = [item for item in sciezki if not (PROJECT_ROOT / item).exists()]
    assert not brakujace, f"README wskazuje na nieistniejące pliki: {brakujace}"


# --------------------------------------------------------------------------- #
# Sekcja ograniczeń — obietnica wobec czytelnika
# --------------------------------------------------------------------------- #


def test_readme_ma_sekcje_ograniczen(readme: str) -> None:
    """Sekcja „Ograniczenia" jest częścią umowy z użytkownikiem, nie ozdobą.

    Ktoś, kto usuwa ją przy porządkach, zabiera czytelnikowi jedyne miejsce,
    w którym napisano wprost, czego ten projekt NIE potrafi.
    """
    assert "## Limitations / Known limitations" in readme
    for temat in (
        "hallucination",         # jakość małego modelu
        "Faster-Whisper",        # jakość STT offline
        "RVC",                   # opóźnienie konwersji głosu
        "HIGH and CRITICAL",     # kompromis bezpieczeństwo/wygoda
        "Crypton Future Media",  # ograniczenia IP
        "synchronisation",       # single-user / single-machine
        "fakes",                 # zielone testy ≠ działanie na sprzęcie
    ):
        assert temat in readme, f"sekcja ograniczeń nie porusza tematu: {temat}"


def test_readme_nie_obiecuje_dzialajacego_rvc(readme: str) -> None:
    """RVC jest przygotowane w konfiguracji, ale niezaimplementowane.

    README ma to mówić wprost — inaczej ktoś wpisze ścieżkę do modelu i będzie
    szukał, czemu nic się nie dzieje.
    """
    assert "not implemented" in readme or "not working yet" in readme


def test_readme_nie_zawiera_sciezek_z_konkretnej_maszyny(readme: str) -> None:
    """Ścieżka z nazwą użytkownika autora nie pomoże nikomu innemu."""
    podejrzane = re.findall(r"/home/[a-z][a-z0-9_-]*", readme, re.IGNORECASE)
    # `~/…` i `%APPDATA%` są w porządku — to zapisy przenośne.
    assert not podejrzane, f"README zawiera ścieżki z konkretnej maszyny: {set(podejrzane)}"


# --------------------------------------------------------------------------- #
# Nazwy narzędzi wymienione w tabeli
# --------------------------------------------------------------------------- #


def test_nazwy_narzedzi_z_readme_istnieja_w_rejestrze(readme: str) -> None:
    """Tabela narzędzi ma opisywać narzędzia, które naprawdę są zarejestrowane."""
    from tools.registry import build_registry

    registry = build_registry(Settings(_env_file=None))
    istniejace = set(registry.names())

    # Z tabeli w README: linie zaczynające się od `| `nazwa.czynnosc``.
    wymienione = set(re.findall(r"^\|\s*`([a-z]+\.[a-z_]+)`", readme, re.MULTILINE))
    assert wymienione, "tabela narzędzi w README zniknęła"

    # Narzędzia pluginów nie są w rejestrze bazowym — pomijamy ich prefiksy.
    prefiksy_pluginow = ("reminders.", "ha.")
    brakujace = sorted(
        name
        for name in wymienione
        if name not in istniejace and not name.startswith(prefiksy_pluginow)
    )
    assert not brakujace, f"README opisuje nieistniejące narzędzia: {brakujace}"


# --------------------------------------------------------------------------- #
# Tabela skryptów instalacyjnych
# --------------------------------------------------------------------------- #


def test_readme_wymienia_kazdy_skrypt_instalacyjny(readme: str) -> None:
    """Skrypt, którego README nie wymienia, jest praktycznie nie do znalezienia."""
    for nazwa in (
        "install-windows.ps1",
        "install-pacman.sh",
        "install-apt.sh",
        "install-linux-generic.sh",
        "install-macos.sh",
        "install.sh",
    ):
        assert nazwa in readme, f"README nie wymienia {nazwa}"


def test_kazdy_skrypt_z_readme_istnieje(readme: str) -> None:
    """Nazwa w tabeli i plik w repozytorium to jedna i ta sama rzecz."""
    import re

    nazwy = set(re.findall(r"(?:scripts[/\\])(install[a-z0-9.-]*\.(?:sh|ps1))", readme))
    assert nazwy, "tabela skryptów instalacyjnych zniknęła z README"
    brakujace = sorted(n for n in nazwy if not (PROJECT_ROOT / "scripts" / n).is_file())
    assert not brakujace, f"README wskazuje na nieistniejące skrypty: {brakujace}"


def test_readme_zgadza_sie_z_wyborem_skryptu_w_kodzie(readme: str) -> None:
    """To, co README każe uruchomić, ma być tym, co `main.py` podpowie użytkownikowi.

    `config._install_script_for` jest jedynym źródłem tej nazwy w kodzie; README
    jest drugim miejscem, w którym ona występuje. Rozjazd oznacza, że jedno
    z dwóch odsyła w złe miejsce.
    """
    from config import OSFamily, PackageManager
    from config import _install_script_for as wybierz

    for os_family, manager in (
        (OSFamily.WINDOWS, PackageManager.NONE),
        (OSFamily.LINUX, PackageManager.APT),
        (OSFamily.LINUX, PackageManager.PACMAN),
        (OSFamily.LINUX, PackageManager.NONE),
        (OSFamily.MACOS, PackageManager.NONE),
    ):
        nazwa = wybierz(os_family, manager).replace("\\", "/").rsplit("/", 1)[-1]
        assert nazwa in readme, (
            f"kod podpowie `{nazwa}` dla {os_family.value}/{manager.value}, "
            "a README go nie wymienia"
        )


def test_readme_opisuje_zachowanie_przy_awarii_kroku(readme: str) -> None:
    """Obietnica „awaria nie ucina instalacji" jest sprawdzalna i ma być zapisana."""
    assert "Idempotence" in readme
    assert "--check-deps" in readme
    assert "does not cut the installation short" in readme


# --------------------------------------------------------------------------- #
# Kompletność plików wzorcowych
# --------------------------------------------------------------------------- #
#
# `.env.example` i `user_settings.example.json` to jedyne, co dostaje ktoś, kto
# klonuje repozytorium: pliki właściwe są w .gitignore. Ustawienie, którego tu
# nie ma, jest praktycznie nie do odkrycia — trzeba by czytać `config.py`.


def test_env_example_opisuje_kazde_ustawienie(defaults: Settings) -> None:
    """Każde pole `Settings` ma mieć swój wpis w `.env.example`.

    Dopuszczamy wpis zakomentowany (`# MIKU_DATA_DIR=`) — to sposób na
    zapisanie „istnieje, domyślnie puste, zostaw jeśli nie wiesz po co".
    """
    import re

    tekst = ENV_EXAMPLE.read_text(encoding="utf-8")
    aktywne = {m.group(1) for m in re.finditer(r"^([A-Z][A-Z0-9_]*)=", tekst, re.MULTILINE)}
    zakomentowane = {
        m.group(1) for m in re.finditer(r"^#\s*([A-Z][A-Z0-9_]*)=", tekst, re.MULTILINE)
    }
    opisane = aktywne | zakomentowane

    brakujace = sorted(
        pole.upper() for pole in type(defaults).model_fields if pole.upper() not in opisane
    )
    assert not brakujace, (
        f"{len(brakujace)} ustawień nie ma w .env.example: {brakujace}"
    )


def test_env_example_nie_niesie_wartosci_wrazliwych() -> None:
    """Klucze API i ścieżki w pliku wzorcowym muszą być PUSTE.

    To jest plik, który trafia do repozytorium publicznego. Wpisana tam
    wartość zostaje w historii gita na zawsze, nawet po późniejszym usunięciu.
    """
    import re

    tekst = ENV_EXAMPLE.read_text(encoding="utf-8")
    wzorzec = r"^([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD))=(.*)$"
    for m in re.finditer(wzorzec, tekst, re.MULTILINE):
        assert not m.group(2).strip(), f"{m.group(1)} ma wpisaną wartość w .env.example"

    # Ścieżki zależne od maszyny też zostają puste — wylicza je config.py.
    for pole in ("PIPER_VOICES_DIR", "PIPER_BINARY", "DATABASE_PATH",
                 "AUDIO_INPUT_DEVICE", "AUDIO_OUTPUT_DEVICE", "FS_ALLOWED_ROOTS"):
        dopasowanie = re.search(rf"^{pole}=(.*)$", tekst, re.MULTILINE)
        assert dopasowanie is not None, f"brak {pole} w .env.example"
        assert not dopasowanie.group(1).strip(), f"{pole} ma wpisaną ścieżkę z konkretnej maszyny"


def test_przyklad_ustawien_uzytkownika_ma_wszystkie_pola() -> None:
    """Łącznie z sekcją `rvc` — inaczej nikt nie wie, że ta opcja istnieje."""
    import json

    from config import RVCSettings, UserSettings

    przyklad = json.loads(
        (PROJECT_ROOT / "config" / "user_settings.example.json").read_text(encoding="utf-8")
    )

    brakujace = sorted(set(UserSettings.model_fields) - set(przyklad))
    assert not brakujace, f"przykład nie ma pól: {brakujace}"

    assert "rvc" in przyklad, "brak sekcji rvc"
    brak_rvc = sorted(set(RVCSettings.model_fields) - set(przyklad["rvc"]))
    assert not brak_rvc, f"sekcja rvc nie ma pól: {brak_rvc}"


def test_przyklad_ustawien_uzytkownika_nie_niesie_prywatnych_sciezek() -> None:
    """Ścieżki do modeli RVC są prywatne — i prawnie obciążone (patrz README)."""
    import json

    przyklad = json.loads(
        (PROJECT_ROOT / "config" / "user_settings.example.json").read_text(encoding="utf-8")
    )
    assert przyklad["rvc"]["model_path"] == ""
    assert przyklad["rvc"]["index_path"] == ""
    assert przyklad["rvc"]["enabled"] is False
    for pole in ("wake_word", "wake_word_model", "personality_traits", "piper_model"):
        assert przyklad[pole] == "", f"{pole} niesie wartość z konkretnej maszyny"


# --------------------------------------------------------------------------- #
# README jako markdown
# --------------------------------------------------------------------------- #
#
# README jest pierwszą (często jedyną) rzeczą, którą ktoś przeczyta na GitHubie.
# Niedomknięty blok kodu albo rozjechana tabela psują wszystko, co jest niżej,
# a w edytorze wyglądają niewinnie.


def _linie_poza_kodem(text: str) -> list[tuple[int, str]]:
    wynik: list[tuple[int, str]] = []
    w_kodzie = False
    for numer, linia in enumerate(text.splitlines(), 1):
        if linia.startswith("```"):
            w_kodzie = not w_kodzie
            continue
        if not w_kodzie:
            wynik.append((numer, linia))
    return wynik


def test_bloki_kodu_sa_domkniete(readme: str) -> None:
    """Nieparzysta liczba ograniczników zamienia resztę dokumentu w jeden blok kodu."""
    ograniczniki = sum(1 for line in readme.splitlines() if line.startswith("```"))
    assert ograniczniki % 2 == 0, f"nieparzysta liczba ograniczników ```: {ograniczniki}"


def test_readme_ma_dokladnie_jeden_naglowek_pierwszego_poziomu(readme: str) -> None:
    """`# cokolwiek` w bloku kodu to komentarz powłoki — liczymy tylko poza kodem."""
    naglowki = [line for _, line in _linie_poza_kodem(readme) if line.startswith("# ")]
    assert len(naglowki) == 1, f"README ma {len(naglowki)} nagłówków poziomu 1: {naglowki}"


def test_tabele_maja_jednolita_liczbe_kolumn(readme: str) -> None:
    """Wiersz z inną liczbą `|` rozjeżdża całą tabelę przy renderowaniu."""
    problemy: list[str] = []
    biezaca: list[tuple[int, int]] = []

    def sprawdz(tabela: list[tuple[int, int]]) -> None:
        if len(tabela) < 2:
            return
        szerokosci = {kolumny for _, kolumny in tabela}
        if len(szerokosci) > 1:
            linie = ", ".join(str(numer) for numer, _ in tabela)
            problemy.append(f"linie {linie}: szerokości {sorted(szerokosci)}")

    for numer, linia in _linie_poza_kodem(readme):
        obcieta = linia.strip()
        if obcieta.startswith("|") and obcieta.endswith("|"):
            biezaca.append((numer, obcieta.count("|")))
        else:
            sprawdz(biezaca)
            biezaca = []
    sprawdz(biezaca)

    assert not problemy, "tabele o niejednolitej liczbie kolumn:\n" + "\n".join(problemy)


def test_badge_maja_poprawne_adresy(readme: str) -> None:
    """Badge z niezakodowaną spacją albo bez schematu nie wyrenderuje się wcale."""
    import re
    from urllib.parse import urlsplit

    obrazki = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", readme)
    assert len(obrazki) >= 4, "badge zniknęły z nagłówka README"
    for opis, url in obrazki:
        assert urlsplit(url).scheme in ("http", "https"), f"badge bez schematu: {url}"
        assert " " not in url, f"niezakodowana spacja w URL: {url}"
        assert opis.strip(), f"badge bez tekstu alternatywnego: {url}"


def test_readme_ostrzega_o_operacjach_wysokiego_ryzyka(readme: str) -> None:
    """Ostrzeżenie o HIGH/CRITICAL ma stać WYSOKO, a nie w połowie dokumentu.

    Ktoś, kto instaluje program mający dostęp do plików i procesów, musi zobaczyć
    tę informację, zanim zdecyduje — a nie po.
    """
    gora = "\n".join(readme.splitlines()[:120])
    assert "HIGH" in gora and "CRITICAL" in gora
    assert "confirmation" in gora
    # Obietnica, że nie da się tego wyłączyć.
    assert "stop asking" in gora or "refusal" in gora


def test_readme_ma_quick_start_i_liste_faz(readme: str) -> None:
    gora = "\n".join(readme.splitlines()[:120])
    assert "Quick start" in gora
    for skrypt in ("install-windows.ps1", "install-apt.sh", "install-pacman.sh"):
        assert skrypt in gora, f"Quick start nie wskazuje {skrypt}"
    assert "Project status" in readme
    # Faza 15 (RVC) ma być oznaczona jako niegotowa, a nie przemilczana.
    assert "15" in readme and "RVC" in readme
