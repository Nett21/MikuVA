"""Testy narzędzi plikowych i ograniczenia do dozwolonych katalogów (Faza 8).

**Żaden test nie dotyka plików użytkownika.** Obszar roboczy to ``tmp_path``
pytesta — katalog utworzony na tę jedną funkcję testową i usuwany po niej. Testy
usuwania kasują pliki, które same wcześniej stworzyły w tym katalogu; wszystko
poza nim jest sprawdzane wyłącznie przez **odmowę dostępu** (nigdy przez próbę
realnej operacji).

Najważniejsza grupa testów: co się dzieje, gdy model poda ścieżkę spoza obszaru —
``..``, ścieżkę bezwzględną, ``~``, dowiązanie symboliczne wyprowadzające na
zewnątrz. Wszystkie mają skończyć się odmową z czytelnym powodem.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from host.paths import PathNotAllowedError, Workspace, looks_binary, read_text_limited
from security.risk import RiskLevel
from tools.base import ToolArgs, ToolContext, ToolError
from tools.filesystem import (
    DeleteArgs,
    ListArgs,
    MkdirArgs,
    MoveArgs,
    ReadArgs,
    SearchArgs,
    WriteArgs,
    build_filesystem_tools,
)


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def make_workspace(root: Path, **overrides: Any) -> Workspace:
    values: dict[str, Any] = {"max_read_bytes": 10_000, "max_write_bytes": 10_000}
    values.update(overrides)
    return Workspace.for_roots([root], **values)


def tools_for(root: Path, **overrides: Any) -> dict[str, Any]:
    workspace = overrides.pop("workspace", None) or make_workspace(root, **overrides)
    built = build_filesystem_tools(make_settings(), workspace=workspace)
    return {tool.spec.name: tool for tool in built}


def ctx() -> ToolContext:
    return ToolContext(settings=make_settings(), language="pl")


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Obszar roboczy z kilkoma plikami — wyłącznie w katalogu tymczasowym."""
    root = tmp_path / "workspace"
    (root / "notatki").mkdir(parents=True)
    (root / "plan.txt").write_text("plan na dziś\nkupić rower\n", encoding="utf-8")
    (root / "notatki" / "rower.md").write_text("# Rower\nKellys, piwnica\n", encoding="utf-8")
    (root / "dane.bin").write_bytes(b"\x00\x01\x02binarne")
    return root


# --------------------------------------------------------------------------- #
# Ograniczenie do dozwolonych katalogów
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sciezka",
    [
        "../poza-obszarem.txt",
        "../../etc/passwd",
        "notatki/../../poza.txt",
        "~/prywatne.txt",
        "/etc/passwd",
        "/",
    ],
)
def test_sciezka_spoza_obszaru_jest_odrzucana(workspace_root: Path, sciezka: str) -> None:
    workspace = make_workspace(workspace_root)
    with pytest.raises(PathNotAllowedError) as blad:
        workspace.resolve(sciezka)
    assert "outside the allowed directories" in blad.value.message


def test_windowsowa_sciezka_bezwzgledna_tez_jest_poza_obszarem(workspace_root: Path) -> None:
    """Model bywa przekonany, że jest na Windowsie — i to też ma się odbić od bramki."""
    workspace = make_workspace(workspace_root)
    with pytest.raises(PathNotAllowedError):
        # Na Uniksie to nazwa pliku, nie ścieżka — dlatego sprawdzamy oba warianty.
        workspace.resolve("C:\\Windows\\System32\\config\\SAM")
        workspace.resolve("//serwer/udział/plik.txt")


@pytest.mark.skipif(
    os.name == "nt", reason="dowiązania symboliczne na Windowsie wymagają uprawnień"
)
def test_dowiazanie_symboliczne_na_zewnatrz_nie_daje_dostepu(
    workspace_root: Path, tmp_path: Path
) -> None:
    """Sedno kolejności „najpierw rozwiń, potem sprawdź"."""
    sekret = tmp_path / "sekret.txt"
    sekret.write_text("hasło", encoding="utf-8")
    (workspace_root / "skrot").symlink_to(sekret)

    workspace = make_workspace(workspace_root)
    with pytest.raises(PathNotAllowedError):
        workspace.resolve("skrot")

    # ...a narzędzie odczytu zamienia to na czytelny błąd dla modelu.
    narzedzia = tools_for(workspace_root)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.read"].run(ReadArgs(path="skrot"), ctx()))
    assert "outside the allowed directories" in blad.value.message


def test_dowiazanie_wewnatrz_obszaru_dziala(workspace_root: Path) -> None:
    """Dowiązanie nie jest zakazane samo w sobie — zakazane jest wyjście poza obszar."""
    (workspace_root / "skrot.txt").symlink_to(workspace_root / "plan.txt")
    workspace = make_workspace(workspace_root)
    assert workspace.resolve("skrot.txt").name == "plan.txt"


def test_sciezka_relatywna_liczy_sie_od_obszaru_nie_od_cwd(
    workspace_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Katalog roboczy procesu nie może wpływać na to, co widzi model."""
    inny = tmp_path / "inny"
    inny.mkdir()
    monkeypatch.chdir(inny)

    workspace = make_workspace(workspace_root)
    assert workspace.resolve("plan.txt") == (workspace_root / "plan.txt").resolve()


def test_etykieta_sciezki_da_sie_odeslac_z_powrotem(workspace_root: Path) -> None:
    """Model widzi etykietę w wyniku i w następnym wywołaniu podaje ją jako ścieżkę.

    Regresja z prawdziwego uruchomienia: etykieta z nazwą katalogu na przodzie
    („ws/plan.txt") była rozwiązywana jako ``<katalog>/ws/plan.txt`` i model dostawał
    „nie ma takiej ścieżki" dla pliku, który właśnie zobaczył.
    """
    workspace = make_workspace(workspace_root)
    etykieta = workspace.label(workspace_root / "notatki" / "rower.md")

    assert etykieta == "notatki/rower.md"  # jeden katalog = zwykła ścieżka względna
    assert workspace.resolve(etykieta, must_exist=True).name == "rower.md"
    # Zapis z nazwą katalogu na przodzie też jest rozumiany.
    prefiks = f"{workspace_root.name}/notatki/rower.md"
    assert workspace.resolve(prefiks, must_exist=True).name == "rower.md"


def test_przy_kilku_katalogach_etykieta_ma_nazwe_katalogu(tmp_path: Path) -> None:
    pierwszy = tmp_path / "kod"
    drugi = tmp_path / "dokumenty"
    pierwszy.mkdir()
    drugi.mkdir()
    (drugi / "umowa.txt").write_text("x", encoding="utf-8")
    workspace = Workspace.for_roots([pierwszy, drugi])

    etykieta = workspace.label(drugi / "umowa.txt")

    assert etykieta == "dokumenty/umowa.txt"
    assert workspace.resolve(etykieta, must_exist=True).name == "umowa.txt"


def test_null_w_argumencie_znaczy_wartosc_domyslna(workspace_root: Path) -> None:
    """Regresja: modele wysyłają ``{"limit": null}`` i to nie może być błędem typu."""
    assert ListArgs.model_validate({"path": ".", "limit": None}).limit == 50
    assert ReadArgs.model_validate({"path": "plan.txt", "max_bytes": None}).max_bytes is None
    # Nieznane pole i zły typ nadal są błędem.
    with pytest.raises(ValueError):
        ListArgs.model_validate({"path": ".", "wymyslone": 1})
    with pytest.raises(ValueError):
        ListArgs.model_validate({"path": ".", "limit": "wszystkie"})


def test_lista_dozwolonych_katalogow_jest_widoczna_dla_modelu(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(narzedzia["fs.roots"].run(ToolArgs(), ctx()))

    assert wynik.ok
    assert wynik.data["roots"][0]["path"] == str(workspace_root.resolve())
    assert "outside" in wynik.data["note"]


def test_domyslny_obszar_to_jeden_katalog_asystenta() -> None:
    """Na świeżej instalacji narzędzia plikowe nie widzą żadnego prywatnego pliku."""
    workspace = Workspace.from_settings(make_settings())
    assert len(workspace.roots) == 1
    assert workspace.primary.name == "workspace"


def test_wskazane_katalogi_z_konfiguracji_sa_uzywane(tmp_path: Path) -> None:
    pierwszy = tmp_path / "jeden"
    drugi = tmp_path / "dwa"
    pierwszy.mkdir()
    drugi.mkdir()
    settings = make_settings(fs_allowed_roots=f"{pierwszy};{drugi}")

    workspace = Workspace.from_settings(settings)

    assert {root.name for root in workspace.roots} == {"jeden", "dwa"}
    assert workspace.contains((drugi / "plik.txt").resolve())


def test_sciezka_relatywna_w_konfiguracji_jest_pomijana(tmp_path: Path) -> None:
    """Katalog względny zależałby od tego, skąd uruchomiono program."""
    settings = make_settings(fs_allowed_roots="./dane")
    workspace = Workspace.from_settings(settings)
    assert workspace.primary.name == "workspace"  # został domyślny obszar


# --------------------------------------------------------------------------- #
# SAFE: odczyt
# --------------------------------------------------------------------------- #


def test_lista_katalogu_pokazuje_katalogi_przed_plikami(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(narzedzia["fs.list"].run(ListArgs(path="."), ctx()))

    rodzaje = [wpis["kind"] for wpis in wynik.data["entries"]]
    assert rodzaje[0] == "dir"
    assert {wpis["name"] for wpis in wynik.data["entries"]} == {
        "notatki", "plan.txt", "dane.bin"
    }


def test_odczyt_pliku_zwraca_tresc_i_oznacza_ja_jako_niezaufana(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(narzedzia["fs.read"].run(ReadArgs(path="plan.txt"), ctx()))

    assert wynik.ok and "kupić rower" in wynik.data["content"]
    # Treść pliku pisał ktoś inny niż użytkownik tej rozmowy.
    assert wynik.untrusted is True


def test_plik_binarny_nie_jest_czytany_jako_tekst(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.read"].run(ReadArgs(path="dane.bin"), ctx()))
    assert "binary file" in blad.value.message
    assert looks_binary(workspace_root / "dane.bin")


def test_dlugi_plik_jest_obcinany_do_limitu(workspace_root: Path) -> None:
    (workspace_root / "duzy.txt").write_text("x" * 5_000, encoding="utf-8")
    narzedzia = tools_for(workspace_root, max_read_bytes=1_000)

    wynik = run(narzedzia["fs.read"].run(ReadArgs(path="duzy.txt"), ctx()))

    assert wynik.data["truncated"] is True
    assert len(wynik.data["content"]) <= 1_000
    text, truncated = read_text_limited(workspace_root / "duzy.txt", 1_000)
    assert truncated and len(text) == 1_000


def test_szukanie_po_nazwie_i_po_tresci(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)

    po_nazwie = run(narzedzia["fs.search"].run(SearchArgs(query="rower"), ctx()))
    assert [hit["name"] for hit in po_nazwie.data["matches"]] == ["rower.md"]

    po_tresci = run(
        narzedzia["fs.search"].run(SearchArgs(query="kupić", in_content=True), ctx())
    )
    nazwy = [hit["name"] for hit in po_tresci.data["matches"]]
    assert "plan.txt" in nazwy
    assert po_tresci.data["matches"][0]["line"] == 2


# --------------------------------------------------------------------------- #
# MEDIUM: tworzenie
# --------------------------------------------------------------------------- #


def test_nowy_plik_to_medium_a_nadpisanie_to_high(workspace_root: Path) -> None:
    """Utworzenie da się cofnąć, nadpisanie niszczy treść — stąd różnica poziomów."""
    zapis = tools_for(workspace_root)["fs.write"]

    assert zapis.effective_risk(WriteArgs(path="nowy.txt", content="x")) is RiskLevel.MEDIUM
    assert (
        zapis.effective_risk(WriteArgs(path="plan.txt", content="x", mode="overwrite"))
        is RiskLevel.HIGH
    )
    assert (
        zapis.effective_risk(WriteArgs(path="plan.txt", content="x", mode="append"))
        is RiskLevel.HIGH
    )


def test_zapis_nowego_pliku(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(
        narzedzia["fs.write"].run(WriteArgs(path="notatki/nowa.md", content="treść"), ctx())
    )

    assert wynik.ok
    assert (workspace_root / "notatki" / "nowa.md").read_text(encoding="utf-8") == "treść"


def test_zapis_nie_nadpisuje_bez_jawnego_trybu(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.write"].run(WriteArgs(path="plan.txt", content="nowe"), ctx()))

    assert "already exists" in blad.value.message
    assert "kupić rower" in (workspace_root / "plan.txt").read_text(encoding="utf-8")


def test_zapis_ponad_limit_jest_odrzucany(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root, max_write_bytes=100)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.write"].run(WriteArgs(path="duzy.txt", content="x" * 500), ctx()))
    assert "write limit" in blad.value.message
    assert not (workspace_root / "duzy.txt").exists()


def test_pytanie_o_zgode_przy_nadpisaniu_pokazuje_rozmiary(workspace_root: Path) -> None:
    zapis = tools_for(workspace_root)["fs.write"]
    args = WriteArgs(path="plan.txt", content="krótko", mode="overwrite")

    pytanie = zapis.confirmation(args, language="pl")

    assert pytanie is not None and pytanie.risk is RiskLevel.HIGH
    assert "Nadpisze plik" in pytanie.summary
    assert str(workspace_root) in "\n".join(pytanie.details)
    assert pytanie.preview == "krótko"


def test_nowy_katalog(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(narzedzia["fs.mkdir"].run(MkdirArgs(path="archiwum/2026"), ctx()))

    assert wynik.ok and (workspace_root / "archiwum" / "2026").is_dir()


# --------------------------------------------------------------------------- #
# HIGH / CRITICAL: przenoszenie i usuwanie
# --------------------------------------------------------------------------- #


def test_przenoszenie_pliku(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(
        narzedzia["fs.move"].run(
            MoveArgs(source="plan.txt", destination="notatki/plan.txt"), ctx()
        )
    )

    assert wynik.ok
    assert not (workspace_root / "plan.txt").exists()
    assert (workspace_root / "notatki" / "plan.txt").exists()


def test_przenoszenie_poza_obszar_jest_odrzucane(workspace_root: Path, tmp_path: Path) -> None:
    narzedzia = tools_for(workspace_root)
    with pytest.raises(ToolError):
        run(
            narzedzia["fs.move"].run(
                MoveArgs(source="plan.txt", destination=str(tmp_path / "wyciek.txt")), ctx()
            )
        )
    assert (workspace_root / "plan.txt").exists()
    assert not (tmp_path / "wyciek.txt").exists()


def test_usuniecie_pliku_to_high_a_katalogu_critical(workspace_root: Path) -> None:
    usun = tools_for(workspace_root)["fs.delete"]

    assert usun.effective_risk(DeleteArgs(path="plan.txt")) is RiskLevel.HIGH
    assert usun.effective_risk(DeleteArgs(path="notatki", recursive=True)) is RiskLevel.CRITICAL


def test_usuniecie_pliku_dziala(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    wynik = run(narzedzia["fs.delete"].run(DeleteArgs(path="plan.txt"), ctx()))

    assert wynik.ok and not (workspace_root / "plan.txt").exists()
    # Pozostałe pliki zostają — usuwamy dokładnie to, o co poproszono.
    assert (workspace_root / "notatki" / "rower.md").exists()


def test_katalog_nie_ginie_bez_jawnego_recursive(workspace_root: Path) -> None:
    narzedzia = tools_for(workspace_root)
    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.delete"].run(DeleteArgs(path="notatki"), ctx()))

    assert "recursive=true" in blad.value.message
    assert (workspace_root / "notatki" / "rower.md").exists()


def test_calego_obszaru_roboczego_nie_da_sie_usunac(workspace_root: Path) -> None:
    """Blokada niezależna od zgody użytkownika i od recursive."""
    narzedzia = tools_for(workspace_root)
    for args in (DeleteArgs(path="."), DeleteArgs(path=".", recursive=True)):
        with pytest.raises(ToolError) as blad:
            run(narzedzia["fs.delete"].run(args, ctx()))
        assert "workspace directory" in blad.value.message
    assert workspace_root.is_dir()


def test_zbyt_duzy_katalog_nie_ginie_jednym_wywolaniem(workspace_root: Path) -> None:
    katalog = workspace_root / "duzo"
    katalog.mkdir()
    for index in range(6):
        (katalog / f"plik{index}.txt").write_text("x", encoding="utf-8")
    narzedzia = tools_for(workspace_root, max_delete_entries=3)

    with pytest.raises(ToolError) as blad:
        run(narzedzia["fs.delete"].run(DeleteArgs(path="duzo", recursive=True), ctx()))

    assert "smaller batches" in blad.value.message
    assert len(list(katalog.iterdir())) == 6


def test_usuniecie_katalogu_z_recursive_dziala(workspace_root: Path) -> None:
    katalog = workspace_root / "do-usuniecia"
    katalog.mkdir()
    (katalog / "a.txt").write_text("a", encoding="utf-8")

    narzedzia = tools_for(workspace_root)
    wynik = run(
        narzedzia["fs.delete"].run(DeleteArgs(path="do-usuniecia", recursive=True), ctx())
    )

    assert wynik.ok and wynik.data["files"] == 1
    assert not katalog.exists()
    assert workspace_root.is_dir()


def test_pytanie_o_zgode_na_usuniecie_katalogu_liczy_pliki(workspace_root: Path) -> None:
    """Użytkownik ma zobaczyć skalę operacji PRZED decyzją."""
    usun = tools_for(workspace_root)["fs.delete"]
    pytanie = usun.confirmation(DeleteArgs(path="notatki", recursive=True), language="pl")

    assert pytanie is not None and pytanie.risk is RiskLevel.CRITICAL
    assert "USUNIE KATALOG" in pytanie.summary and "1 plików" in pytanie.summary
    assert pytanie.preview is not None and "rower.md" in pytanie.preview
    assert pytanie.requires_phrase  # CRITICAL wymaga pełnej frazy


def test_blokady_usuwania_dzialaja_przed_pytaniem_o_zgode(workspace_root: Path) -> None:
    """Nie pytamy o coś, co i tak odrzucimy — bramka NORMALIZE jest przed CONFIRM.

    Zauważone przy uruchomieniu na żywo: użytkownik dostawał pytanie „usunąć cały
    obszar roboczy?", a po odpowiedzi narzędzie mówiło „nie usuwam".
    """
    import asyncio as _asyncio

    from conftest import SpyBroker

    from brain.tool_router import ToolCall, ToolRouter
    from security.audit import AuditLog
    from security.policy import SecurityPolicy

    settings = make_settings(security_allow_critical=True)
    workspace = make_workspace(workspace_root)
    from tools.registry import ToolRegistry

    registry = ToolRegistry(build_filesystem_tools(settings, workspace=workspace))
    broker = SpyBroker(approve=True)
    router = ToolRouter(
        registry,
        settings=settings,
        policy=SecurityPolicy(settings),
        broker=broker,
        audit=AuditLog(enabled=True),
    )

    for arguments in ({"path": "."}, {"path": "notatki"}):
        outcome = _asyncio.run(
            router.dispatch(ToolCall(name="fs.delete", arguments=arguments), ctx())
        )
        assert not outcome.ok and outcome.gate == "NORMALIZE"

    assert broker.requests == []  # nikogo o nic nie pytaliśmy
    assert workspace_root.is_dir() and (workspace_root / "notatki").is_dir()


def test_tryb_probny_nie_usuwa_niczego(workspace_root: Path) -> None:
    usun = tools_for(workspace_root)["fs.delete"]
    podglad = run(usun.preview(DeleteArgs(path="notatki", recursive=True), ctx()))

    assert "usunęłoby katalog" in podglad
    assert (workspace_root / "notatki" / "rower.md").exists()
