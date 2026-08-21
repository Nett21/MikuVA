"""Testy skryptów instalacyjnych.

Powód istnienia: komunikat o brakach kieruje użytkownika do konkretnego pliku
(`config.install_instruction`). Jeśli nazwa w kodzie rozjedzie się z zawartością
katalogu ``scripts/``, człowiek na świeżej maszynie dostanie polecenie
uruchomienia pliku, którego nie ma — i nie ma jak tego zauważyć bez testu.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from config import (
    PROJECT_ROOT,
    USER_SETTINGS_EXAMPLE_FILE,
    OSFamily,
    PackageManager,
    PlatformInfo,
    Settings,
    UserSettings,
    install_instruction,
)
from config import _install_script_for as install_script_for

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

UNIX_INSTALLERS = (
    "install-pacman.sh",
    "install-apt.sh",
    "install-linux-generic.sh",
    "install-macos.sh",
    "install-offline.sh",
)
POWERSHELL_INSTALLERS = ("install-windows.ps1", "install-offline.ps1", "install.ps1")


def script_path(name: str) -> Path:
    """Nazwa z ``config.py`` (także w zapisie windowsowym) → ścieżka w repozytorium."""
    return PROJECT_ROOT / Path(name.replace("\\", "/"))


# --------------------------------------------------------------------------- #
# Spójność komunikatów z zawartością repozytorium
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("os_family", list(OSFamily))
@pytest.mark.parametrize("manager", list(PackageManager))
def test_kazdy_wskazywany_skrypt_instalacyjny_istnieje(
    os_family: OSFamily, manager: PackageManager
) -> None:
    """Dla ŻADNEJ kombinacji systemu i menedżera nie wolno wskazać nieistniejącego pliku."""
    name = install_script_for(os_family, manager)
    path = script_path(name)

    assert path.is_file(), f"{os_family}/{manager} wskazuje na nieistniejący {name}"
    assert path.stat().st_size > 0


def test_komunikat_o_brakach_wskazuje_istniejacy_plik() -> None:
    info = PlatformInfo(
        os_family=OSFamily.LINUX,
        os_label="Test Linux",
        os_release="0",
        os_version="0",
        distro_id="test",
        distro_like=(),
        machine="x86_64",
        processor="test",
        cpu_count=1,
        python_version="3.12.0",
        python_executable="python3",
        package_manager=PackageManager.APT,
        install_script=install_script_for(OSFamily.LINUX, PackageManager.APT),
        is_wsl=False,
    )

    message = install_instruction(info)

    assert "install-apt.sh" in message
    assert script_path("scripts/install-apt.sh").is_file()


# --------------------------------------------------------------------------- #
# Wykonywalność i kształt skryptów
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", UNIX_INSTALLERS)
def test_instalatory_uniksowe_maja_shebang_i_prawo_wykonania(name: str) -> None:
    path = SCRIPTS_DIR / name
    content = path.read_text(encoding="utf-8")

    assert content.startswith("#!/usr/bin/env bash")
    assert path.stat().st_mode & stat.S_IXUSR, f"{name} nie ma prawa wykonania"


@pytest.mark.parametrize("name", UNIX_INSTALLERS)
def test_nakladki_korzystaja_ze_wspolnej_logiki(name: str) -> None:
    """Warianty dystrybucyjne mają ustawiać nazwy pakietów, a nie kopiować logikę."""
    content = (SCRIPTS_DIR / name).read_text(encoding="utf-8")

    assert "install-common.sh" in content
    assert "run_installer" in content
    # Nakładka ma być cienka — logika mieszka w install-common.sh. Liczymy
    # WYŁĄCZNIE linie kodu: komentarz wyjaśniający, czemu coś wygląda tak,
    # a nie inaczej, jest w tym projekcie wartością, a nie objętością do cięcia.
    kod = [
        line
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(kod) < 50, f"{name} zaczyna dublować wspólną logikę ({len(kod)} linii kodu)"


@pytest.mark.parametrize("name", POWERSHELL_INSTALLERS)
def test_skrypty_powershell_maja_bom(name: str) -> None:
    """Bez BOM-u Windows PowerShell 5.1 czyta plik jako ANSI i psuje polskie znaki."""
    assert (SCRIPTS_DIR / name).read_bytes().startswith(b"\xef\xbb\xbf")


# --------------------------------------------------------------------------- #
# Bezpieczeństwo
# --------------------------------------------------------------------------- #


_PIPE_TO_SHELL = re.compile(r"curl[^\n|]*\|\s*(ba)?sh", re.IGNORECASE)


def strip_comments(content: str) -> str:
    """Zostaw same instrukcje — komentarze mogą o czymś wspominać, nie wykonują tego."""
    return "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))


@pytest.mark.parametrize(
    "name",
    [*UNIX_INSTALLERS, *POWERSHELL_INSTALLERS, "install-common.sh", "install.sh", "install-rvc.sh"],
)
def test_zaden_instalator_nie_wykonuje_pobranego_skryptu(name: str) -> None:
    """`curl | sh` uniemożliwia obejrzenie, co się wykona — nie robimy tego."""
    content = strip_comments((SCRIPTS_DIR / name).read_text(encoding="utf-8-sig"))
    assert _PIPE_TO_SHELL.search(content) is None


def test_instalatory_nie_zakladaja_sciezek_z_konkretnej_maszyny() -> None:
    """Żadnej ścieżki bezwzględnej spoza katalogu projektu ani nazwy użytkownika."""
    suspicious = re.compile(r"(/home/[a-z]|C:\\\\Users|/Users/[a-z])", re.IGNORECASE)
    for path in SCRIPTS_DIR.iterdir():
        if path.suffix not in (".sh", ".ps1", ".py"):
            continue
        content = path.read_text(encoding="utf-8-sig")
        assert suspicious.search(content) is None, (
            f"{path.name} zawiera ścieżkę z konkretnej maszyny"
        )


# --------------------------------------------------------------------------- #
# Udokumentowana ścieżka instalacji
# --------------------------------------------------------------------------- #


def test_env_example_wczytuje_sie_jako_poprawna_konfiguracja() -> None:
    """Regresja z prawdziwej instalacji: `cp .env.example .env` wywracał start.

    Wartości z plików środowiskowych przychodzą jako TEKST, a `Literal[10, 20, 30]`
    nie konwertuje `"20"` na `20` — więc każdy, kto wykonał udokumentowany krok
    instalacji, dostawał błąd walidacji zamiast działającego programu. W repo nie
    ma pliku `.env`, więc bez tego testu nikt by tego nie zauważył.
    """
    settings = Settings(_env_file=str(PROJECT_ROOT / ".env.example"))

    assert settings.audio_frame_ms == 20
    assert settings.wake_enabled is True
    assert settings.offline_mode == "auto"


def test_przyklad_ustawien_uzytkownika_jest_poprawny() -> None:
    """`config/user_settings.example.json` musi przechodzić walidację modelu."""
    payload = json.loads(USER_SETTINGS_EXAMPLE_FILE.read_text(encoding="utf-8"))
    user = UserSettings.model_validate(payload)

    assert user.effective_wake_word == "hej Miku"
    assert user.speech_language == "auto"


# --------------------------------------------------------------------------- #
# Jedno wejście na każdą platformę
# --------------------------------------------------------------------------- #

ENTRY_POINTS = ("install.sh", "install.ps1")


def make_fake_path(tmp_path: Path, system: str, managers: tuple[str, ...]) -> Path:
    """Katalog udający cały PATH: podany system i podane menedżery, nic więcej."""
    fake_path = tmp_path / "bin"
    fake_path.mkdir(exist_ok=True)
    uname = fake_path / "uname"
    uname.write_text(f"#!/usr/bin/env bash\necho {system}\n", encoding="utf-8")
    uname.chmod(0o755)
    for manager in managers:
        program = fake_path / manager
        program.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        program.chmod(0o755)
    # Programy, których używa sam skrypt — bez nich nie ruszy z pustym PATH.
    for tool in ("dirname", "basename", "bash"):
        found = shutil.which(tool)
        if found:
            (fake_path / tool).symlink_to(found)
    return fake_path


def test_wejscie_uniksowe_jest_wykonywalne() -> None:
    """`./scripts/install.sh` ma działać na świeżo sklonowanym repozytorium."""
    path = SCRIPTS_DIR / "install.sh"

    assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert path.stat().st_mode & stat.S_IXUSR, "install.sh nie ma prawa wykonania"


def test_wejscie_windowsowe_ma_bom() -> None:
    assert (SCRIPTS_DIR / "install.ps1").read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_wejscie_nie_dubluje_logiki_instalacji(name: str) -> None:
    """Wejście ma WSKAZYWAĆ instalator, a nie instalować samo."""
    content = (SCRIPTS_DIR / name).read_text(encoding="utf-8-sig")

    assert "pip install" not in content
    assert len(content.splitlines()) < 70


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
@pytest.mark.parametrize(
    ("system", "managers", "expected"),
    [
        ("Darwin", (), "install-macos.sh"),
        ("Linux", ("pacman",), "install-pacman.sh"),
        ("Linux", ("apt-get",), "install-apt.sh"),
        ("Linux", (), "install-linux-generic.sh"),
        # Fedora i openSUSE obsługuje jeden plik — ten sam co „brak menedżera".
        ("Linux", ("dnf",), "install-linux-generic.sh"),
    ],
)
def test_wejscie_wybiera_instalator_po_systemie(
    tmp_path: Path, system: str, managers: tuple[str, ...], expected: str
) -> None:
    """Wybór zależy od OBECNOŚCI menedżera w PATH, nie od nazwy dystrybucji.

    Pochodne (Manjaro, Mint, Omarchy, Pop!_OS) mają własne nazwy i ten sam
    menedżer — lista nazw dystrybucji nigdy nie byłaby kompletna.
    """
    sandbox = tmp_path / "scripts"
    sandbox.mkdir()
    shutil.copy(SCRIPTS_DIR / "install.sh", sandbox / "install.sh")
    for candidate in (
        "install-macos.sh",
        "install-pacman.sh",
        "install-apt.sh",
        "install-linux-generic.sh",
    ):
        stub = sandbox / candidate
        stub.write_text(f"#!/usr/bin/env bash\necho WYBRANO {candidate}\n", encoding="utf-8")
        stub.chmod(0o755)

    # PATH zawiera WYŁĄCZNIE atrapy: prawdziwy menedżer pakietów tej maszyny
    # (tu: pacman) wygrałby z każdym scenariuszem i test sprawdzałby sam siebie.
    fake_path = make_fake_path(tmp_path, system, managers)

    result = subprocess.run(
        [str(sandbox / "install.sh")],
        capture_output=True,
        text=True,
        env={"PATH": str(fake_path), "HOME": str(tmp_path)},
        timeout=30,
        check=False,
    )

    assert f"WYBRANO {expected}" in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_wejscie_na_git_bashu_odsyla_do_powershella(tmp_path: Path) -> None:
    """Pod Git Bashem nie ma czym instalować pakietów systemu Windows."""
    sandbox = tmp_path / "scripts"
    sandbox.mkdir()
    shutil.copy(SCRIPTS_DIR / "install.sh", sandbox / "install.sh")
    fake_path = make_fake_path(tmp_path, "MINGW64_NT-10.0", ())

    result = subprocess.run(
        [str(sandbox / "install.sh")],
        capture_output=True,
        text=True,
        env={"PATH": str(fake_path)},
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "install.ps1" in result.stderr


# --------------------------------------------------------------------------- #
# Tryb pełny
# --------------------------------------------------------------------------- #


def source_common(
    snippet: str, tmp_path: Path, *, python_stub: str
) -> subprocess.CompletedProcess[str]:
    """Uruchom fragment kodu z załadowanym install-common.sh i atrapą Pythona."""
    stub = tmp_path / "fake-python"
    stub.write_text(python_stub, encoding="utf-8")
    stub.chmod(0o755)
    script = f'''
PKG_LABEL="test"; PKG_INSTALL=(echo)
source "{SCRIPTS_DIR}/install-common.sh"
VENV_PYTHON="{stub}"
{snippet}
'''
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_full_wlacza_takze_pakiety_deweloperskie(tmp_path: Path) -> None:
    """„Wszystko" ma znaczyć wszystko — bez drugiej flagi do zapamiętania."""
    result = source_common(
        'parse_arguments --full; echo "FULL=$FULL DEV=$WITH_DEV"',
        tmp_path,
        python_stub="#!/usr/bin/env bash\nexit 0\n",
    )

    assert "FULL=1 DEV=1" in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_nieudany_pakiet_opcjonalny_nie_przerywa_instalacji(tmp_path: Path) -> None:
    """Brak koła dla jednego pakietu nie może zabrać pozostałych ani całej instalacji.

    Skrypt działa pod `set -e`, więc bez świadomej obsługi błędu pierwszy pakiet
    bez koła (a takie się zdarzają przy nowym Pythonie) kończyłby instalację
    w połowie — po cichu, bo pip pisze na stderr.
    """
    stub = (
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do [[ $arg == openwakeword ]] && exit 1; done\n'
        "exit 0\n"
    )
    result = source_common(
        "parse_arguments --full; install_optional_packages; report_missing; echo KONIEC",
        tmp_path,
        python_stub=stub,
    )

    assert result.returncode == 0
    assert "KONIEC" in result.stdout, "instalacja przerwała się na nieudanym pakiecie"
    assert "pypdf" in result.stdout, "pakiety po nieudanym nie zostały zainstalowane"
    # Nieudany pakiet ma trafić do podsumowania razem z poleceniem naprawy.
    assert "openwakeword" in result.stdout
    assert "pip install openwakeword" in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_bez_full_nie_ma_pakietow_dodatkowych(tmp_path: Path) -> None:
    """Zwykła instalacja zostaje taka jak była — pełna jest świadomym wyborem."""
    result = source_common(
        "install_optional_packages",
        tmp_path,
        python_stub="#!/usr/bin/env bash\nexit 0\n",
    )

    assert "piper-tts" in result.stdout
    assert "pypdf" not in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_tryb_offline_nie_siega_do_sieci_po_opcje(tmp_path: Path) -> None:
    result = source_common(
        "parse_arguments --full --offline; install_optional_packages; fetch_models",
        tmp_path,
        python_stub="#!/usr/bin/env bash\nexit 0\n",
    )

    assert "offline" in result.stdout.lower()
    assert "pypdf" not in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_bez_karty_nvidia_nie_ruszamy_pakietow_cuda(tmp_path: Path) -> None:
    """Na maszynie bez NVIDII instalowanie CUDA byłoby kilkoma GB bez powodu."""
    result = source_common(
        "parse_arguments --full; has_nvidia_gpu() { return 1; }; install_gpu_packages",
        tmp_path,
        python_stub="#!/usr/bin/env bash\nexit 0\n",
    )

    assert "NVIDIA" in result.stdout
    assert "Wykonuję" not in result.stdout, "próbowaliśmy coś zainstalować mimo braku karty"


# --------------------------------------------------------------------------- #
# Zgodność między platformami
# --------------------------------------------------------------------------- #


def optional_packages_from_shell() -> set[str]:
    content = (SCRIPTS_DIR / "install-common.sh").read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([a-z0-9-]+)\|', content, re.MULTILINE))


def optional_packages_from_powershell() -> set[str]:
    content = (SCRIPTS_DIR / "install-windows.ps1").read_text(encoding="utf-8-sig")
    # Pierwszy znak musi być literą: „-m", „-c" i „--no-index" to argumenty
    # pipa w tym samym pliku, a nie nazwy pakietów.
    return set(re.findall(r'@\("([a-z][a-z0-9-]*)",\s*"', content))


def test_ten_sam_zestaw_opcji_na_kazdej_platformie() -> None:
    """Windows nie może dostawać innego zestawu funkcji niż Linux i macOS.

    Bez tego testu dopisanie pakietu w jednym skrypcie i zapomnienie o drugim
    jest niewidoczne do czasu, aż ktoś zainstaluje asystenta na drugim systemie.
    """
    assert optional_packages_from_shell() == optional_packages_from_powershell()


def test_pelna_instalacja_jest_opisana_w_pomocy() -> None:
    shell = (SCRIPTS_DIR / "install-common.sh").read_text(encoding="utf-8")
    windows = (SCRIPTS_DIR / "install-windows.ps1").read_text(encoding="utf-8-sig")

    assert "--full" in shell
    assert "-Full" in windows


# --------------------------------------------------------------------------- #
# Odporność na błędy: pojedynczy nieudany krok nie może uciąć instalacji
# --------------------------------------------------------------------------- #
#
# Skrypty działają pod `set -e`, więc KAŻDE niesprawdzone polecenie kończące się
# błędem przerywa je natychmiast — bez podsumowania, bez raportu --check-deps
# i bez słowa o tym, co poszło nie tak. To jest domyślne zachowanie basha i
# trzeba je świadomie wyłączyć w każdym miejscu, gdzie awaria jest możliwa.


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_nieudany_pip_nie_przerywa_instalacji(tmp_path: Path) -> None:
    """Zerwana sieć w środku `pip install` to najczęstsza awaria instalacji.

    Ma skończyć się wpisem w podsumowaniu i przejściem dalej, a nie urwanym
    skryptem — użytkownik musi zobaczyć raport `--check-deps`, żeby wiedzieć,
    czego brakuje.
    """
    stub = (
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do [[ $arg == *requirements.txt ]] && exit 1; done\n'
        "exit 0\n"
    )
    result = source_common(
        "install_python_packages; report_missing; echo KONIEC",
        tmp_path,
        python_stub=stub,
    )

    assert result.returncode == 0
    assert "KONIEC" in result.stdout, "instalacja urwała się na nieudanym pip"
    assert "requirements.txt" in result.stdout
    assert "pip install -r requirements.txt" in result.stdout, "brak polecenia naprawczego"


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_nieudany_venv_konczy_sie_diagnoza_a_nie_sladem(tmp_path: Path) -> None:
    """Brak modułu venv (Debian/Ubuntu) ma dać nazwę pakietu, nie ślad Pythona."""
    stub = "#!/usr/bin/env bash\nexit 1\n"  # każde wywołanie Pythona zawodzi
    fake = tmp_path / "fake-python"
    fake.write_text(stub, encoding="utf-8")
    fake.chmod(0o755)

    script = f'''
PKG_LABEL="apt"; PKG_INSTALL=(apt-get install -y); PKG_PYTHON="python3 python3-venv python3-pip"
source "{SCRIPTS_DIR}/install-common.sh"
find_python() {{ echo "{fake}"; }}
VENV_DIR="{tmp_path}/venv"
setup_venv || echo "ZWROCONO_BLAD"
echo KONIEC
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert "KONIEC" in result.stdout, "skrypt urwał się zamiast zwrócić błąd"
    assert "ZWROCONO_BLAD" in result.stdout
    combined = result.stdout + result.stderr
    assert "python3-venv" in combined, "brak nazwy pakietu, którego zabrakło"
    assert "Traceback" not in combined


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_bez_menedzera_pakietow_dostajemy_liste_skladnikow(tmp_path: Path) -> None:
    """Nieznana dystrybucja ma dostać listę SKŁADNIKÓW, nie zgadywane nazwy pakietów."""
    fake = tmp_path / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    script = f'''
PKG_LABEL=""; PKG_INSTALL=()
source "{SCRIPTS_DIR}/install-common.sh"
install_system_packages
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0
    for skladnik in ("PortAudio", "Ollama", "venv", "ffmpeg", "Tk"):
        assert skladnik in result.stdout, f"lista ręczna nie wymienia: {skladnik}"
    # Bez zgadywania konkretnego polecenia dla nieznanego menedżera.
    assert "apt-get install" not in result.stdout
    assert "pacman -S" not in result.stdout


# --------------------------------------------------------------------------- #
# AUR: wykrywamy pomocnika, nigdy go nie instalujemy
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_bez_pomocnika_aur_dostajemy_instrukcje_zamiast_instalacji(tmp_path: Path) -> None:
    """`paru` i `yay` same pochodzą z AUR-a — instalowanie ich w tle to budowanie
    obcego kodu bez pytania. Ma być instrukcja, nie akcja."""
    fake = tmp_path / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    script = f'''
PKG_LABEL="pacman"; PKG_INSTALL=(echo); PKG_AUR_OPTIONAL="piper-tts-bin"
source "{SCRIPTS_DIR}/install-common.sh"
parse_arguments --full --yes
aur_helper() {{ return 1; }}
install_aur_packages
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0
    assert "makepkg -si" in result.stdout, "brak instrukcji ręcznej"
    assert "paru -S" in result.stdout
    # Nie wolno próbować zainstalować samego pomocnika.
    assert "-S paru" not in result.stdout
    assert "-S yay" not in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_aur_dziala_tylko_w_trybie_full(tmp_path: Path) -> None:
    """Pakiety z AUR-a są dodatkiem — zwykła instalacja ich nie dotyka."""
    fake = tmp_path / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    script = f'''
PKG_LABEL="pacman"; PKG_INSTALL=(echo); PKG_AUR_OPTIONAL="piper-tts-bin"
source "{SCRIPTS_DIR}/install-common.sh"
parse_arguments --yes
aur_helper() {{ echo paru; }}
install_aur_packages
echo KONIEC
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert "KONIEC" in result.stdout
    assert "piper-tts-bin" not in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_aur_nie_dotyczy_innych_dystrybucji(tmp_path: Path) -> None:
    """Poza Archem `PKG_AUR_OPTIONAL` jest puste i krok ma się pominąć bez śladu."""
    fake = tmp_path / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    script = f'''
PKG_LABEL="apt"; PKG_INSTALL=(echo)
source "{SCRIPTS_DIR}/install-common.sh"
parse_arguments --full --yes
install_aur_packages
echo KONIEC
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )

    assert "KONIEC" in result.stdout
    assert "AUR" not in result.stdout


# --------------------------------------------------------------------------- #
# ffmpeg i nagłówki PortAudio
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_ffmpeg_dokladany_tylko_gdy_go_nie_ma(tmp_path: Path) -> None:
    """ffmpeg jest OPCJONALNY i idempotentny: obecny w PATH nie jest instalowany ponownie."""
    fake = tmp_path / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (bin_dir / "ffmpeg").chmod(0o755)

    def run(path_prefix: str) -> subprocess.CompletedProcess[str]:
        script = f'''
export PATH="{path_prefix}$PATH"
PKG_LABEL="apt"; PKG_INSTALL=(echo); PKG_FFMPEG="ffmpeg"; PKG_PYTHON="python3"
source "{SCRIPTS_DIR}/install-common.sh"
parse_arguments --yes
find_python() {{ echo "{fake}"; }}
install_system_packages
'''
        return subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )

    z_ffmpegiem = run(f"{bin_dir}:")
    assert "ffmpeg" not in z_ffmpegiem.stdout, "ffmpeg obecny w PATH był instalowany ponownie"


def test_apt_i_pacman_znaja_ffmpeg() -> None:
    """Obie główne dystrybucje mają go wymieniony — o to prosi dokumentacja."""
    for name in ("install-apt.sh", "install-pacman.sh"):
        content = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert 'PKG_FFMPEG="ffmpeg"' in content, f"{name} nie zna ffmpeg"


def test_naglowki_portaudio_nie_sa_instalowane_domyslnie() -> None:
    """Pakiety -dev wciągają kompilator; koło `sounddevice` z PyPI go nie potrzebuje.

    Nazwa pakietu ma być ZNANA (żeby podpowiedzieć ją po nieudanym pip),
    ale nie ma trafiać do listy instalowanej bezwarunkowo.
    """
    apt = (SCRIPTS_DIR / "install-apt.sh").read_text(encoding="utf-8")
    assert 'PKG_AUDIO_BUILD="portaudio19-dev"' in apt
    assert 'PKG_AUDIO="libportaudio2"' in apt

    common = (SCRIPTS_DIR / "install-common.sh").read_text(encoding="utf-8")
    # Nagłówki pojawiają się WYŁĄCZNIE w podpowiedzi po nieudanej instalacji.
    assert "PKG_AUDIO_BUILD" in common
    assert "packages+=(${PKG_AUDIO_BUILD})" not in common


# --------------------------------------------------------------------------- #
# Sprawdzenie urządzeń audio
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_brak_mikrofonu_trafia_do_podsumowania(tmp_path: Path) -> None:
    """Sprzętu nie doinstaluje żaden skrypt — użytkownik ma to zobaczyć wprost."""
    stub = (
        "#!/usr/bin/env bash\n"
        "cat >/dev/null\n"
        "echo 'MIC|0|'\n"
        "echo 'OUT|2|Głośniki'\n"
    )
    result = source_common(
        "check_audio_devices; report_missing", tmp_path, python_stub=stub
    )

    assert result.returncode == 0
    assert "Mikrofon: nie znaleziono" in result.stdout
    assert "Wyjście dźwięku: 2" in result.stdout
    assert "podlacz go" in result.stdout or "podłącz go" in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_awaria_detekcji_audio_nie_przerywa_instalacji(tmp_path: Path) -> None:
    """Brak PortAudio rzuca OSError już przy imporcie — to nie może uciąć skryptu."""
    stub = "#!/usr/bin/env bash\ncat >/dev/null\necho 'MICERR|brak biblioteki PortAudio'\n"
    result = source_common(
        "check_audio_devices; echo KONIEC", tmp_path, python_stub=stub
    )

    assert "KONIEC" in result.stdout
    assert "PortAudio" in result.stdout


def test_detekcja_audio_nie_dubluje_logiki_z_projektu() -> None:
    """Instalator ma WOŁAĆ funkcje projektu, a nie mieć własną detekcję.

    Druga implementacja rozjechałaby się z pierwszą przy pierwszej zmianie —
    a wtedy instalator melduje mikrofon, którego `--check-deps` nie widzi.
    """
    for name, marker in (
        ("install-common.sh", "from audio.microphone import list_input_devices"),
        ("install-windows.ps1", "from audio.microphone import list_input_devices"),
    ):
        content = (SCRIPTS_DIR / name).read_text(encoding="utf-8-sig")
        assert marker in content, f"{name} nie korzysta z detekcji projektu"
        assert "from audio.output import list_output_devices" in content


# --------------------------------------------------------------------------- #
# Pętla zwrotna z main.py: wskazuj, nigdy nie uruchamiaj
# --------------------------------------------------------------------------- #


def test_main_nigdy_nie_uruchamia_instalatora() -> None:
    """`main.py` ma WSKAZAĆ skrypt, nigdy go nie odpalić.

    Uruchomienie instalatora bez pytania oznaczałoby instalowanie pakietów
    systemowych (i pytanie o sudo) w odpowiedzi na zwykłe `python main.py` —
    czyli dokładnie to, czego użytkownik się nie spodziewa.
    """
    content = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")

    for wzorzec in ("install-apt.sh", "install-pacman.sh", "install-windows.ps1",
                    "install-linux-generic.sh", "install.sh"):
        assert wzorzec not in content, (
            f"main.py wymienia {wzorzec} z nazwy — nazwa ma pochodzić "
            "z config.install_instruction()"
        )

    # Żadnego uruchamiania procesów w main.py w ogóle.
    assert "subprocess" not in content, "main.py nie ma prawa uruchamiać procesów"
    assert "os.system" not in content
    assert "os.exec" not in content


@pytest.mark.parametrize(
    ("os_family", "manager", "oczekiwany"),
    [
        (OSFamily.WINDOWS, PackageManager.NONE, "install-windows.ps1"),
        (OSFamily.LINUX, PackageManager.APT, "install-apt.sh"),
        (OSFamily.LINUX, PackageManager.PACMAN, "install-pacman.sh"),
        (OSFamily.LINUX, PackageManager.DNF, "install-linux-generic.sh"),
        (OSFamily.LINUX, PackageManager.NONE, "install-linux-generic.sh"),
        (OSFamily.UNKNOWN, PackageManager.NONE, "install-linux-generic.sh"),
    ],
)
def test_nazwa_skryptu_zgadza_sie_z_plikiem_w_repozytorium(
    os_family: OSFamily, manager: PackageManager, oczekiwany: str
) -> None:
    """Nazwa w komunikacie i nazwa pliku to jedna i ta sama rzecz — pilnuje tego test.

    Zmiana nazwy pliku bez zmiany `config._install_script_for` daje komunikat
    odsyłający do skryptu, którego nie ma; test wychwyci to od razu.
    """
    name = install_script_for(os_family, manager)
    assert name.replace("\\", "/").endswith(oczekiwany)
    assert script_path(name).is_file()


# --------------------------------------------------------------------------- #
# PowerShell: sprawdzenia statyczne
# --------------------------------------------------------------------------- #
#
# Na maszynie deweloperskiej (Linux) nie ma czym uruchomić PowerShella, więc
# składni nie da się sprawdzić wykonaniem. Te testy łapią klasy błędów, które
# realnie da się popełnić edytując plik z drugiego systemu — i które wychodzą
# dopiero u użytkownika, na Windowsie, w połowie instalacji.


def powershell_source(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8-sig")


def _strip_ps_strings_and_comments(content: str) -> str:
    """Zostaw sam kod: bez komentarzy, łańcuchów i bloków tekstowych.

    Nawias klamrowy w komentarzu albo w napisie nie jest nawiasem kodu —
    liczenie ich razem dawałoby fałszywy alarm przy każdym opisie ze znakiem `{`.
    """
    out: list[str] = []
    in_here = False
    for line in content.splitlines():
        if in_here:
            if line.startswith(("'@", '"@')):
                in_here = False
            continue
        if re.search(r"@['\"]\s*$", line):
            in_here = True
            continue
        without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "", line)
        without_comment = without_strings.split("#", 1)[0]
        out.append(without_comment)
    return "\n".join(out)


@pytest.mark.parametrize("name", POWERSHELL_INSTALLERS)
def test_powershell_ma_zbilansowane_nawiasy(name: str) -> None:
    code = _strip_ps_strings_and_comments(powershell_source(name))
    for otwierajacy, zamykajacy, opis in (("{", "}", "klamrowe"), ("(", ")", "okrągłe")):
        assert code.count(otwierajacy) == code.count(zamykajacy), (
            f"{name}: niezbilansowane nawiasy {opis} "
            f"({code.count(otwierajacy)} otwarć, {code.count(zamykajacy)} zamknięć)"
        )


@pytest.mark.parametrize("name", POWERSHELL_INSTALLERS)
def test_terminator_bloku_tekstowego_stoi_w_pierwszej_kolumnie(name: str) -> None:
    """PowerShell wymaga `'@` na POCZĄTKU linii — wcięty terminator to błąd składni.

    To jedna z tych rzeczy, które wyglądają poprawnie w edytorze
    z automatycznym wcięciem i wywracają skrypt dopiero przy uruchomieniu.
    """
    content = powershell_source(name)
    otwarcia = len(re.findall(r"@['\"]\s*$", content, re.MULTILINE))
    zamkniecia = len(re.findall(r"^['\"]@", content, re.MULTILINE))
    assert otwarcia == zamkniecia, (
        f"{name}: {otwarcia} otwarć bloku tekstowego, {zamkniecia} zamknięć "
        "w pierwszej kolumnie — sprawdź wcięcie terminatora"
    )


def test_windows_sprawdza_kod_wyjscia_po_kluczowych_poleceniach() -> None:
    """PowerShell NIE przerywa pracy, gdy program zewnętrzny zwróci błąd.

    `$ErrorActionPreference = "Stop"` dotyczy poleceń PowerShella, nie natywnych.
    Bez jawnego sprawdzenia `$LASTEXITCODE` nieudany `pip install` przechodzi
    po cichu, a skrypt melduje sukces — czyli awaria jest UKRYTA, co jest
    gorsze niż przerwanie.
    """
    content = powershell_source("install-windows.ps1")

    # Po instalacji requirements.txt musi być sprawdzenie kodu wyjścia.
    po_pip = content.split('-r (Join-Path $ProjectRoot "requirements.txt")', 1)
    assert len(po_pip) == 2, "nie znalazłem instalacji requirements.txt"
    assert "$LASTEXITCODE" in po_pip[1][:400], "brak sprawdzenia wyniku pip"

    # To samo po utworzeniu środowiska.
    po_venv = content.split("-m venv $VenvDir", 1)
    assert len(po_venv) == 2
    assert "$LASTEXITCODE" in po_venv[1][:400], "brak sprawdzenia wyniku venv"


def test_windows_nie_instaluje_pythona_bez_zgody() -> None:
    """Instalacja interpretera zmienia PATH całego konta — to decyzja użytkownika."""
    content = powershell_source("install-windows.ps1")

    # Interesuje nas WYWOŁANIE, a nie linia, która je zapowiada w komunikacie
    # („Wykonuję: winget install …"). Rozróżniamy je po tym, że wywołanie stoi
    # na początku linii, a komunikat siedzi w środku napisu.
    wywolania = [
        line.strip()
        for line in content.splitlines()
        if line.strip().startswith("winget install")
    ]
    assert wywolania, "nie znalazłem żadnego wywołania winget"

    # Każde wywołanie ma stać wewnątrz gałęzi Confirm-Step — czyli po pytaniu.
    for wywolanie in wywolania:
        przed = content.split(wywolanie, 1)[0]
        assert "Confirm-Step" in przed[-600:], (
            f"wywołanie bez poprzedzającego pytania o zgodę: {wywolanie}"
        )


def test_windows_podaje_link_gdy_brak_pythona() -> None:
    """Bez Pythona nie da się zrobić nic — komunikat ma prowadzić do rozwiązania."""
    content = powershell_source("install-windows.ps1")
    assert "python.org/downloads" in content
    assert "Add python.exe to PATH" in content
    assert "tcl/tk" in content


def test_windows_nie_wymaga_administratora() -> None:
    """Żaden krok nie podnosi uprawnień sam z siebie."""
    code = _strip_ps_strings_and_comments(powershell_source("install-windows.ps1"))
    for zakazane in ("Start-Process -Verb RunAs", "RunAs", "#Requires -RunAsAdministrator"):
        assert zakazane not in code, f"skrypt podnosi uprawnienia: {zakazane}"


def test_windows_pobiera_model_z_konfiguracji_a_nie_z_zaszytej_nazwy() -> None:
    """Zmiana OLLAMA_MODEL w .env ma wpływać na to, co instalator pobiera."""
    content = powershell_source("install-windows.ps1")
    assert "get_settings().ollama_model" in content
    # Obecność modelu sprawdzamy przez detect_ollama, nie przez parsowanie
    # `ollama list` — porównanie po fragmencie nazwy dawało fałszywe trafienia.
    assert "detect_ollama" in content
    assert "ollama list" not in _strip_ps_strings_and_comments(content)


# --------------------------------------------------------------------------- #
# Wariant zapasowy oddaje robotę skryptowi dedykowanemu
# --------------------------------------------------------------------------- #


def _delegation_probe(tmp_path: Path, manager: str) -> subprocess.CompletedProcess[str]:
    """Uruchom install-linux-generic.sh w PATH, w którym istnieje tylko `manager`.

    Skrypty docelowe podmieniamy na atrapy wypisujące swoją nazwę — interesuje
    nas wyłącznie to, KTÓRY z nich zostaje uruchomiony.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / manager).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (fake_bin / manager).chmod(0o755)

    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir(exist_ok=True)
    shutil.copy(SCRIPTS_DIR / "install-linux-generic.sh", fake_scripts)
    for name in ("install-apt.sh", "install-pacman.sh"):
        stub = fake_scripts / name
        stub.write_text(
            f'#!/usr/bin/env bash\necho "URUCHOMIONO {name} ARGS=$*"\n', encoding="utf-8"
        )
        stub.chmod(0o755)

    # PATH musi zawierać WYŁĄCZNIE atrapę tego jednego menedżera. Dołożenie
    # /usr/bin „żeby był bash" przepuszcza pacmana tej maszyny i test mierzy
    # środowisko zamiast skryptu — sprawdzone, właśnie tak przechodził fałszywie.
    # Dlatego potrzebne narzędzia dowiązujemy pojedynczo.
    for narzedzie in ("bash", "dirname"):
        zrodlo = shutil.which(narzedzie)
        assert zrodlo is not None, f"brak {narzedzie} — nie da się uruchomić testu"
        link = fake_bin / narzedzie
        if not link.exists():
            link.symlink_to(zrodlo)

    return subprocess.run(
        [str(fake_scripts / "install-linux-generic.sh"), "--yes", "--dev"],
        capture_output=True,
        text=True,
        env={"PATH": str(fake_bin), "HOME": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_wariant_zapasowy_oddaje_robote_pacmanowi(tmp_path: Path) -> None:
    """Na Archu `install-pacman.sh` zna CUDA i pakiety z AUR-a, których generic nie ma.

    Dwie różne obsługi tego samego menedżera dawałyby gorszą instalację
    zależnie od tego, który skrypt ktoś uruchomił.
    """
    result = _delegation_probe(tmp_path, "pacman")

    assert "URUCHOMIONO install-pacman.sh" in result.stdout
    assert "URUCHOMIONO install-apt.sh" not in result.stdout
    # Argumenty mają przejść dalej bez zmian.
    assert "--yes" in result.stdout and "--dev" in result.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_wariant_zapasowy_oddaje_robote_aptowi(tmp_path: Path) -> None:
    result = _delegation_probe(tmp_path, "apt-get")

    assert "URUCHOMIONO install-apt.sh" in result.stdout
    assert "URUCHOMIONO install-pacman.sh" not in result.stdout
    assert "--yes" in result.stdout and "--dev" in result.stdout


def test_wariant_zapasowy_zna_pozostale_menedzery() -> None:
    """dnf, zypper i apk mają być obsłużone na miejscu — dla nich nie ma osobnego pliku."""
    content = (SCRIPTS_DIR / "install-linux-generic.sh").read_text(encoding="utf-8")
    for manager in ("dnf", "zypper", "apk"):
        assert f'command -v {manager} ' in content, f"brak obsługi {manager}"
    # ...i mają znać nagłówki, bo dla części z nich nie ma gotowego koła.
    assert "portaudio-devel" in content


# --------------------------------------------------------------------------- #
# Wykrywanie sieci: sprawdzamy adres, z którego pip FAKTYCZNIE skorzysta
# --------------------------------------------------------------------------- #


def _has_network(env_extra: dict[str, str]) -> bool:
    """Uruchom `has_network` z install-common.sh w zadanym środowisku.

    Funkcję wycinamy z pliku zamiast ładować cały skrypt: `install-common.sh`
    pod `set -euo pipefail` oczekuje ustawionych zmiennych nakładki, a nas
    interesuje jedna funkcja.
    """
    script = f'''
VENV_PYTHON="{sys.executable}"
eval "$(sed -n "/^has_network() {{/,/^}}/p" "{SCRIPTS_DIR}/install-common.sh")"
has_network
'''
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, **env_extra},
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


@pytest.mark.skipif(not shutil.which("bash"), reason="wymaga basha")
def test_wykrywanie_sieci_pyta_o_skonfigurowane_lustro() -> None:
    """W firmie z własnym lustrem PyPI bywa nieosiągalne, a pakiety — w zasięgu.

    Zaszyte `pypi.org` dawało wtedy zejście w tryb offline i komunikat o pustym
    magazynie kół na maszynie z pełnym dostępem do swojego repozytorium.
    """
    assert _has_network({"PIP_INDEX_URL": "https://nie.ma.takiego.hosta.invalid/simple"}) is False


def test_wykrywanie_sieci_nie_zaszywa_pypi_w_kodzie() -> None:
    content = (SCRIPTS_DIR / "install-common.sh").read_text(encoding="utf-8")
    fragment = content.split("has_network() {", 1)[1].split("\n}", 1)[0]
    assert "PIP_INDEX_URL" in fragment, "test sieci ignoruje skonfigurowane lustro"


# --------------------------------------------------------------------------- #
# scripts/install-rvc.sh (Faza 15)
#
# Osobny skrypt, bo osobny problem: buduje DRUGIE środowisko Pythona wyłącznie
# dla RVC. Nie jest nakładką na install-common.sh i celowo nie ma go w
# UNIX_INSTALLERS — nie instaluje asystenta, tylko dokłada do niego głos.
# --------------------------------------------------------------------------- #

RVC_INSTALLER = SCRIPTS_DIR / "install-rvc.sh"


def test_install_rvc_ma_shebang_i_prawo_wykonania() -> None:
    assert RVC_INSTALLER.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert RVC_INSTALLER.stat().st_mode & stat.S_IXUSR


def test_install_rvc_nie_potrzebuje_roota() -> None:
    """Drugi venv powstaje w katalogu projektu — nic tu nie wymaga uprawnień.

    Szukamy WYWOŁANIA sudo, nie samego słowa: skrypt ma prawo wypisać
    podpowiedź „zainstaluj Pythona jako root", i to nie czyni z niego czegoś,
    co samo sięga po uprawnienia.
    """
    kod = strip_comments(RVC_INSTALLER.read_text(encoding="utf-8"))
    wywolania = [
        line
        for line in kod.splitlines()
        if re.search(r"(^|[;&|(]\s*|\$\()\s*sudo\b", line)
    ]
    assert not wywolania, f"skrypt wywołuje sudo: {wywolania}"


def test_install_rvc_nie_zaklada_wersji_po_nazwie_pliku() -> None:
    """`python3.10` w PATH bywa dowiązaniem do czegoś innego — trzeba URUCHOMIĆ kandydata."""
    kod = RVC_INSTALLER.read_text(encoding="utf-8")
    assert "sys.version_info" in kod, "skrypt nie sprawdza wersji uruchomieniem interpretera"


def test_install_rvc_tlumaczy_dlaczego_cofa_pipa() -> None:
    """Przypięte wersje bez powodu to dług; z powodem w pliku — decyzja.

    Powody są konkretne i sprawdzone: fairseq nie importuje się na 3.11+,
    omegaconf 2.0.6 ma metadane odrzucane przez pip 24.1, pyworld potrzebuje
    pkg_resources usuniętego w setuptools 81.
    """
    kod = RVC_INSTALLER.read_text(encoding="utf-8")
    for powod in ("fairseq", "omegaconf", "pyworld", "pkg_resources"):
        assert powod in kod, f"brak wyjaśnienia, czemu przypięto coś z powodu: {powod}"


def test_install_rvc_pokazuje_pomoc_bez_dotykania_systemu() -> None:
    wynik = subprocess.run(
        ["bash", str(RVC_INSTALLER), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert wynik.returncode == 0
    assert "install-rvc.sh" in wynik.stdout


def test_install_rvc_odrzuca_zly_interpreter_z_jasnym_komunikatem(tmp_path: Path) -> None:
    """Wskazanie Pythona w złej wersji ma się skończyć zdaniem, a nie stosem wywołań."""
    wynik = subprocess.run(
        ["bash", str(RVC_INSTALLER), "--python", sys.executable],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=str(tmp_path),
    )
    wersja = f"{sys.version_info.major}.{sys.version_info.minor}"
    if wersja == "3.10":  # pragma: no cover - CI na dokładnie tej wersji
        pytest.skip("ten interpreter JEST wersją, której skrypt szuka")
    assert wynik.returncode != 0
    assert "3.10" in wynik.stdout
    # Najważniejsze zdanie w całym skrypcie: brak RVC nie oznacza braku mowy.
    assert "Pipera" in wynik.stdout


def test_install_rvc_mowi_co_zostalo_do_zrobienia_recznie() -> None:
    """Model i wpis w ustawieniach to dwie rzeczy, których skrypt zrobić nie może."""
    kod = RVC_INSTALLER.read_text(encoding="utf-8")
    assert "models/rvc/" in kod
    assert "voice_engine" in kod
    assert "--check-deps" in kod
