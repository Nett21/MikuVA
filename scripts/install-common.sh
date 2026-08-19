#!/usr/bin/env bash
# Wspólna logika instalatorów dla systemów uniksowych.
#
# Ten plik nie jest uruchamiany bezpośrednio — dołączają go cienkie nakładki
# (install-apt.sh, install-pacman.sh, install-linux-generic.sh,
# install-macos.sh), które ustawiają tylko to, co różni menedżery pakietów.
# Dzięki temu logika instalacji istnieje w JEDNYM miejscu i nie rozjeżdża się
# między dystrybucjami.
#
# Nakładka ustawia przed dołączeniem:
#   PKG_LABEL          — nazwa menedżera do komunikatów (np. "pacman")
#   PKG_INSTALL        — polecenie instalacji jako tablica (np. (pacman -S --needed))
#   PKG_PYTHON         — pakiety dające Pythona z modułem venv i pip
#   PKG_AUDIO          — pakiety PortAudio (opcjonalne; koła sounddevice zwykle
#                        niosą własną kopię biblioteki)
#   PKG_OLLAMA         — pakiet Ollamy albo puste, gdy dystrybucja go nie ma
#   PKG_TK             — pakiet z biblioteką Tk dla GUI (Faza 10); pusty, gdy
#                        dystrybucja dostarcza ją razem z Pythonem
#   PKG_GPU            — pakiety CUDA/cuDNN dla trybu --full (opcjonalne)
#   PKG_NEEDS_ROOT     — 1 (domyślnie) albo 0 dla menedżerów działających
#                        bez sudo (Homebrew)
#
# Nic nie jest instalowane bez pytania. Każde polecenie z podniesionymi
# uprawnieniami jest najpierw wypisywane, a użytkownik je zatwierdza (albo
# uruchamia sam). Skrypt NIGDY nie wykonuje `curl | sh`.

set -euo pipefail

# Wartości domyślne, gdy nakładka czegoś nie ustawiła (albo gdy dystrybucja
# nie ma menedżera pakietów) — inaczej `set -u` przerwałby skrypt.
PKG_LABEL="${PKG_LABEL:-}"
PKG_PYTHON="${PKG_PYTHON:-}"
PKG_AUDIO="${PKG_AUDIO:-}"
PKG_OLLAMA="${PKG_OLLAMA:-}"
PKG_TK="${PKG_TK:-}"
# Nagłówki PortAudio — potrzebne WYŁĄCZNIE tam, gdzie nie ma gotowego koła
# `sounddevice` i pip musi kompilować. Domyślnie puste: dokładanie pakietów
# -dev „na wszelki wypadek" wciąga kompilator na maszynę, która go nie chce.
PKG_AUDIO_BUILD="${PKG_AUDIO_BUILD:-}"
# ffmpeg — opcjonalny (patrz komentarz w install-apt.sh).
PKG_FFMPEG="${PKG_FFMPEG:-}"
# Pakiety OPCJONALNE z AUR-a (tylko Arch). Puste wszędzie indziej.
PKG_AUR_OPTIONAL="${PKG_AUR_OPTIONAL:-}"
# Pakiety CUDA/cuDNN dla trybu pełnego. Puste tam, gdzie nazw nie znamy albo
# gdzie nie mają sensu (macOS liczy na Metalu, bez żadnego pakietu).
PKG_GPU="${PKG_GPU:-}"
# Homebrew instaluje do katalogu użytkownika i wprost odmawia pracy pod sudo —
# dlatego podnoszenie uprawnień jest sterowane, a nie zakładane.
PKG_NEEDS_ROOT="${PKG_NEEDS_ROOT:-1}"
declare -p PKG_INSTALL >/dev/null 2>&1 || PKG_INSTALL=()

TAG_SYSTEM="[SYSTEM]"
TAG_ERROR="[ERROR]"

# BASH_SOURCE[1] to nakładka, która nas dołączyła — nazwa i katalog muszą
# pochodzić od niej, nie od tego pliku.
INSTALLER_NAME="$(basename "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")/.." && pwd)"
VENV_DIR="${MIKU_VENV_DIR:-$PROJECT_ROOT/.venv}"
WHEELHOUSE="${MIKU_WHEELHOUSE_DIR:-$PROJECT_ROOT/vendor/wheels}"
REQUIRED_PYTHON_MINOR=12

ASSUME_YES=0
WITH_DEV=0
SKIP_SYSTEM=0
FORCE_OFFLINE=0
FULL=0

usage() {
    cat <<USAGE
Instalator asystenta (${PKG_LABEL:-bez menedżera pakietów}).

Użycie: ./scripts/$INSTALLER_NAME [opcje]

  -y, --yes         nie pytaj o potwierdzenie (tryb nieinteraktywny)
      --full        WSZYSTKO: pakiety opcjonalne, modele, CUDA (jeśli jest
                    karta NVIDIA) i narzędzia deweloperskie
      --dev         zainstaluj też zależności deweloperskie (testy, ruff, mypy)
      --no-system   pomiń pakiety systemowe, zrób samo środowisko Pythona
      --offline     instaluj z vendor/wheels, bez sięgania do sieci
  -h, --help        ta pomoc

Skrypt nie instaluje niczego bez pytania i wypisuje każde polecenie,
które zamierza wykonać z podniesionymi uprawnieniami.
USAGE
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes) ASSUME_YES=1 ;;
            # Pełna instalacja obejmuje narzędzia deweloperskie — „wszystko"
            # ma znaczyć wszystko, bez drugiej flagi do zapamiętania.
            --full) FULL=1; WITH_DEV=1 ;;
            --dev) WITH_DEV=1 ;;
            --no-system) SKIP_SYSTEM=1 ;;
            --offline) FORCE_OFFLINE=1 ;;
            -h|--help) usage; exit 0 ;;
            *) echo "$TAG_ERROR Nieznana opcja: $1" >&2; usage >&2; exit 2 ;;
        esac
        shift
    done
}

info() { echo "$TAG_SYSTEM $*"; }
fail() { echo "$TAG_ERROR $*" >&2; }

confirm() {
    # Pytanie tak/nie. W trybie --yes zawsze „tak"; bez terminala zawsze „nie",
    # żeby skrypt uruchomiony z potoku nigdy nie instalował niczego po cichu.
    local question="$1"
    if [[ "$ASSUME_YES" == "1" ]]; then
        info "$question — tak (--yes)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        info "$question — pomijam (brak terminala, użyj --yes)"
        return 1
    fi
    local answer
    read -r -p "$TAG_SYSTEM $question [t/N] " answer
    [[ "$answer" =~ ^([tT]|[tT][aA][kK]|[yY]|[yY][eE][sS])$ ]]
}

privileged_prefix() {
    # Root nie potrzebuje sudo; bez sudo nie zgadujemy niczego innego.
    if [[ "$(id -u)" == "0" ]]; then
        return 0
    fi
    if command -v sudo >/dev/null 2>&1; then
        echo "sudo"
        return 0
    fi
    return 1
}

run_privileged() {
    local prefix
    if ! prefix="$(privileged_prefix)"; then
        fail "Potrzebne są uprawnienia administratora, a nie znalazłem polecenia sudo."
        info "Uruchom ręcznie: $*"
        return 1
    fi
    info "Wykonuję: ${prefix:+$prefix }$*"
    # shellcheck disable=SC2086 - prefix jest pusty albo dokładnie "sudo"
    ${prefix} "$@"
}

# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

python_is_new_enough() {
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, $REQUIRED_PYTHON_MINOR) else 1)" \
        >/dev/null 2>&1
}

find_python() {
    # Kolejność od najnowszych: chcemy Pythona >= 3.12, a nie „jakiegokolwiek".
    local candidate
    for candidate in python3.14 python3.13 python3.12 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && python_is_new_enough "$candidate"; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# --------------------------------------------------------------------------- #
# Pakiety systemowe
# --------------------------------------------------------------------------- #

run_package_manager() {
    # Instalacja pakietów właściwym trybem uprawnień dla tego menedżera.
    if [[ "$PKG_NEEDS_ROOT" == "1" ]]; then
        run_privileged "$@"
        return $?
    fi
    info "Wykonuję: $*"
    "$@"
}

print_manual_package_list() {
    # Lista SKŁADNIKÓW, nie nazw pakietów: nazwy różnią się między
    # dystrybucjami (libportaudio2 / portaudio / portaudio-dev), a składniki
    # są wszędzie te same. Zgadywanie nazwy dla nieznanego menedżera kończy się
    # poleceniem, które nie działa — a to gorsze niż brak polecenia.
    info "Zainstaluj poniższe narzędziami swojej dystrybucji:"
    info "  · Python >= 3.$REQUIRED_PYTHON_MINOR — razem z modułem 'venv' i z 'pip'"
    info "      (na części dystrybucji to trzy osobne pakiety: python3, python3-venv, python3-pip)"
    info "  · PortAudio — biblioteka wykonawcza pod 'sounddevice' (mikrofon i głośnik)"
    info "      (nazwy spotykane: libportaudio2, portaudio, portaudio-dev)"
    info "  · Tk — biblioteka pod interfejs graficzny; POMIŃ, jeśli wystarczy Ci terminal"
    info "      (nazwy spotykane: tk, python3-tk, python3-tkinter)"
    info "  · ffmpeg — OPCJONALNY; przydaje się przy pracy z plikami dźwiękowymi"
    info "  · Ollama — serwer modelu językowego: https://ollama.com/download"
    info "Nic z tego nie jest zgadywane przez ten skrypt, bo nazwy pakietów"
    info "różnią się między dystrybucjami, a błędne polecenie jest gorsze niż jego brak."
    info "Potem uruchom ponownie: ./scripts/$INSTALLER_NAME --no-system"
}

install_system_packages() {
    if [[ "$SKIP_SYSTEM" == "1" ]]; then
        info "Pomijam pakiety systemowe (--no-system)."
        return 0
    fi
    if [[ -z "$PKG_LABEL" || ${#PKG_INSTALL[@]} -eq 0 ]]; then
        # Nie zgadujemy menedżera pakietów. Zamiast tego mówimy, CZEGO potrzeba —
        # nazwy pakietów różnią się między dystrybucjami, ale składniki nie.
        info "Nie wykryto obsługiwanego menedżera pakietów — pomijam krok systemowy."
        print_manual_package_list
        return 0
    fi

    local packages=()
    if ! find_python >/dev/null; then
        packages+=(${PKG_PYTHON})
    else
        info "Python >= 3.$REQUIRED_PYTHON_MINOR już jest — nie ruszam pakietów Pythona."
    fi

    # Tk (GUI, Faza 10) to biblioteka SYSTEMOWA — pip jej nie zainstaluje.
    # Dokładamy ją tylko wtedy, gdy `import tkinter` faktycznie nie działa:
    # część dystrybucji (i Python z python.org) niesie ją razem z Pythonem.
    if [[ -n "$PKG_TK" ]]; then
        local python_bin
        if python_bin="$(find_python)" && ! "$python_bin" -c "import tkinter" >/dev/null 2>&1; then
            packages+=(${PKG_TK})
        fi
    fi

    # PortAudio bierze się zwykle z koła sounddevice; pakiet systemowy to
    # zabezpieczenie dla dystrybucji, dla których koła nie ma. Sprawdzamy tylko
    # tam, gdzie jest ldconfig — na macOS-ie takiego rejestru nie ma.
    if [[ -n "$PKG_AUDIO" ]]; then
        if command -v ldconfig >/dev/null 2>&1; then
            ldconfig -p 2>/dev/null | grep -q portaudio || packages+=(${PKG_AUDIO})
        else
            packages+=(${PKG_AUDIO})
        fi
    fi

    # ffmpeg: dokładamy tylko wtedy, gdy nie ma go w PATH. Jest OPCJONALNY —
    # nic, co asystent robi dzisiaj, nie woła go bezpośrednio.
    if [[ -n "$PKG_FFMPEG" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
        packages+=(${PKG_FFMPEG})
    fi

    if [[ ${#packages[@]} -eq 0 ]]; then
        info "Wszystkie pakiety systemowe są już na miejscu."
        return 0
    fi

    info "Do zainstalowania przez $PKG_LABEL: ${packages[*]}"
    if confirm "Zainstalować te pakiety?"; then
        run_package_manager "${PKG_INSTALL[@]}" "${packages[@]}" || {
            fail "Instalacja pakietów systemowych nie powiodła się — idę dalej."
            return 0
        }
    else
        info "Pomijam pakiety systemowe. Możesz je zainstalować sam:"
        info "  ${PKG_INSTALL[*]} ${packages[*]}"
    fi
}

install_ollama() {
    if [[ "$SKIP_SYSTEM" == "1" ]]; then
        return 0
    fi
    if command -v ollama >/dev/null 2>&1; then
        info "Ollama jest już zainstalowana: $(command -v ollama)"
        return 0
    fi

    if [[ -n "$PKG_OLLAMA" && ${#PKG_INSTALL[@]} -gt 0 ]]; then
        info "Model językowy uruchamia Ollama (pakiet: $PKG_OLLAMA)."
        if confirm "Zainstalować Ollamę przez $PKG_LABEL?"; then
            run_package_manager "${PKG_INSTALL[@]}" "$PKG_OLLAMA" || \
                fail "Nie udało się zainstalować Ollamy — zrób to ręcznie."
            return 0
        fi
    fi

    # Świadomie NIE uruchamiamy `curl ... | sh`: pobieranie i wykonywanie
    # skryptu w jednym kroku uniemożliwia obejrzenie, co się wykona.
    info "Ollamy nie ma w repozytoriach tego systemu. Zainstaluj ją samodzielnie:"
    info "  https://ollama.com/download  (pakiet dla dystrybucji albo oficjalny instalator)"
    info "Bez Ollamy asystent uruchomi się, ale rozmowa nie ruszy."
}

# --------------------------------------------------------------------------- #
# Środowisko Pythona
# --------------------------------------------------------------------------- #

setup_venv() {
    local python_bin
    if ! python_bin="$(find_python)"; then
        fail "Nie znalazłem Pythona >= 3.$REQUIRED_PYTHON_MINOR."
        info "Zainstaluj go i uruchom ten skrypt ponownie."
        exit 1
    fi
    info "Python: $python_bin ($("$python_bin" --version 2>&1))"

    if [[ ! -d "$VENV_DIR" ]]; then
        info "Tworzę środowisko wirtualne: $VENV_DIR"
        # BEZ `set -e`: nieudane `venv` to najczęstsza porażka na Debianie
        # i Ubuntu (moduł venv jest tam osobnym pakietem). Gołe przerwanie
        # skryptu pokazuje wtedy ślad Pythona i ani słowa o tym, co zrobić.
        if ! "$python_bin" -m venv "$VENV_DIR"; then
            fail "Nie udało się utworzyć środowiska w $VENV_DIR."
            if [[ -n "$PKG_PYTHON" && "$PKG_PYTHON" == *venv* ]]; then
                info "Na tym systemie moduł venv jest osobnym pakietem:"
                info "  ${PKG_INSTALL[*]:-<menedżer pakietów>} ${PKG_PYTHON}"
            else
                info "Sprawdź, czy Python ma moduł venv: $python_bin -m venv --help"
            fi
            info "Częsta przyczyna: brak miejsca na dysku albo brak prawa zapisu do $VENV_DIR."
            return 1
        fi
    else
        info "Środowisko wirtualne już istnieje: $VENV_DIR"
    fi

    # Układ katalogów venv zależy od systemu, nie od nas: POSIX kładzie
    # interpreter w `bin/`, a venv utworzony na Windowsie — w `Scripts/`.
    # Ten skrypt działa tylko na systemach uniksowych, ale katalog projektu
    # bywa współdzielony (dysk sieciowy, dual boot), więc sprawdzamy oba.
    local candidate
    VENV_PYTHON=""
    for candidate in "$VENV_DIR/bin/python" "$VENV_DIR/bin/python3" "$VENV_DIR/Scripts/python.exe"; do
        if [[ -x "$candidate" ]]; then
            VENV_PYTHON="$candidate"
            break
        fi
    done
    if [[ -z "$VENV_PYTHON" ]]; then
        fail "Środowisko $VENV_DIR jest uszkodzone (nie ma w nim interpretera)."
        info "Usuń ten katalog i uruchom skrypt ponownie: rm -rf '$VENV_DIR'"
        return 1
    fi
}

wheelhouse_has_packages() {
    [[ -d "$WHEELHOUSE" ]] && compgen -G "$WHEELHOUSE/*" >/dev/null 2>&1
}

install_python_packages() {
    local pip_args=(-m pip install)
    if [[ "$FORCE_OFFLINE" == "1" ]] || { wheelhouse_has_packages && ! has_network; }; then
        if ! wheelhouse_has_packages; then
            fail "Tryb offline, a magazyn kół $WHEELHOUSE jest pusty."
            info "Na maszynie z internetem: python scripts/prepare_offline.py --wheels"
            note_missing "pakiety Pythona — pusty magazyn kół $WHEELHOUSE"
            return 1
        fi
        info "Instaluję pakiety z $WHEELHOUSE (bez sieci)."
        pip_args+=(--no-index --find-links "$WHEELHOUSE")
    else
        info "Instaluję pakiety z PyPI."
        "$VENV_PYTHON" -m pip install --upgrade pip >/dev/null || true
    fi

    # Instalacja pakietów to najczęstsze miejsce awarii (zerwana sieć, pakiet
    # bez koła dla tej wersji Pythona, brak kompilatora). Awaria NIE może uciąć
    # skryptu: użytkownik ma zobaczyć podsumowanie i raport --check-deps, z
    # którego wynika, czego dokładnie brakuje.
    if ! "$VENV_PYTHON" "${pip_args[@]}" -r "$PROJECT_ROOT/requirements.txt"; then
        fail "Instalacja pakietów z requirements.txt nie powiodła się."
        note_missing "pakiety z requirements.txt — powtórz: $VENV_PYTHON -m pip install -r requirements.txt"
        if [[ -n "$PKG_AUDIO_BUILD" ]]; then
            # Najczęstsza przyczyna na dystrybucji bez gotowego koła: pip próbuje
            # zbudować `sounddevice`/`webrtcvad` i nie znajduje nagłówków.
            info "Jeśli błąd dotyczy budowania pakietu audio, brakuje nagłówków:"
            info "  ${PKG_INSTALL[*]:-<menedżer pakietów>} ${PKG_AUDIO_BUILD}"
        fi
        info "Idę dalej — raport na końcu pokaże, czego brakuje."
    fi
    if [[ "$WITH_DEV" == "1" ]]; then
        if ! "$VENV_PYTHON" "${pip_args[@]}" -r "$PROJECT_ROOT/requirements-dev.txt"; then
            fail "Instalacja pakietów deweloperskich nie powiodła się."
            note_missing "pakiety z requirements-dev.txt (pytest, ruff, mypy)"
        fi
    fi

    install_optional_packages
}

# --------------------------------------------------------------------------- #
# Pakiety opcjonalne
# --------------------------------------------------------------------------- #
#
# Każdy z nich włącza JEDNĄ funkcję i niczego więcej nie psuje, gdy go zabraknie.
# Instalujemy je pojedynczo i po cichu przełykamy błąd: część nie ma kół dla
# każdej wersji Pythona, a nieudana kompilacja jednego pakietu nie może
# przerwać instalacji reszty. To, czego nie udało się zainstalować, trafia do
# podsumowania na końcu — razem z poleceniem do powtórzenia.

# Wpis: pakiet|co daje|czego brak, gdy go nie ma
OPTIONAL_BASE=(
    "webrtcvad-wheels|lepsza detekcja mowy (VAD)|zadziała detektor energetyczny"
    "piper-tts|synteza mowy w procesie asystenta|odpowiedzi zostaną tekstowe"
)
OPTIONAL_FULL=(
    "pypdf|czytanie PDF-ów (narzędzia pdf.read i pdf.search)|te dwa narzędzia będą niedostępne"
    "psutil|lista i zamykanie procesów poza Linuksem|na Linuksie zastępuje je /proc"
    "openwakeword|wykrywanie frazy budzącej modelem KWS|zadziała detektor na Whisperze"
    # Szybsze wyszukiwanie wektorowe. Bez niego liczy NumPy — przy skali
    # osobistego asystenta (10^4-10^5 wektorów) różnicy się nie zauważy.
    "faiss-cpu|szybsze wyszukiwanie w pamięci semantycznej|policzy NumPy"
)
# Osobno, bo ciągnie PyTorcha (kilka GB). Alternatywa bez tego pakietu:
# embeddingi liczy Ollama, która i tak musi działać.
OPTIONAL_HEAVY="sentence-transformers"

# Czego nie udało się zainstalować — wypisywane na końcu, z poleceniem naprawy.
MISSING_REPORT=()

note_missing() { MISSING_REPORT+=("$1"); }

pip_try() {
    # Instalacja „na próbę": sukces albo wpis w podsumowaniu. Nigdy błąd krytyczny.
    local package="$1" gives="$2" without="$3"
    if "$VENV_PYTHON" -m pip install "$package" >/dev/null 2>&1; then
        info "Zainstalowano $package — $gives."
        return 0
    fi
    info "$package niedostępny dla tej wersji Pythona — $without."
    note_missing "$package  →  $VENV_PYTHON -m pip install $package"
    return 1
}

install_optional_packages() {
    if [[ "$FORCE_OFFLINE" == "1" ]]; then
        info "Tryb offline — pomijam pakiety opcjonalne (są w vendor/wheels albo ich nie ma)."
        return 0
    fi

    local entry package gives without
    local -a wanted=("${OPTIONAL_BASE[@]}")
    if [[ "$FULL" == "1" ]]; then
        wanted+=("${OPTIONAL_FULL[@]}")
    fi

    for entry in "${wanted[@]}"; do
        IFS='|' read -r package gives without <<<"$entry"
        pip_try "$package" "$gives" "$without" || true
    done

    if [[ "$FULL" != "1" ]]; then
        return 0
    fi

    # PyTorch to kilka gigabajtów — pytamy osobno, nawet w trybie pełnym.
    info "Pozostał $OPTIONAL_HEAVY: lokalne embeddingi bez Ollamy, ale pobiera PyTorcha (kilka GB)."
    info "Bez niego embeddingi policzy Ollama modelem nomic-embed-text (~270 MB)."
    if confirm "Zainstalować $OPTIONAL_HEAVY?"; then
        pip_try "$OPTIONAL_HEAVY" "lokalne embeddingi" "embeddingi policzy Ollama" || true
    else
        info "Pomijam $OPTIONAL_HEAVY — zadba o to Ollama."
    fi
}

# --------------------------------------------------------------------------- #
# Akceleracja GPU (tryb pełny)
# --------------------------------------------------------------------------- #

has_nvidia_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

install_gpu_packages() {
    # Bez karty NVIDII nie ma czego przyspieszać: Whisper i tak policzy na CPU,
    # a na macOS-ie liczy Metal, który nie wymaga żadnego pakietu.
    if [[ "$FULL" != "1" || "$SKIP_SYSTEM" == "1" ]]; then
        return 0
    fi
    if ! has_nvidia_gpu; then
        info "Nie wykryto karty NVIDIA — pomijam pakiety CUDA (rozpoznawanie mowy pójdzie na CPU)."
        return 0
    fi
    if [[ -z "$PKG_GPU" || ${#PKG_INSTALL[@]} -eq 0 ]]; then
        # Nazw nie zgadujemy: na Debianie i Fedorze cuDNN bierze się z repozytorium
        # NVIDII, a nie z repozytorium dystrybucji. Zła nazwa pakietu byłaby
        # gorsza niż jej brak — instalator powiedziałby, że coś zrobił.
        info "Wykryto kartę NVIDIA, ale nazw pakietów CUDA dla tego systemu nie zgaduję."
        info "Whisper potrzebuje CUDA i cuDNN: https://developer.nvidia.com/cudnn-downloads"
        info "Bez nich rozpoznawanie mowy działa dalej, tylko liczy na procesorze."
        note_missing "CUDA + cuDNN  →  https://developer.nvidia.com/cudnn-downloads"
        return 0
    fi

    info "Karta NVIDIA jest — pakiety $PKG_GPU dają Whisperowi liczenie na GPU."
    if confirm "Zainstalować $PKG_GPU przez $PKG_LABEL?"; then
        run_package_manager "${PKG_INSTALL[@]}" ${PKG_GPU} || {
            fail "Instalacja pakietów CUDA nie powiodła się — Whisper policzy na CPU."
            note_missing "$PKG_GPU  →  ${PKG_INSTALL[*]} $PKG_GPU"
        }
    else
        info "Pomijam CUDA. Później: ${PKG_INSTALL[*]} $PKG_GPU"
        note_missing "$PKG_GPU  →  ${PKG_INSTALL[*]} $PKG_GPU"
    fi
}

# --------------------------------------------------------------------------- #
# Modele (tryb pełny)
# --------------------------------------------------------------------------- #

fetch_models() {
    # Modele to nie pakiety — pip ich nie zainstaluje. Każdy pobieramy osobno,
    # żeby nieudane pobranie jednego nie zabrało pozostałych.
    if [[ "$FULL" != "1" ]]; then
        return 0
    fi
    if [[ "$FORCE_OFFLINE" == "1" ]]; then
        info "Tryb offline — modele muszą już być na dysku (prepare_offline.py na maszynie z siecią)."
        return 0
    fi

    local prepare="$PROJECT_ROOT/scripts/prepare_offline.py"

    info "Pobieram model rozpoznawania mowy (Whisper) — kilkaset MB."
    (cd "$PROJECT_ROOT" && "$VENV_PYTHON" "$prepare" --whisper) || {
        fail "Nie udało się pobrać modelu Whispera."
        note_missing "model Whispera  →  $VENV_PYTHON scripts/prepare_offline.py --whisper"
    }

    info "Pobieram głos do syntezy mowy (Piper)."
    (cd "$PROJECT_ROOT" && "$VENV_PYTHON" "$prepare" --piper) || {
        fail "Nie udało się pobrać głosu Pipera."
        note_missing "głos Pipera  →  $VENV_PYTHON scripts/prepare_offline.py --piper"
    }

    # Embeddingi: albo lokalnie (sentence-transformers), albo przez Ollamę.
    if "$VENV_PYTHON" -c "import sentence_transformers" >/dev/null 2>&1; then
        info "Pobieram model embeddingów do pamięci semantycznej."
        (cd "$PROJECT_ROOT" && "$VENV_PYTHON" "$prepare" --embeddings) || {
            fail "Nie udało się pobrać modelu embeddingów."
            note_missing "model embeddingów  →  $VENV_PYTHON scripts/prepare_offline.py --embeddings"
        }
    elif command -v ollama >/dev/null 2>&1; then
        info "Pobieram model embeddingów Ollamy: nomic-embed-text"
        ollama pull nomic-embed-text || {
            fail "Nie udało się pobrać nomic-embed-text."
            note_missing "nomic-embed-text  →  ollama pull nomic-embed-text"
        }
    else
        note_missing "nomic-embed-text  →  ollama pull nomic-embed-text (po instalacji Ollamy)"
    fi
}

report_missing() {
    if [[ ${#MISSING_REPORT[@]} -eq 0 ]]; then
        return 0
    fi
    echo
    info "Nie udało się (albo pominięto) — każdą z tych rzeczy można dorobić później:"
    local item
    for item in "${MISSING_REPORT[@]}"; do
        info "  · $item"
    done
}

has_network() {
    # Krótki test bez zewnętrznych narzędzi: gniazdo do repozytorium pakietów.
    #
    # Sprawdzamy adres, z którego pip FAKTYCZNIE będzie korzystał, a nie zaszyte
    # `pypi.org`. W firmie z własnym lustrem (PIP_INDEX_URL) PyPI bywa
    # nieosiągalne, choć pakiety są w zasięgu ręki — sprawdzanie nie tego hosta
    # kończyło się zejściem w tryb offline i komunikatem o pustym magazynie kół
    # na maszynie, która ma pełny dostęp do swojego lustra.
    "$VENV_PYTHON" - <<'PYTHON' >/dev/null 2>&1
import os
import socket
import sys
from urllib.parse import urlsplit

index = (
    os.environ.get("PIP_INDEX_URL")
    or os.environ.get("PIP_EXTRA_INDEX_URL")
    or "https://pypi.org/simple"
)
parts = urlsplit(index)
host = parts.hostname or "pypi.org"
port = parts.port or (443 if parts.scheme == "https" else 80)

socket.setdefaulttimeout(3)
try:
    socket.create_connection((host, port)).close()
except OSError:
    sys.exit(1)
PYTHON
}

setup_env_file() {
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        info "Plik .env już istnieje — nie ruszam go."
        return 0
    fi
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    info "Utworzono .env na podstawie .env.example"
}

pull_language_model() {
    if ! command -v ollama >/dev/null 2>&1; then
        return 0
    fi
    # Stan sprawdzamy po HTTP przez config.detect_ollama, a nie przez
    # `ollama list`: to drugie potrafi wystartować demona, którego logi lecą
    # potem na nasze wyjście, a otwarty potok grozi zawieszeniem skryptu.
    local status model present
    # Import `config` wymaga katalogu projektu — skrypt może być wywołany z dowolnego miejsca.
    status="$(cd "$PROJECT_ROOT" && "$VENV_PYTHON" - <<'PYTHON' 2>/dev/null || true
from config import detect_ollama, get_settings

settings = get_settings()
print(settings.ollama_model)
print("1" if detect_ollama(settings).model_present else "0")
PYTHON
)"
    model="$(printf '%s\n' "$status" | sed -n 1p)"
    present="$(printf '%s\n' "$status" | sed -n 2p)"
    [[ -z "$model" ]] && return 0

    if [[ "$present" == "1" ]]; then
        info "Model językowy '$model' jest już pobrany."
        return 0
    fi
    info "Model językowy '$model' nie jest jeszcze pobrany (kilka GB)."
    if confirm "Pobrać go teraz poleceniem 'ollama pull $model'?"; then
        ollama pull "$model" || fail "Pobieranie modelu nie powiodło się — powtórz później."
    else
        info "Pobierzesz go później: ollama pull $model"
    fi
}

aur_helper() {
    # Zwróć nazwę zainstalowanego pomocnika AUR albo nic.
    #
    # Kolejność: paru, potem yay — obie są równoważne, więc bierzemy pierwszą,
    # którą użytkownik faktycznie ma. NIE instalujemy żadnej z nich: `paru`
    # i `yay` same pochodzą z AUR-a, więc zainstalowanie ich w tle oznaczałoby
    # zbudowanie i uruchomienie obcego kodu bez pytania — dokładnie to, czego
    # ten instalator nie robi (patrz zasada „nigdy curl | sh").
    local candidate
    for candidate in paru yay; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

install_aur_packages() {
    # Pakiety OPCJONALNE z AUR-a. Uruchamiane wyłącznie w trybie --full i tylko
    # wtedy, gdy użytkownik ma już pomocnika AUR.
    if [[ -z "$PKG_AUR_OPTIONAL" || "$SKIP_SYSTEM" == "1" || "$FULL" != "1" ]]; then
        return 0
    fi

    local helper
    if ! helper="$(aur_helper)"; then
        info "Pakiety opcjonalne z AUR-a: ${PKG_AUR_OPTIONAL}"
        info "Nie masz pomocnika AUR (paru/yay) i świadomie go nie instaluję."
        info "Gdybyś chciał je mieć, zrób to sam — jednym z dwóch sposobów:"
        info "  · przez pomocnika:  paru -S ${PKG_AUR_OPTIONAL}"
        info "  · ręcznie:          git clone https://aur.archlinux.org/<pakiet>.git"
        info "                      cd <pakiet> && makepkg -si"
        info "To są dodatki — asystent działa bez nich (Piper jako pakiet Pythona)."
        return 0
    fi

    info "Wykryto pomocnika AUR: $helper"
    info "Pakiety opcjonalne z AUR-a: ${PKG_AUR_OPTIONAL}"
    if ! confirm "Zainstalować je przez $helper?"; then
        info "Pomijam. Później: $helper -S ${PKG_AUR_OPTIONAL}"
        return 0
    fi
    # Pomocnika AUR uruchamiamy BEZ sudo — sam poprosi o hasło tam, gdzie musi.
    # `makepkg` wprost odmawia pracy na koncie roota, więc `sudo paru` byłoby
    # błędem, a nie ostrożnością.
    # shellcheck disable=SC2086 - lista pakietów ma się rozwinąć na wyrazy
    if ! "$helper" -S --needed $PKG_AUR_OPTIONAL; then
        fail "Instalacja pakietów z AUR-a nie powiodła się."
        note_missing "pakiety AUR (${PKG_AUR_OPTIONAL}) — opcjonalne, powtórz: $helper -S ${PKG_AUR_OPTIONAL}"
    fi
}

check_audio_devices() {
    # Mikrofon i głośnik to jedyne wymagania, których NIE da się doinstalować —
    # albo sprzęt jest, albo go nie ma. Dlatego pytamy o to osobno i nazywamy
    # rzecz po imieniu, zamiast zostawiać użytkownika z jedną linijką w gąszczu
    # raportu zależności.
    #
    # Detekcja NIE jest tu powtarzana: wołamy dokładnie te funkcje, których
    # używa `--check-deps` (audio/microphone.py, audio/output.py). Druga
    # implementacja rozjechałaby się z pierwszą przy pierwszej zmianie.
    info "Sprawdzam urządzenia audio (mikrofon i głośnik)."
    local output
    output="$(cd "$PROJECT_ROOT" && "$VENV_PYTHON" - <<'PYTHON' 2>/dev/null || true
import sys

try:
    from config import get_settings
except Exception as exc:  # brak pakietów = brak sensu w dalszym sprawdzaniu
    print(f"SKIP|nie da sie wczytac konfiguracji ({exc})")
    sys.exit(0)

settings = get_settings()

try:
    from audio.microphone import list_input_devices
    devices = list_input_devices(settings)
    print(f"MIC|{len(devices)}|" + (devices[0].name if devices else ""))
except Exception as exc:
    print(f"MICERR|{exc}")

try:
    from audio.output import list_output_devices
    devices = list_output_devices(settings)
    print(f"OUT|{len(devices)}|" + (devices[0].name if devices else ""))
except Exception as exc:
    print(f"OUTERR|{exc}")
PYTHON
)"

    local line kind count name
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        kind="${line%%|*}"
        case "$kind" in
            MIC|OUT)
                count="$(printf '%s' "$line" | cut -d'|' -f2)"
                name="$(printf '%s' "$line" | cut -d'|' -f3-)"
                local label="Mikrofon"
                [[ "$kind" == "OUT" ]] && label="Wyjście dźwięku"
                if [[ "$count" -gt 0 ]]; then
                    info "  $label: $count urządzeń (domyślne: ${name:-bez nazwy})"
                else
                    info "  $label: nie znaleziono żadnego urządzenia"
                    note_missing "$label — sprzętu nie doinstaluje żaden skrypt; podłącz go i powtórz --check-deps"
                fi
                ;;
            MICERR|OUTERR)
                name="${line#*|}"
                info "  Nie udało się odpytać urządzeń: $name"
                note_missing "detekcja audio — $name"
                ;;
            SKIP)
                info "  ${line#*|}"
                ;;
        esac
    done <<< "$output"
}

final_check() {
    info "Sprawdzam środowisko: python main.py --check-deps"
    echo
    set +e
    "$VENV_PYTHON" "$PROJECT_ROOT/main.py" --check-deps
    local status=$?
    set -e
    echo
    report_missing
    if [[ $status -eq 0 ]]; then
        info "Instalacja zakończona. Uruchomienie: $VENV_PYTHON main.py"
    else
        info "Instalacja zakończona, ale czegoś brakuje — patrz raport powyżej."
        info "Najczęściej wystarczy uruchomić `ollama serve` i powtórzyć: $VENV_PYTHON main.py --check-deps"
    fi
    return $status
}

run_installer() {
    parse_arguments "$@"
    if [[ "$FULL" == "1" ]]; then
        info "Instalacja PEŁNA asystenta w $PROJECT_ROOT (pakiety opcjonalne + modele)"
    else
        info "Instalacja asystenta w $PROJECT_ROOT"
    fi

    # Kroki systemowe: każdy sam pilnuje swoich błędów i wraca z zerem.
    install_system_packages
    install_gpu_packages
    install_aur_packages || true

    # Środowisko Pythona jest jedynym krokiem NAPRAWDĘ blokującym: bez
    # interpretera w venv nie da się ani zainstalować pakietów, ani policzyć
    # raportu. Mimo to kończymy podsumowaniem, a nie przerwaniem w pół zdania —
    # użytkownik ma wiedzieć, co się wydarzyło i co zrobić dalej.
    if ! setup_venv; then
        note_missing "środowisko wirtualne $VENV_DIR — bez niego nie ruszy nic dalej"
        echo
        report_missing
        echo
        fail "Instalacja przerwana na etapie środowiska Pythona."
        info "Napraw powyższe i uruchom ponownie: ./scripts/$INSTALLER_NAME"
        return 1
    fi

    # Od tego miejsca wszystko jest opcjonalne: pojedyncza awaria trafia do
    # podsumowania, a skrypt idzie dalej i ZAWSZE kończy raportem --check-deps.
    install_python_packages || true
    setup_env_file || note_missing "plik .env — skopiuj ręcznie: cp .env.example .env"
    install_ollama || true
    pull_language_model || true
    fetch_models || true
    check_audio_devices || true
    final_check
}
