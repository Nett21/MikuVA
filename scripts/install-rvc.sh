#!/usr/bin/env bash
# Osobne środowisko dla konwersji głosu RVC (Faza 15).
#
#   ./scripts/install-rvc.sh                 # znajdź Pythona 3.10 i zainstaluj
#   ./scripts/install-rvc.sh --python /ścieżka/do/python3.10
#   ./scripts/install-rvc.sh --force         # odtwórz środowisko od zera
#
# DLACZEGO to jest osobny skrypt i osobne środowisko, a nie linijka w
# requirements.txt — trzy powody, każdy sprawdzony, nie domniemany:
#
# 1. `fairseq==0.12.2` (wymagany przez rvc-python) NIE DZIAŁA na Pythonie 3.11
#    i nowszym. Moduł dataclasses zaostrzył w 3.11 reguły dla domyślnych
#    wartości mutowalnych i import fairseq kończy się:
#      ValueError: mutable default <class 'fairseq.dataclass.configs.CommonConfig'>
#    Asystent wymaga Pythona 3.12+, więc te dwa światy nie zmieszczą się
#    w jednym środowisku. Stąd drugi venv i proces-pracownik.
#
# 2. `omegaconf==2.0.6` ma NIEPOPRAWNE metadane (`PyYAML (>=5.1.*)`), które
#    pip 24.1 i nowszy odrzuca. Dlatego w tym środowisku pip jest cofnięty.
#
# 3. `pyworld` importuje `pkg_resources`, którego nie ma w setuptools 81+.
#    Stąd `setuptools<81`.
#
# Żadna z tych rzeczy nie jest naszym wyborem — to stan bibliotek RVC w chwili
# pisania. Gdy się zmieni, ten plik jest jedynym miejscem do poprawienia.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-rvc"
WYMAGANA_WERSJA="3.10"

PYTHON_WSKAZANY=""
FORCE=0
BLEDY=()
UWAGI=()

while [ $# -gt 0 ]; do
    case "$1" in
        --python) PYTHON_WSKAZANY="${2:-}"; shift 2 ;;
        --force)  FORCE=1; shift ;;
        -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[BŁĄD] Nieznany argument: $1"; exit 2 ;;
    esac
done

info()  { echo "[SYSTEM] $*"; }
ok()    { echo "[OK]     $*"; }
blad()  { echo "[BŁĄD]   $*"; BLEDY+=("$*"); }
uwaga() { echo "[UWAGA]  $*"; UWAGI+=("$*"); }

# --- 1. Interpreter ---------------------------------------------------------- #
#
# Wersję sprawdzamy URUCHAMIAJĄC kandydata, a nie po nazwie pliku: `python3.10`
# w PATH bywa dowiązaniem do czegoś innego, a `python3` na jednej maszynie jest
# 3.10, a na drugiej 3.14.

wersja_pythona() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

# UWAGA na treść tej funkcji: jej wynik jest przechwytywany przez $( ), więc
# na standardowe wyjście wolno jej wypisać WYŁĄCZNIE ścieżkę. Każde `echo`
# z komunikatem trafiłoby do zmiennej i zostało wzięte za ścieżkę do
# interpretera — dlatego o błędach melduje wołający, nie ta funkcja.
znajdz_pythona() {
    local kandydat wersja

    for kandydat in python3.10 python3 python; do
        command -v "$kandydat" >/dev/null 2>&1 || continue
        wersja="$(wersja_pythona "$kandydat")"
        if [ "$wersja" = "$WYMAGANA_WERSJA" ]; then command -v "$kandydat"; return 0; fi
    done

    # Menedżery wersji — szukamy tylko wtedy, gdy są zainstalowane. Nie
    # instalujemy ich sami i nie zakładamy, że ktoś ich używa.
    local baza
    if command -v mise >/dev/null 2>&1; then
        baza="$(mise where "python@$WYMAGANA_WERSJA" 2>/dev/null)"
        if [ -n "$baza" ] && [ -x "$baza/bin/python$WYMAGANA_WERSJA" ]; then
            echo "$baza/bin/python$WYMAGANA_WERSJA"; return 0
        fi
    fi
    if command -v pyenv >/dev/null 2>&1; then
        baza="$(pyenv root 2>/dev/null)/versions"
        if [ -d "$baza" ]; then
            kandydat="$(find "$baza" -maxdepth 1 -name "$WYMAGANA_WERSJA.*" -type d 2>/dev/null | sort -V | tail -1)"
            if [ -n "$kandydat" ] && [ -x "$kandydat/bin/python" ]; then
                echo "$kandydat/bin/python"; return 0
            fi
        fi
    fi
    return 1
}

info "Szukam Pythona $WYMAGANA_WERSJA (RVC nie działa na nowszym — patrz nagłówek pliku)"

PYTHON=""
if [ -n "$PYTHON_WSKAZANY" ]; then
    # Wskazany wprost sprawdzamy tutaj, a nie w funkcji, żeby móc powiedzieć
    # WPROST, co jest nie tak — funkcja nie może nic wypisać (patrz wyżej).
    WERSJA_WSKAZANEGO="$(wersja_pythona "$PYTHON_WSKAZANY" || true)"
    if [ "$WERSJA_WSKAZANEGO" = "$WYMAGANA_WERSJA" ]; then
        PYTHON="$PYTHON_WSKAZANY"
    else
        blad "Wskazany interpreter to Python ${WERSJA_WSKAZANEGO:-nieznany}, a potrzebny jest $WYMAGANA_WERSJA"
    fi
else
    PYTHON="$(znajdz_pythona || true)"
fi

if [ -z "$PYTHON" ]; then
    blad "Nie znalazłem Pythona $WYMAGANA_WERSJA"
    echo
    echo "Zainstaluj go jednym z poniższych, a potem uruchom ten skrypt ponownie:"
    if command -v mise >/dev/null 2>&1; then
        echo "    mise install python@$WYMAGANA_WERSJA   # mise jest już na tej maszynie"
    else
        echo "    mise install python@$WYMAGANA_WERSJA   # https://mise.jdx.dev"
    fi
    echo "    pyenv install $WYMAGANA_WERSJA             # https://github.com/pyenv/pyenv"
    echo "    pacman -S python310                   # jako root, jeśli jest w repozytoriach"
    echo
    echo "Albo wskaż istniejący:  ./scripts/install-rvc.sh --python /ścieżka/do/python3.10"
    echo
    echo "[PODSUMOWANIE] Środowisko RVC NIE powstało. Asystent będzie mówił zwykłym"
    echo "               głosem Pipera — to działa, tylko nie jest to głos Miku."
    exit 1
fi
ok "Interpreter: $PYTHON ($("$PYTHON" -V 2>&1))"

# --- 2. Środowisko ----------------------------------------------------------- #

if [ -d "$VENV_DIR" ] && [ "$FORCE" -eq 1 ]; then
    info "Usuwam poprzednie środowisko (--force)"
    rm -rf "$VENV_DIR"
fi

if [ -d "$VENV_DIR" ]; then
    ok "Środowisko już istnieje: $VENV_DIR (--force odtworzy je od zera)"
else
    info "Tworzę środowisko: $VENV_DIR"
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        blad "Nie udało się utworzyć środowiska (brakuje pakietu python3-venv?)"
    fi
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$VENV_DIR/Scripts/python.exe"

if [ ! -x "$VENV_PY" ]; then
    blad "Brak interpretera w $VENV_DIR — dalsze kroki nie mają sensu"
else
    # --- 3. Narzędzia w cofniętych wersjach (powody w nagłówku pliku) -------- #
    info "Ustawiam pip<24.1 i setuptools<81 (wymuszone przez zależności RVC)"
    "$VENV_PY" -m pip install --quiet --upgrade "pip<24.1" "setuptools<81" wheel \
        || blad "Nie udało się ustawić pipa/setuptools"

    # --- 4. RVC -------------------------------------------------------------- #
    info "Instaluję rvc-python (to potrwa: schodzi torch, kilka GB)"
    if ! "$VENV_PY" -m pip install rvc-python; then
        blad "Instalacja rvc-python nie powiodła się"
    fi
fi

# --- 5. Sprawdzenie ---------------------------------------------------------- #
#
# Instalacja bez błędu nie znaczy, że to się importuje — sprawdzamy realnie.

if [ -x "$VENV_PY" ]; then
    info "Sprawdzam, czy to się w ogóle uruchamia"
    WYNIK="$("$VENV_PY" - <<'PYCHECK' 2>&1 | tail -1
try:
    import torch
    from rvc_python.infer import RVCInference  # noqa: F401
except Exception as exc:
    print(f"BLAD|{type(exc).__name__}: {exc}")
else:
    if torch.cuda.is_available():
        print(f"OK|GPU|{torch.cuda.get_device_name(0)}")
    else:
        print("OK|CPU|")
PYCHECK
)"
    case "$WYNIK" in
        OK\|GPU\|*) ok "RVC działa, GPU: ${WYNIK#OK|GPU|}" ;;
        OK\|CPU\|*) ok "RVC działa, ale na CPU"
                    uwaga "Brak CUDA — konwersja zadziała, tylko wolniej. Asystent ostrzeże o tym w logu." ;;
        BLAD\|*)    blad "RVC się nie uruchamia: ${WYNIK#BLAD|}" ;;
        *)          blad "Nieoczekiwany wynik sprawdzenia: $WYNIK" ;;
    esac
fi

# --- 6. Podsumowanie --------------------------------------------------------- #

echo
echo "══ PODSUMOWANIE ═══════════════════════════════════════════════════════"
if [ ${#BLEDY[@]} -gt 0 ]; then
    echo "Nie udało się (${#BLEDY[@]}):"
    for item in "${BLEDY[@]}"; do echo "  ✗ $item"; done
    echo
    echo "Asystent będzie mówił zwykłym głosem Pipera — to działa, tylko nie jest"
    echo "to głos Miku. Szczegóły błędów są wyżej, nic nie zostało ukryte."
    exit 1
fi

for item in "${UWAGI[@]}"; do echo "  ! $item"; done
echo "Środowisko RVC gotowe: $VENV_DIR"

# Katalog na model tworzymy TERAZ, żeby następny krok był oczywisty:
# models/ jest w .gitignore, więc w świeżo sklonowanym repozytorium go nie ma.
mkdir -p "$PROJECT_DIR/models/rvc"
echo
echo "Zostały DWIE rzeczy, których ten skrypt nie zrobi za ciebie:"
echo
echo "  1. Wrzuć własny model do models/rvc/ — plik .pth i opcjonalnie .index."
echo "     Repozytorium nie zawiera żadnego modelu i nie może zawierać."
echo
echo "  2. Włącz go w config/user_settings.json:"
echo '       "voice_engine": "rvc_miku",'
echo '       "rvc": { "enabled": true, "model_path": "models/rvc/twoj-model.pth",'
echo '                "index_path": "models/rvc/twoj-model.index",'
echo '                "pitch_shift": 12, "index_rate": 0.75 }'
echo
echo "Potem sprawdź:  python main.py --check-deps   i   python main.py --voice-test"
