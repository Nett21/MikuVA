#!/usr/bin/env bash
# Osobne środowisko dla konwersji głosu przez Applio (Faza 15, backend „applio").
#
#   ./scripts/install-applio.sh                  # znajdź Pythona i zainstaluj
#   ./scripts/install-applio.sh --python /ścieżka/do/python3.12
#   ./scripts/install-applio.sh --force          # odtwórz środowisko od zera
#   ./scripts/install-applio.sh --full           # wszystkie zależności Applio
#
# DLACZEGO Applio dostaje własne środowisko, skoro RVC ma już swoje:
#
# 1. To dwa różne światy zależności. `rvc-python` stoi na `fairseq`, który
#    NIE DZIAŁA powyżej Pythona 3.10. Applio fairseq się pozbyło — embedder
#    idzie przez `transformers` — ale w zamian przypina `torch==2.11.0`,
#    `numpy==2.4.6` i `transformers==5.13.1`. Wsadzenie obu zestawów do
#    jednego venva kończy się tym, że jeden z nich przestaje działać.
#
# 2. Applio przy imporcie robi `now_dir = os.getcwd()` i po TYM katalogu
#    szuka swoich embedderów i predyktorów (`rvc/models/...`). Musi więc być
#    uruchamiane z własnego katalogu — czyli w osobnym procesie, nie w
#    procesie asystenta, który ma swój własny katalog roboczy.
#
# Domyślnie instalujemy tylko to, czego wymaga ŚCIEŻKA INFERENCJI. Pełne
# `requirements.txt` Applio ciągnie dodatkowo gradio, tensorboard i matplotlib
# — kilkaset MB na interfejs webowy i wykresy treningu, których asystent nie
# uruchamia ani razu. Gdyby okazało się, że czegoś brakuje, `--full` wgrywa
# komplet bez zgadywania.
#
# Skrypt NIE UŻYWA `sudo` i niczego nie instaluje w systemie — inaczej niż
# `run-install.sh` z samego Applio, które dociąga build-essential i ffmpeg.
# Jeśli czegoś systemowego brakuje, powie o tym i przerwie.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv-applio"
APPLIO_DIR="$PROJECT_DIR/third_party/Applio"
APPLIO_REPO="https://github.com/IAHispano/Applio"
# Wagi inferencji. Pretrainedy z tej samej półki są WYŁĄCZNIE do treningu
# i świadomie ich nie pobieramy — to kilka GB, których asystent nie tknie.
WAGI_BASE="https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

PYTHON_WSKAZANY=""
FORCE=0
PELNE=0
BLEDY=()
UWAGI=()

while [ $# -gt 0 ]; do
    case "$1" in
        --python)     PYTHON_WSKAZANY="${2:-}"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        --full)       PELNE=1; shift ;;
        --applio-dir) APPLIO_DIR="${2:-}"; shift 2 ;;
        -h|--help)    sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[BŁĄD] Nieznany argument: $1"; exit 2 ;;
    esac
done

info()  { echo "[SYSTEM] $*"; }
ok()    { echo "[OK]     $*"; }
blad()  { echo "[BŁĄD]   $*"; BLEDY+=("$*"); }
uwaga() { echo "[UWAGA]  $*"; UWAGI+=("$*"); }

# --- 1. Interpreter ---------------------------------------------------------- #
#
# Applio wymaga Pythona 3.12 lub nowszego — i nie jest to jego kaprys, tylko
# konsekwencja pinów z `requirements.txt`: `scipy==1.18.0` ma `Requires-Python
# >=3.12`, więc na 3.11 instalacja kończy się „No matching distribution found".
# Sprawdzone, nie domniemane. Wersję ustalamy URUCHAMIAJĄC kandydata — bo
# `python3.12` w PATH bywa dowiązaniem do czegoś innego, a shim `mise` bez
# ustawionej wersji globalnej w ogóle nie startuje.

wersja_pythona() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

wersja_obslugiwana() {
    case "$1" in
        3.12|3.13|3.14) return 0 ;;
        *) return 1 ;;
    esac
}

znajdz_pythona() {
    if [ -n "$PYTHON_WSKAZANY" ]; then
        echo "$PYTHON_WSKAZANY"
        return
    fi
    # Kolejność od najnowszej obsługiwanej: im nowszy, tym dłużej pożyje.
    for kandydat in python3.12 python3.13 python3.14; do
        sciezka="$(command -v "$kandydat" 2>/dev/null)"
        [ -n "$sciezka" ] && wersja_obslugiwana "$(wersja_pythona "$sciezka")" && { echo "$sciezka"; return; }
    done
    # `mise` trzyma interpretery poza PATH, a jego shimy bez ustawionej wersji
    # globalnej nie działają. Sięgamy więc po prawdziwe binarki.
    for katalog in "$HOME/.local/share/mise/installs/python"/*/; do
        kandydat="$katalog/bin/python"
        [ -x "$kandydat" ] && wersja_obslugiwana "$(wersja_pythona "$kandydat")" && { echo "$kandydat"; return; }
    done
    echo ""
}

PYTHON="$(znajdz_pythona)"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    blad "Nie znalazłem Pythona w wersji 3.12 lub nowszej"
    echo "Zainstaluj któryś:   mise install python@3.12"
    echo "Albo wskaż własny:   ./scripts/install-applio.sh --python /ścieżka/do/python3.12"
    exit 1
fi

WERSJA="$(wersja_pythona "$PYTHON")"
if ! wersja_obslugiwana "$WERSJA"; then
    blad "$PYTHON to Python $WERSJA — Applio potrzebuje 3.12+ (wymusza to scipy z jego requirements.txt)"
    exit 1
fi
ok "Interpreter: $PYTHON (Python $WERSJA)"

# --- 2. Rzeczy systemowe ----------------------------------------------------- #
#
# Sprawdzamy, ale NIE instalujemy: dokładanie pakietów do systemu to nie jest
# decyzja skryptu instalującego środowisko wirtualne.

if ! command -v git >/dev/null 2>&1; then
    blad "Brak gita — bez niego nie pobiorę Applio"
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    uwaga "Brak ffmpeg. Applio używa go do formatów innych niż WAV; asystent"
    uwaga "  woła Applio wyłącznie na plikach WAV, więc to prawdopodobnie nie zaboli."
fi

# --- 3. Kod Applio ----------------------------------------------------------- #

if [ -d "$APPLIO_DIR/.git" ]; then
    ok "Applio już jest: $APPLIO_DIR"
else
    info "Pobieram Applio do $APPLIO_DIR"
    mkdir -p "$(dirname "$APPLIO_DIR")"
    # Płytki klon: historia Applio nie jest nam do niczego potrzebna.
    if ! git clone --depth 1 "$APPLIO_REPO" "$APPLIO_DIR"; then
        blad "Nie udało się sklonować $APPLIO_REPO"
        exit 1
    fi
    ok "Applio pobrane"
fi

# --- 4. Środowisko ----------------------------------------------------------- #

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
        exit 1
    fi
fi

VENV_PY="$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$VENV_DIR/Scripts/python.exe"
if [ ! -x "$VENV_PY" ]; then
    blad "Brak interpretera w $VENV_DIR — dalsze kroki nie mają sensu"
    exit 1
fi

info "Aktualizuję pip"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel || uwaga "Nie udało się odświeżyć pipa — próbuję dalej"

# --- 5. Zależności ----------------------------------------------------------- #
#
# Wersje są PRZEPISANE z requirements.txt Applio, a nie wymyślone. Gdy Applio
# je podniesie, ten plik jest jedynym miejscem do poprawienia.

INDEKS_TORCHA="https://download.pytorch.org/whl/cu128"

if [ "$PELNE" -eq 1 ]; then
    info "Instaluję PEŁNE zależności Applio (z gradio i tensorboardem)"
    if ! "$VENV_PY" -m pip install -r "$APPLIO_DIR/requirements.txt" \
            --extra-index-url "$INDEKS_TORCHA"; then
        blad "Instalacja pełnych zależności nie powiodła się"
        exit 1
    fi
else
    info "Instaluję zależności ścieżki inferencji (bez gradio, tensorboarda i matplotliba)"
    # Lista wyprowadzona z importów: rvc/infer/infer.py, rvc/infer/pipeline.py,
    # rvc/lib/utils.py, rvc/lib/predictors/f0.py. Nie zgadywana.
    if ! "$VENV_PY" -m pip install \
            "torch==2.11.0" "torchaudio==2.11.0" \
            --extra-index-url "$INDEKS_TORCHA"; then
        blad "Instalacja torcha nie powiodła się"
        exit 1
    fi
    if ! "$VENV_PY" -m pip install \
            "numpy==2.4.6" "scipy==1.18.0" "librosa==0.11.0" "soundfile==0.14.0" \
            "soxr==1.1.0" "noisereduce==3.0.3" "pedalboard==0.9.24" \
            "stftpitchshift==2.0" "faiss-cpu==1.14.3" "torchcrepe==0.0.24" \
            "torchfcpe==0.0.4" "einops==0.8.2" "transformers==5.13.1" \
            "requests==2.34.2" "tqdm==4.68.4" "wget==3.2"; then
        blad "Instalacja zależności inferencji nie powiodła się"
        echo "Spróbuj kompletu:  ./scripts/install-applio.sh --full"
        exit 1
    fi
fi
ok "Zależności zainstalowane"

# --- 6. Wagi ----------------------------------------------------------------- #
#
# Embedder (contentvec) zamienia dźwięk na cechy mowy, predyktor (rmvpe) czyta
# wysokość tonu. Bez nich Applio wywali się dopiero przy pierwszej konwersji.

pobierz() {
    local url="$1" cel="$2"
    if [ -s "$cel" ]; then
        ok "Jest już: ${cel#$APPLIO_DIR/}"
        return 0
    fi
    mkdir -p "$(dirname "$cel")"
    info "Pobieram ${cel#$APPLIO_DIR/}"
    # `-f` żeby błąd HTTP nie został zapisany jako plik, `-L` bo HuggingFace
    # przekierowuje na CDN.
    if curl -fL --retry 3 --progress-bar -o "$cel.czesciowy" "$url"; then
        mv "$cel.czesciowy" "$cel"
        return 0
    fi
    rm -f "$cel.czesciowy"
    blad "Nie udało się pobrać $url"
    return 1
}

pobierz "$WAGI_BASE/predictors/rmvpe.pt" \
        "$APPLIO_DIR/rvc/models/predictors/rmvpe.pt" || true
# Drugi predyktor. Nie jest domyślny, ale bez niego RVC_F0_METHOD=fcpe kończy
# się wyjątkiem o brakującym pliku dopiero przy pierwszej konwersji.
pobierz "$WAGI_BASE/predictors/fcpe.pt" \
        "$APPLIO_DIR/rvc/models/predictors/fcpe.pt" || true
pobierz "$WAGI_BASE/embedders/contentvec/pytorch_model.bin" \
        "$APPLIO_DIR/rvc/models/embedders/contentvec/pytorch_model.bin" || true
pobierz "$WAGI_BASE/embedders/contentvec/config.json" \
        "$APPLIO_DIR/rvc/models/embedders/contentvec/config.json" || true

# --- 7. Sprawdzenie ---------------------------------------------------------- #
#
# Import z KATALOGU APPLIO, bo tamtejszy `now_dir = os.getcwd()` decyduje,
# gdzie biblioteka będzie szukać wag. Uruchomienie stąd „działa" tylko pozornie
# — do pierwszej konwersji.

info "Sprawdzam, czy Applio da się zaimportować"
WYNIK="$(cd "$APPLIO_DIR" && "$VENV_PY" - <<'PYCHECK' 2>&1 | tail -1
try:
    from rvc.infer.infer import VoiceConverter
    import torch
    print(f"OK cuda={torch.cuda.is_available()}")
except Exception as exc:
    print(f"BLAD {type(exc).__name__}: {exc}")
PYCHECK
)"

case "$WYNIK" in
    OK*)
        ok "Applio działa ($WYNIK)"
        case "$WYNIK" in
            *cuda=False*) uwaga "Torch nie widzi karty — konwersja pójdzie na CPU i będzie wolna" ;;
        esac
        ;;
    *)
        blad "Applio nie daje się zaimportować: $WYNIK"
        echo "Spróbuj kompletu zależności:  ./scripts/install-applio.sh --full"
        ;;
esac

# --- 8. Podsumowanie --------------------------------------------------------- #

echo
if [ ${#UWAGI[@]} -gt 0 ]; then
    echo "Uwagi:"
    for u in "${UWAGI[@]}"; do echo "  - $u"; done
fi
if [ ${#BLEDY[@]} -gt 0 ]; then
    echo "Błędy:"
    for b in "${BLEDY[@]}"; do echo "  - $b"; done
    exit 1
fi

echo "Środowisko Applio gotowe: $VENV_DIR"
echo "Kod Applio:               $APPLIO_DIR"
echo
echo "Włącz je w .env:"
echo "  RVC_BACKEND=applio"
echo "  RVC_APPLIO_PATH=$APPLIO_DIR"
echo "  RVC_APPLIO_PYTHON=$VENV_PY"
