#!/usr/bin/env bash
# Jedno wejście instalacyjne dla wszystkich systemów uniksowych.
#
#   ./scripts/install.sh --full          # wszystko, z pytaniami
#   ./scripts/install.sh --full --yes    # wszystko, bez pytań
#   ./scripts/install.sh                 # sam rdzeń (jak dotąd)
#
# Ten plik NIE instaluje niczego sam. Rozpoznaje system i menedżer pakietów, po
# czym oddaje robotę właściwej nakładce — logika instalacji zostaje w jednym
# miejscu (install-common.sh), a użytkownik ma jedną nazwę do zapamiętania
# zamiast pięciu. Wszystkie argumenty lecą dalej bez zmian.
#
# Menedżer wykrywamy po obecności programu w PATH, a nie po nazwie dystrybucji:
# pochodne (Manjaro, Mint, Omarchy, Pop!_OS) mają własne nazwy, ale ten sam
# menedżer, a lista nazw dystrybucji nigdy nie jest kompletna.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_installer() {
    case "$(uname -s)" in
        Darwin)
            echo "install-macos.sh"
            ;;
        Linux|GNU/kFreeBSD|FreeBSD|OpenBSD|NetBSD)
            # Kolejność ma znaczenie tylko tam, gdzie ktoś ma dwa menedżery
            # naraz (np. apt obok pacmana w kontenerze) — wtedy wygrywa ten
            # natywny dla systemu plików pakietów, czyli sprawdzany pierwszy.
            if command -v pacman >/dev/null 2>&1; then
                echo "install-pacman.sh"
            elif command -v apt-get >/dev/null 2>&1; then
                echo "install-apt.sh"
            else
                # dnf, zypper, apk albo brak menedżera — obsługuje je jeden plik.
                echo "install-linux-generic.sh"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Git Bash na Windowsie: tu nie ma czym instalować pakietów systemu.
            echo "[ERROR] To jest Windows — użyj PowerShella:" >&2
            echo "[ERROR]   .\\scripts\\install.ps1 -Full" >&2
            exit 2
            ;;
        *)
            # Nieznany Unix: środowisko Pythona da się zbudować wszędzie, a krok
            # systemowy sam się pominie, gdy nie znajdzie menedżera pakietów.
            echo "install-linux-generic.sh"
            ;;
    esac
}

installer="$(pick_installer)"
echo "[SYSTEM] System: $(uname -s) — uruchamiam scripts/$installer"
exec "$SCRIPTS_DIR/$installer" "$@"
