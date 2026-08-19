# Instalacja asystenta na Windowsie 10/11.
#
#   .\scripts\install-windows.ps1              # z pytaniami
#   .\scripts\install-windows.ps1 -Yes         # bez pytań
#   .\scripts\install-windows.ps1 -Dev         # razem z pakietami do testów
#   .\scripts\install-windows.ps1 -Full        # wszystko: opcje, modele, testy
#   .\scripts\install-windows.ps1 -Offline     # instalacja z vendor\wheels
#
# Nic nie jest instalowane bez pytania, a każde polecenie jest wypisywane przed
# wykonaniem. Skrypt nie pobiera ani nie uruchamia żadnych instalatorów spoza
# winget — brakujące programy wskazuje linkiem do strony producenta.
#
# Plik jest zapisany w UTF-8 Z ZNACZNIKIEM BOM — bez niego Windows PowerShell 5.1
# czyta go jako ANSI i polskie znaki w komunikatach zamieniają się w krzaki.
#
# Jeśli PowerShell odmówi uruchomienia (polityka wykonywania), użyj:
#   powershell -ExecutionPolicy Bypass -File .\scripts\install-windows.ps1

[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$Full,
    [switch]$Dev,
    [switch]$NoSystem,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"

# Pełna instalacja obejmuje narzędzia deweloperskie — „wszystko" ma znaczyć
# wszystko, bez drugiej flagi do zapamiętania.
if ($Full) { $Dev = $true }

# Czego nie udało się zainstalować — wypisywane na końcu, z poleceniem naprawy.
$script:MissingReport = New-Object System.Collections.Generic.List[string]

$TagSystem = "[SYSTEM]"
$TagError = "[ERROR]"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = if ($env:MIKU_VENV_DIR) { $env:MIKU_VENV_DIR } else { Join-Path $ProjectRoot ".venv" }
$Wheelhouse = if ($env:MIKU_WHEELHOUSE_DIR) { $env:MIKU_WHEELHOUSE_DIR } else { Join-Path $ProjectRoot "vendor\wheels" }
$RequiredPythonMinor = 12

function Write-Info { param([string]$Message) Write-Host "$TagSystem $Message" }
function Write-Failure { param([string]$Message) Write-Host "$TagError $Message" -ForegroundColor Red }

function Confirm-Step {
    param([string]$Question)
    if ($Yes) {
        Write-Info "$Question - tak (-Yes)"
        return $true
    }
    $answer = Read-Host "$TagSystem $Question [t/N]"
    return $answer -match '^(t|tak|y|yes)$'
}

# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

function Find-Python {
    # Najpierw launcher `py` (standard na Windowsie), potem python z PATH.
    # Wersja musi być >= 3.12 — starsza nie uruchomi projektu.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += ,@("py", @("-3.14")), ,@("py", @("-3.13")), ,@("py", @("-3.12")), ,@("py", @("-3"))
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += ,@("python", @())
    }

    foreach ($candidate in $candidates) {
        $exe = $candidate[0]
        $prefix = $candidate[1]
        $arguments = @($prefix) + @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, $RequiredPythonMinor) else 1)")
        try {
            & $exe @arguments 2>$null
            if ($LASTEXITCODE -eq 0) {
                return ,@($exe, $prefix)
            }
        } catch {
            continue
        }
    }
    return $null
}

function Install-SystemPackages {
    if ($NoSystem) {
        Write-Info "Pomijam pakiety systemowe (-NoSystem)."
        return
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Info "Nie znaleziono winget — pomijam krok systemowy."
        Write-Info "Zainstaluj ręcznie: Python >= 3.$RequiredPythonMinor (https://python.org) oraz Ollamę (https://ollama.com/download)."
        return
    }

    if (-not (Find-Python)) {
        Write-Info "Nie znaleziono Pythona >= 3.$RequiredPythonMinor."
        if (Confirm-Step "Zainstalować Pythona przez winget?") {
            Write-Info "Wykonuję: winget install --id Python.Python.3.12 --source winget"
            winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
            Write-Info "Jeśli kolejny krok nie widzi Pythona, otwórz nowe okno terminala (PATH odświeża się po instalacji)."
        }
    } else {
        Write-Info "Python >= 3.$RequiredPythonMinor już jest — nie ruszam go."
    }

    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Info "Model językowy uruchamia Ollama."
        if (Confirm-Step "Zainstalować Ollamę przez winget?") {
            Write-Info "Wykonuję: winget install --id Ollama.Ollama --source winget"
            winget install --id Ollama.Ollama --source winget --accept-package-agreements --accept-source-agreements
        } else {
            Write-Info "Bez Ollamy asystent uruchomi się, ale rozmowa nie ruszy: https://ollama.com/download"
        }
    } else {
        Write-Info "Ollama jest już zainstalowana."
    }
}

# --------------------------------------------------------------------------- #
# Środowisko Pythona
# --------------------------------------------------------------------------- #

function Show-PythonInstructions {
    # Świadomie NIE instalujemy Pythona sami, gdy użytkownik się nie zgodził
    # albo gdy nie ma winget. Instalacja interpretera zmienia PATH całego konta —
    # to decyzja użytkownika, nie skryptu. Zostaje jasna instrukcja z linkiem.
    Write-Failure "Nie znalazłem Pythona >= 3.$RequiredPythonMinor. Bez niego nie da się nic zainstalować."
    Write-Info ""
    Write-Info "Zainstaluj go jednym z dwóch sposobów:"
    Write-Info "  1) https://www.python.org/downloads/   <- pobierz instalator"
    Write-Info "     W instalatorze ZAZNACZ:"
    Write-Info "       [x] Add python.exe to PATH      (bez tego skrypt go nie zobaczy)"
    Write-Info "       [x] tcl/tk and IDLE             (bez tego nie bedzie okna graficznego)"
    Write-Info "  2) winget install --id Python.Python.3.12 --source winget"
    Write-Info ""
    Write-Info "Potem OTWORZ NOWE OKNO terminala (PATH nie odswieza sie w juz otwartym)"
    Write-Info "i uruchom ponownie: .\scripts\install-windows.ps1"
}

function Initialize-Venv {
    $python = Find-Python
    if (-not $python) {
        Show-PythonInstructions
        exit 1
    }
    $exe = $python[0]
    $prefix = $python[1]

    if (-not (Test-Path $VenvDir)) {
        Write-Info "Tworzę środowisko wirtualne: $VenvDir"
        # PowerShell NIE przerywa pracy, gdy program zewnętrzny zwróci błąd
        # ($ErrorActionPreference dotyczy poleceń PowerShella, nie natywnych),
        # więc kod wyjścia trzeba sprawdzić samemu. Bez tego nieudany `venv`
        # przechodziłby po cichu, a błąd wychodziłby dopiero przy pip.
        & $exe @($prefix) -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Nie udało się utworzyć środowiska w $VenvDir (kod $LASTEXITCODE)."
            Write-Info "Częste przyczyny: brak miejsca na dysku, brak prawa zapisu do katalogu projektu,"
            Write-Info "albo Python zainstalowany ze Sklepu Microsoft (ma ograniczony dostęp do dysku)."
            Write-Info "Sprawdź: $exe $prefix -m venv --help"
            exit 1
        }
    } else {
        Write-Info "Środowisko wirtualne już istnieje: $VenvDir"
    }

    $script:VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $script:VenvPython)) {
        Write-Failure "Środowisko $VenvDir jest uszkodzone (brak $script:VenvPython)."
        Write-Info "Usuń ten katalog i uruchom skrypt ponownie:"
        Write-Info "  Remove-Item -Recurse -Force '$VenvDir'"
        exit 1
    }
}

function Test-Network {
    try {
        & $script:VenvPython -c "import socket; socket.setdefaulttimeout(3); socket.create_connection(('pypi.org', 443)).close()" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-Wheelhouse {
    return (Test-Path $Wheelhouse) -and (Get-ChildItem -Path $Wheelhouse -File -ErrorAction SilentlyContinue)
}

function Install-PythonPackages {
    $pipArguments = @("-m", "pip", "install")
    if ($Offline -or ((Test-Wheelhouse) -and -not (Test-Network))) {
        if (-not (Test-Wheelhouse)) {
            Write-Failure "Tryb offline, a magazyn kół $Wheelhouse jest pusty."
            Write-Info "Na maszynie z internetem: python scripts\prepare_offline.py --wheels"
            $script:MissingReport.Add("pakiety Pythona  ->  pusty magazyn kol $Wheelhouse")
            return
        }
        Write-Info "Instaluję pakiety z $Wheelhouse (bez sieci)."
        $pipArguments += @("--no-index", "--find-links", $Wheelhouse)
    } else {
        Write-Info "Instaluję pakiety z PyPI."
        & $script:VenvPython -m pip install --upgrade pip | Out-Null
    }

    # Kod wyjścia programu zewnętrznego trzeba sprawdzić WPROST — inaczej
    # nieudana instalacja przechodzi po cichu i skrypt melduje sukces, mimo że
    # nic się nie zainstalowało. Awaria nie przerywa pracy: raport na końcu
    # (i --check-deps) mają pokazać, czego naprawdę brakuje.
    & $script:VenvPython @pipArguments -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Instalacja pakietów z requirements.txt nie powiodła się (kod $LASTEXITCODE)."
        $script:MissingReport.Add("pakiety z requirements.txt  ->  powtorz: .venv\Scripts\python.exe -m pip install -r requirements.txt")
        Write-Info "Idę dalej — raport na końcu pokaże, czego brakuje."
    }
    if ($Dev) {
        & $script:VenvPython @pipArguments -r (Join-Path $ProjectRoot "requirements-dev.txt")
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Instalacja pakietów deweloperskich nie powiodła się."
            $script:MissingReport.Add("pakiety z requirements-dev.txt (pytest, ruff, mypy)")
        }
    }

    Install-OptionalPackages
}

# --------------------------------------------------------------------------- #
# Pakiety opcjonalne
# --------------------------------------------------------------------------- #
#
# Kazdy z nich wlacza JEDNA funkcje i niczego wiecej nie psuje, gdy go zabraknie.
# Instalujemy pojedynczo i przelykamy blad: czesc nie ma kol dla kazdej wersji
# Pythona, a nieudana instalacja jednego pakietu nie moze przerwac reszty.

function Install-OptionalPackage {
    param([string]$Package, [string]$Gives, [string]$Without)
    & $script:VenvPython -m pip install $Package 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Zainstalowano $Package — $Gives."
        return $true
    }
    Write-Info "$Package niedostępny dla tej wersji Pythona — $Without."
    $script:MissingReport.Add("$Package  ->  $script:VenvPython -m pip install $Package")
    return $false
}

function Install-OptionalPackages {
    if ($Offline) {
        Write-Info "Tryb offline — pomijam pakiety opcjonalne (są w vendor\wheels albo ich nie ma)."
        return
    }

    $wanted = @(
        @("webrtcvad-wheels", "lepsza detekcja mowy (VAD)", "zadziała detektor energetyczny"),
        @("piper-tts", "synteza mowy w procesie asystenta", "odpowiedzi zostaną tekstowe")
    )
    if ($Full) {
        $wanted += @(
            @("pypdf", "czytanie PDF-ów (narzędzia pdf.read i pdf.search)", "te dwa narzędzia będą niedostępne"),
            @("psutil", "lista i zamykanie procesów", "narzędzia procesowe będą niedostępne"),
            @("openwakeword", "wykrywanie frazy budzącej modelem KWS", "zadziała detektor na Whisperze"),
            @("faiss-cpu", "szybsze wyszukiwanie w pamięci semantycznej", "policzy NumPy")
        )
    }
    foreach ($item in $wanted) {
        Install-OptionalPackage -Package $item[0] -Gives $item[1] -Without $item[2] | Out-Null
    }

    if (-not $Full) {
        return
    }

    # PyTorch to kilka gigabajtow - pytamy osobno, nawet w trybie pelnym.
    Write-Info "Pozostał sentence-transformers: lokalne embeddingi bez Ollamy, ale pobiera PyTorcha (kilka GB)."
    Write-Info "Bez niego embeddingi policzy Ollama modelem nomic-embed-text (~270 MB)."
    if (Confirm-Step "Zainstalować sentence-transformers?") {
        Install-OptionalPackage -Package "sentence-transformers" -Gives "lokalne embeddingi" -Without "embeddingi policzy Ollama" | Out-Null
    } else {
        Write-Info "Pomijam sentence-transformers — zadba o to Ollama."
    }
}

# --------------------------------------------------------------------------- #
# Modele (tryb pelny)
# --------------------------------------------------------------------------- #

function Get-Models {
    # Modele to nie pakiety - pip ich nie zainstaluje. Kazdy pobieramy osobno,
    # zeby nieudane pobranie jednego nie zabralo pozostalych.
    if (-not $Full) {
        return
    }
    if ($Offline) {
        Write-Info "Tryb offline — modele muszą już być na dysku (prepare_offline.py na maszynie z siecią)."
        return
    }

    $prepare = Join-Path $ProjectRoot "scripts\prepare_offline.py"

    Write-Info "Pobieram model rozpoznawania mowy (Whisper) — kilkaset MB."
    & $script:VenvPython $prepare --whisper
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Nie udało się pobrać modelu Whispera."
        $script:MissingReport.Add("model Whispera  ->  $script:VenvPython scripts\prepare_offline.py --whisper")
    }

    Write-Info "Pobieram głos do syntezy mowy (Piper)."
    & $script:VenvPython $prepare --piper
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Nie udało się pobrać głosu Pipera."
        $script:MissingReport.Add("głos Pipera  ->  $script:VenvPython scripts\prepare_offline.py --piper")
    }

    # Embeddingi: albo lokalnie (sentence-transformers), albo przez Ollame.
    & $script:VenvPython -c "import sentence_transformers" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Pobieram model embeddingów do pamięci semantycznej."
        & $script:VenvPython $prepare --embeddings
        if ($LASTEXITCODE -ne 0) {
            $script:MissingReport.Add("model embeddingów  ->  $script:VenvPython scripts\prepare_offline.py --embeddings")
        }
    } elseif (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Info "Pobieram model embeddingów Ollamy: nomic-embed-text"
        ollama pull nomic-embed-text
        if ($LASTEXITCODE -ne 0) {
            $script:MissingReport.Add("nomic-embed-text  ->  ollama pull nomic-embed-text")
        }
    } else {
        $script:MissingReport.Add("nomic-embed-text  ->  ollama pull nomic-embed-text (po instalacji Ollamy)")
    }
}

function Show-MissingReport {
    if ($script:MissingReport.Count -eq 0) {
        return
    }
    Write-Host ""
    Write-Info "Nie udało się (albo pominięto) — każdą z tych rzeczy można dorobić później:"
    foreach ($item in $script:MissingReport) {
        Write-Info "  - $item"
    }
}

function Initialize-EnvFile {
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        Write-Info "Plik .env już istnieje — nie ruszam go."
        return
    }
    Copy-Item (Join-Path $ProjectRoot ".env.example") $envFile
    Write-Info "Utworzono .env na podstawie .env.example"
}

function Get-LanguageModel {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        return
    }
    # Nazwa modelu bierze się z KONFIGURACJI (.env, utworzonego przed chwilą
    # z .env.example), a nie z wartości wpisanej tutaj — inaczej zmiana
    # OLLAMA_MODEL nie miałaby wpływu na to, co instalator pobiera.
    #
    # Obecność sprawdzamy przez detect_ollama (HTTP), a NIE przez parsowanie
    # `ollama list`. Dwa powody: to drugie potrafi wystartować demona, którego
    # logi lecą potem na nasze wyjście, a porównanie po fragmencie nazwy dawało
    # fałszywe trafienia — przy zainstalowanym `qwen2.5:0.5b` model
    # `qwen2.5:7b-instruct` był meldowany jako obecny i nigdy się nie pobierał.
    Push-Location $ProjectRoot
    try {
        $model = & $script:VenvPython -c "from config import get_settings; print(get_settings().ollama_model)" 2>$null
        $present = & $script:VenvPython -c "from config import detect_ollama, get_settings; print('1' if detect_ollama(get_settings()).model_present else '0')" 2>$null
    } finally {
        Pop-Location
    }
    if (-not $model) {
        return
    }
    $model = $model.Trim()
    if ($present -and $present.Trim() -eq "1") {
        Write-Info "Model językowy '$model' jest już pobrany."
        return
    }
    Write-Info "Model językowy '$model' nie jest jeszcze pobrany (kilka GB)."
    if (Confirm-Step "Pobrać go teraz poleceniem 'ollama pull $model'?") {
        ollama pull $model
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Pobieranie modelu nie powiodło się (kod $LASTEXITCODE)."
            $script:MissingReport.Add("model jezykowy $model  ->  powtorz: ollama pull $model")
        }
    } else {
        Write-Info "Pobierzesz go później: ollama pull $model"
        $script:MissingReport.Add("model jezykowy $model  ->  ollama pull $model")
    }
}

function Test-AudioDevices {
    # Mikrofon i głośnik to jedyne wymagania, których NIE da się doinstalować —
    # albo sprzęt jest, albo go nie ma. Dlatego pytamy o nie osobno i nazywamy
    # rzecz po imieniu, zamiast zostawiać użytkownika z jedną linijką w gąszczu
    # raportu zależności.
    #
    # Detekcja NIE jest tu powtarzana: wołamy dokładnie te funkcje, których
    # używa --check-deps (audio/microphone.py, audio/output.py). Druga
    # implementacja rozjechałaby się z pierwszą przy pierwszej zmianie.
    Write-Info "Sprawdzam urządzenia audio (mikrofon i głośnik)."

    $probe = @'
import sys

try:
    from config import get_settings
except Exception as exc:
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
'@

    Push-Location $ProjectRoot
    try {
        $output = $probe | & $script:VenvPython - 2>$null
    } catch {
        Write-Info "  Nie udało się odpytać urządzeń audio."
        return
    } finally {
        Pop-Location
    }

    foreach ($line in @($output)) {
        if (-not $line) { continue }
        $parts = $line -split "\|", 3
        switch ($parts[0]) {
            { $_ -in @("MIC", "OUT") } {
                $label = if ($parts[0] -eq "MIC") { "Mikrofon" } else { "Wyjście dźwięku" }
                $count = [int]$parts[1]
                if ($count -gt 0) {
                    $name = if ($parts[2]) { $parts[2] } else { "bez nazwy" }
                    Write-Info "  ${label}: $count urządzeń (domyślne: $name)"
                } else {
                    Write-Info "  ${label}: nie znaleziono żadnego urządzenia"
                    $script:MissingReport.Add("$label  ->  sprzetu nie doinstaluje zaden skrypt; podlacz go i powtorz --check-deps")
                }
            }
            { $_ -in @("MICERR", "OUTERR") } {
                Write-Info "  Nie udało się odpytać urządzeń: $($parts[1])"
                $script:MissingReport.Add("detekcja audio  ->  $($parts[1])")
            }
            "SKIP" {
                Write-Info "  $($parts[1])"
            }
        }
    }
}

function Invoke-FinalCheck {
    Write-Info "Sprawdzam środowisko: python main.py --check-deps"
    Write-Host ""
    & $script:VenvPython (Join-Path $ProjectRoot "main.py") --check-deps
    $status = $LASTEXITCODE
    Write-Host ""
    Show-MissingReport
    if ($status -eq 0) {
        Write-Info "Instalacja zakończona. Uruchomienie: $script:VenvPython main.py"
    } else {
        Write-Info "Instalacja zakończona, ale czegoś brakuje — patrz raport powyżej."
        Write-Info "Najczęściej wystarczy uruchomić Ollamę i powtórzyć: $script:VenvPython main.py --check-deps"
    }
    exit $status
}

if ($Full) {
    Write-Info "Instalacja PEŁNA asystenta w $ProjectRoot (pakiety opcjonalne + modele)"
} else {
    Write-Info "Instalacja asystenta w $ProjectRoot"
}
Install-SystemPackages
Initialize-Venv
Install-PythonPackages
Initialize-EnvFile
Get-LanguageModel
Get-Models
Test-AudioDevices
Invoke-FinalCheck
