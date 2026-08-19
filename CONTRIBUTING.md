# Jak współtworzyć

Dzięki, że chcesz coś dołożyć. Ten dokument opisuje dwie rzeczy, które robi się
tu najczęściej — **nowe narzędzie** i **nowy plugin** — oraz zasady, których
trzyma się reszta kodu.

Wszystko poniżej da się sprawdzić przed wysłaniem zmiany:

```bash
pip install -r requirements-dev.txt
pytest          # ~1300 testów, ~25 s, bez mikrofonu i GPU
ruff check .
mypy .
```

---

## Zanim zaczniesz: jak to jest poskładane

Zależności idą **w jedną stronę**:

```
config.py  ──►  audio/   host/   security/
     │              │       │        │
     └──────────►  database/ ◄───────┘
                       │
                    brain/  ◄──  tools/  ◄──  plugins/
                       │
                    gui/  main.py
```

`config.py` nie wie o niczym innym. `audio/` nie wie o `brain/`. `tools/` nie
wie o modelu językowym. Dzięki temu każdą warstwę da się przetestować atrapami —
i dlatego cały zestaw testów przechodzi na maszynie bez mikrofonu, bez GPU
i bez działającej Ollamy.

**Jeśli Twoja zmiana wymaga importu „w drugą stronę", prawie na pewno trafiła
w złą warstwę.** Napisz o tym w PR-ze zamiast obchodzić to importem lokalnym.

---

## Dodanie nowego narzędzia

Narzędzie to jedyna droga od modelu do jakiegokolwiek działania. Model **nie
wykonuje kodu** — może wyłącznie poprosić o wywołanie czegoś, co ktoś wcześniej
napisał w Pythonie, a router sprawdza to zanim cokolwiek się stanie:

```
model prosi → router: czy takie narzędzie istnieje i jest włączone?
            → pydantic: czy argumenty mają właściwe typy i zakresy?
            → polityka: jakie to ryzyko?
            → [pytanie do CZŁOWIEKA, gdy HIGH albo CRITICAL]
            → dopiero teraz kod coś robi
            → wynik wraca do modelu jako tekst
```

### 1. Model argumentów

```python
from pydantic import Field
from tools.base import ToolArgs

class PogodaArgs(ToolArgs):
    miasto: str = Field(min_length=1, max_length=100)
    dni: int = Field(default=1, ge=1, le=7)
```

Limity nie są ozdobą: argumenty przychodzą **od modelu językowego**, który
potrafi wysłać pustą nazwę, liczbę ujemną albo łańcuch na 40 kB. Walidacja jest
pierwszą linią obrony i ma być ciasna.

### 2. Ciało narzędzia

```python
from tools.base import BaseTool, ToolContext, ToolResult

class PogodaTool(BaseTool[PogodaArgs]):
    async def run(self, args: PogodaArgs, ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return ToolResult.success(
                {"podglad": f"sprawdziłbym pogodę dla {args.miasto}"},
                display=f"(próbnie) pogoda dla {args.miasto}",
            )
        ...
        return ToolResult.success({"temperatura": 12}, display="12 °C, pochmurno")
```

* `run` jest **asynchroniczne**, ale nie może blokować pętli zdarzeń. Pracę
  synchroniczną (dysk, `subprocess`) oddaj do `asyncio.to_thread`.
* Błąd zwracaj jako `ToolResult.failure(...)`, nie jako wyjątek. Wyjątek
  zobaczy tylko log; `failure` wróci do modelu, który może spróbować inaczej.
* `display` czyta CZŁOWIEK, `data` czyta MODEL. To nie to samo.

### 3. Deklaracja

```python
from security.risk import RiskLevel
from tools.base import ToolSpec

PogodaTool(ToolSpec(
    name="weather.forecast",        # obszar.czynność, małymi literami
    description="Weather forecast for the next few days for a place.",
    args_model=PogodaArgs,
    risk=RiskLevel.MEDIUM,
))
```

**`description` czyta model — pisz je po angielsku i konkretnie.** To jedyna
podstawa, na której model decyduje, kiedy narzędzia użyć. Zły opis daje
narzędzie, którego model nigdy nie zawoła albo woła zawsze.

### 4. Poziom ryzyka — cztery, bez „to zależy"

| Poziom | Znaczenie | Zachowanie |
|---|---|---|
| `SAFE` | tylko odczyt, nic nie zmienia | wykonuje się bez pytania |
| `MEDIUM` | zmienia coś **odwracalnego** | wykonuje się bez pytania |
| `HIGH` | skutków nie da się cofnąć | **zawsze pyta użytkownika** |
| `CRITICAL` | może zepsuć system | domyślnie **zablokowane** |

Ryzyko wolno **podnieść** po zajrzeniu w argumenty, nigdy obniżyć:

```python
def dynamic_risk(self, args: UsunArgs) -> RiskLevel:
    # Jeden plik to HIGH; całe drzewo katalogów to już inna rozmowa.
    return RiskLevel.CRITICAL if args.recursive else RiskLevel.HIGH
```

Przy wątpliwości wybierz wyżej. Domyślne ryzyko w `BaseTool` to `CRITICAL`
(czyli zablokowane) — to nie złośliwość, tylko wybór strony, po której ma być
błąd.

### 5. Pytanie o zgodę buduje NARZĘDZIE, nie model

```python
def confirmation(self, args, *, language="en") -> ConfirmationRequest | None:
    return ConfirmationRequest.build(
        tool=self.spec.name,
        risk=RiskLevel.HIGH,
        summary=f"usunąć plik {args.path}",
        details=[f"rozmiar: {rozmiar} B", "tego nie da się cofnąć"],
        language=language,
    )
```

Gdyby treść pytania układał model, mógłby napisać „drobna operacja porządkowa"
i skłonić do zgody na coś innego, niż się dzieje. Dlatego nie układa.

### 6. Dostępność na tej maszynie

```python
def available(self) -> tuple[bool, str]:
    if shutil.which("pdftotext") is None:
        return False, "brak programu pdftotext (pakiet poppler-utils)"
    return True, ""
```

**Nie zakładaj, że coś jest zainstalowane.** Narzędzie niedostępne jest
niewidoczne dla modelu i pokazuje powód w `--check-deps` — zamiast wywalać się
przy pierwszym wywołaniu.

### 7. Rejestracja i testy

Dopisz narzędzie w `tools/registry.py` (odpowiednia grupa). Potem test:

```python
async def test_pogoda_zwraca_temperature(settings):
    tool = PogodaTool(SPEC)
    wynik = await tool.run(PogodaArgs(miasto="Kraków"), ToolContext(settings=settings))
    assert wynik.ok
```

Minimum, którego oczekujemy od nowego narzędzia:

- [ ] przypadek udany,
- [ ] odrzucenie złych argumentów (walidacja robi swoje),
- [ ] `available()` mówi prawdę, gdy zależności brakuje,
- [ ] `dry_run` nie robi nic w świecie,
- [ ] przy ryzyku ≥ HIGH: odmowa użytkownika **naprawdę** wstrzymuje akcję.

**Bez prawdziwej sieci, dysku poza `tmp_path` i bez sprzętu.** Wzory atrap są
w `tests/conftest.py`.

---

## Dodanie pluginu

Plugin to katalog w `plugins/`. Jego narzędzia przechodzą przez **ten sam**
router, walidację, budżet tury, politykę ryzyka i audyt. Nie ma dla nich furtki.

```bash
cp -r plugins/przyklad plugins/moj_plugin
```

Trzy elementy: wizytówka (`PluginInfo`), narzędzia (jak wyżej) i obiekt `PLUGIN`,
który znajdzie menedżer. Pełny, skomentowany szkielet jest w
[`plugins/przyklad/__init__.py`](plugins/przyklad/__init__.py); działający
przykład ze stanem w bazie — w `plugins/reminders/`.

Czego plugin **nie może**:

* obejść polityki bezpieczeństwa — jego narzędzia idą tą samą drogą,
* sięgnąć do systemu inaczej niż przez `host/` i `security/`,
* zablokować startu asystenta — plugin rzucający wyjątkiem przy ładowaniu jest
  pomijany z wpisem w logu,
* trzymać stanu w plikach obok kodu — od tego jest baza z `PluginContext`.

> **Menedżer pluginów nie jest piaskownicą.** Moduł jest importowany
> i wykonywany z pełnymi uprawnieniami konta. To ograniczenie architektury,
> opisane wprost w sekcji Ograniczeń w README — nie zgłaszaj go jako błędu.

---

## Zasady, których trzyma się reszta kodu

**Komentarz odpowiada „dlaczego", nie „co".** Kod mówi, co robi. Komentarz jest
od tego, żeby następna osoba nie „poprawiła" czegoś, co wygląda dziwnie
z powodu, o którym nie wie. Jeśli coś jest zrobione inaczej, niż podpowiada
odruch — napisz dlaczego, najlepiej z liczbą albo objawem, który to wymusił.

**Nowy tekst dla użytkownika idzie do `i18n.py`**, nie do `print()`. Katalog
angielski (`_EN`) jest wzorcem; brak tłumaczenia pokazuje tekst angielski, nigdy
pusty napis. Test pilnuje, żeby oba katalogi miały ten sam zestaw kluczy.

**Nowe ustawienie to trzy miejsca naraz**: pole w `Settings` (`config.py`), wpis
w `.env.example` z komentarzem po co ono jest, i test. Ustawienie bez wpisu
w `.env.example` jest praktycznie nie do znalezienia — pilnuje tego
`tests/test_docs.py`.

**Nic nie zakłada konkretnej maszyny.** Żadnych ścieżek bezwzględnych, nazw
użytkownika, założeń o systemie plików ani o obecności sprzętu. O system pyta
`config.detect_platform()`, o ścieżki — funkcje z `config.py`, nigdy sklejanie
stringów. Brak sprzętu ma **wyłączyć funkcję**, a nie wywrócić program.

**Degradacja zamiast awarii.** Brak mikrofonu → czat tekstowy. Brak Pipera →
odpowiedź tekstem. Brak FAISS-a → wyszukiwanie w NumPy. Brak bazy → okno
rozmowy w RAM. Każdy brak ma dać jedno zdanie wyjaśnienia i pracować dalej.

---

## Zgłaszanie błędów i pomysłów

Szablony są w [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE). Do zgłoszenia
błędu dołącz **wyjście `python main.py --check-deps`** — to jedna komenda, która
opisuje całe środowisko i oszczędza rundę pytań.

Zanim zgłosisz: przeczytaj sekcję **Ograniczenia / Known limitations** w README.
Halucynacje małego modelu, pomyłki rozpoznawania mowy w hałasie i pytanie o zgodę
przy każdej akcji HIGH to **udokumentowane właściwości**, nie błędy.

## Pull requesty

* jedna zmiana = jeden PR; opisz **dlaczego**, nie tylko co,
* `pytest`, `ruff check .` i `mypy .` mają przechodzić (CI sprawdza to samo),
* nowe zachowanie ma mieć test — najlepiej taki, który bez poprawki nie przechodzi,
* zmiana w zachowaniu widocznym dla użytkownika = aktualizacja README w tym
  samym PR-ze.

Uwaga na stan bazowy: `ruff check .` zgłasza ~430 istniejących trafień
(w większości celowo leniwe importy warstw sprzętowych i polskie znaki
w tekstach), a `mypy .` ma trafienia w testach. **Nie naprawiaj ich przy okazji**
— porównuj z tym, co było przed Twoją zmianą.
