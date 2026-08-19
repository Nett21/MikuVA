# Contributing

Thanks for wanting to add something. This document covers the two things that
are done here most often — **a new tool** and **a new plugin** — plus the rules
the rest of the code follows.

Everything below can be checked before you send a change:

```bash
pip install -r requirements-dev.txt
pytest          # ~1350 tests, ~25 s, no microphone and no GPU needed
ruff check .
mypy .
```

---

## Before you start: how this is put together

Dependencies run **one way**:

```
config.py  ──►  audio/   host/   security/
     │              │       │        │
     └──────────►  database/ ◄───────┘
                       │
                    brain/  ◄──  tools/  ◄──  plugins/
                       │
                    gui/  main.py
```

`config.py` knows about nothing else. `audio/` does not know about `brain/`.
`tools/` does not know about the language model. That is what makes every layer
testable with fakes — and why the whole suite passes on a machine with no
microphone, no GPU and no running Ollama.

**If your change needs an import going the other way, it almost certainly landed
in the wrong layer.** Say so in the PR rather than working around it with a
local import.

---

## Adding a new tool

A tool is the only route from the model to any action at all. The model **does
not execute code** — it can only ask for something written in Python to be
called, and the router checks that before anything happens:

```
model asks → router: does this tool exist and is it enabled?
           → pydantic: do the arguments have the right types and ranges?
           → policy: what risk is this?
           → [a question to a HUMAN, when HIGH or CRITICAL]
           → only now does the code do anything
           → the result goes back to the model as text
```

### 1. The argument model

```python
from pydantic import Field
from tools.base import ToolArgs

class WeatherArgs(ToolArgs):
    city: str = Field(min_length=1, max_length=100)
    days: int = Field(default=1, ge=1, le=7)
```

The limits are not decoration: the arguments come **from a language model**,
which is perfectly capable of sending an empty name, a negative number, or a
40 kB string. Validation is the first line of defence and should be tight.

### 2. The body of the tool

```python
from tools.base import BaseTool, ToolContext, ToolResult

class WeatherTool(BaseTool[WeatherArgs]):
    async def run(self, args: WeatherArgs, ctx: ToolContext) -> ToolResult:
        if ctx.dry_run:
            return ToolResult.success(
                {"preview": f"I would check the weather for {args.city}"},
                display=f"(dry run) weather for {args.city}",
            )
        ...
        return ToolResult.success({"temperature": 12}, display="12 °C, cloudy")
```

* `run` is **async**, but it must not block the event loop. Hand synchronous
  work (disk, `subprocess`) to `asyncio.to_thread`.
* Return errors as `ToolResult.failure(...)`, not as exceptions. An exception is
  seen only by the log; a `failure` goes back to the model, which can then try
  something else.
* `display` is read by a HUMAN, `data` is read by the MODEL. They are not the
  same thing.

### 3. The declaration

```python
from security.risk import RiskLevel
from tools.base import ToolSpec

WeatherTool(ToolSpec(
    name="weather.forecast",        # area.action, lowercase
    description="Weather forecast for the next few days for a place.",
    args_model=WeatherArgs,
    risk=RiskLevel.MEDIUM,
))
```

**`description` is read by the model — write it in English and concretely.** It
is the only basis on which the model decides when to use the tool. A bad
description gives you a tool the model never calls, or calls always.

### 4. Risk level — four of them, no "it depends"

| Level | Meaning | Behaviour |
|---|---|---|
| `SAFE` | read only, changes nothing | runs without asking |
| `MEDIUM` | changes something **reversible** | runs without asking |
| `HIGH` | the consequences cannot be undone | **always asks the user** |
| `CRITICAL` | can break the system | **blocked** by default |

The risk may be **raised** after inspecting the arguments, never lowered:

```python
def dynamic_risk(self, args: DeleteArgs) -> RiskLevel:
    # One file is HIGH; a whole directory tree is a different conversation.
    return RiskLevel.CRITICAL if args.recursive else RiskLevel.HIGH
```

When in doubt, pick higher. The default risk in `BaseTool` is `CRITICAL` (that
is, blocked) — not out of spite, but as a choice of which side the error should
fall on.

### 5. The consent prompt is composed by the TOOL, not the model

```python
def confirmation(self, args, *, language="en") -> ConfirmationRequest | None:
    return ConfirmationRequest.build(
        tool=self.spec.name,
        risk=RiskLevel.HIGH,
        summary=f"delete the file {args.path}",
        details=[f"size: {size} B", "this cannot be undone"],
        language=language,
    )
```

If the model composed the prompt, it could write "a small tidy-up operation" and
obtain consent for something other than what happens. So it does not compose it.

### 6. Availability on this machine

```python
def available(self) -> tuple[bool, str]:
    if shutil.which("pdftotext") is None:
        return False, "the pdftotext program is missing (package poppler-utils)"
    return True, ""
```

**Do not assume anything is installed.** A tool that is unavailable is invisible
to the model and shows the reason in `--check-deps` — rather than blowing up on
the first call.

### 7. Registration and tests

Register the tool in `tools/registry.py` (the appropriate group). Then a test:

```python
async def test_weather_returns_a_temperature(settings):
    tool = WeatherTool(SPEC)
    result = await tool.run(WeatherArgs(city="Kraków"), ToolContext(settings=settings))
    assert result.ok
```

The minimum we expect from a new tool:

- [ ] the successful case,
- [ ] rejection of bad arguments (validation doing its job),
- [ ] `available()` tells the truth when a dependency is missing,
- [ ] `dry_run` does nothing to the world,
- [ ] at risk ≥ HIGH: a refusal by the user **actually** stops the action.

**Without real network access, without disk outside `tmp_path`, and without
hardware.** Patterns for the fakes are in `tests/conftest.py`.

---

## Adding a plugin

A plugin is a directory in `plugins/`. Its tools go through **the same** router,
validation, per-turn budget, risk policy and audit. There is no side door for
them.

```bash
cp -r plugins/przyklad plugins/my_plugin
```

Three elements: the business card (`PluginInfo`), the tools (as above) and a
`PLUGIN` object for the manager to find. A full, commented skeleton is in
[`plugins/przyklad/__init__.py`](plugins/przyklad/__init__.py); a working example
with state in the database is in `plugins/reminders/`.

What a plugin **cannot** do:

* bypass the security policy — its tools travel the same road,
* reach the system other than through `host/` and `security/`,
* block the assistant from starting — a plugin that raises while loading is
  skipped, with an entry in the log,
* keep state in files next to the code — the database from `PluginContext` is
  there for that.

> **The plugin manager is not a sandbox.** The module is imported and executed
> with the full privileges of the account. That is a limitation of the
> architecture, stated plainly in the Limitations section of the README — do not
> report it as a bug.

---

## Rules the rest of the code follows

**A comment answers "why", not "what".** The code says what it does. A comment
exists so that the next person does not "fix" something that looks odd for a
reason they do not know about. If something is done differently from the obvious
way — write down why, preferably with the number or the symptom that forced it.

**New user-facing text goes into `i18n.py`**, not into `print()`. The English
catalogue (`_EN`) is the reference; a missing translation shows the English text,
never an empty label. A test checks that both catalogues carry the same set of
keys.

**A new setting means three places at once**: a field in `Settings`
(`config.py`), an entry in `.env.example` with a comment on what it is for, and
a test. A setting with no entry in `.env.example` is practically undiscoverable —
`tests/test_docs.py` enforces this.

**Nothing assumes a particular machine.** No absolute paths, no user names, no
assumptions about the file system or about hardware being present. Ask
`config.detect_platform()` about the system and the functions in `config.py`
about paths, never string concatenation. Missing hardware must **disable a
feature**, not bring the program down.

**Degradation instead of failure.** No microphone → text chat. No Piper → answers
in text. No FAISS → similarity in NumPy. No database → the conversation window in
RAM. Every absence should produce one sentence of explanation and carry on.

---

## Reporting bugs and ideas

The templates are in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE). Attach
the output of **`python main.py --check-deps`** to a bug report — it is one
command that describes the whole environment and saves us both a round of
questions.

Before you report: read the **Limitations / Known limitations** section of the
README. Hallucinations from a small model, speech-recognition mistakes in noise,
and being asked for consent on every HIGH action are **documented properties**,
not bugs.

## Pull requests

* one change = one PR; describe **why**, not just what,
* `pytest`, `ruff check .` and `mypy .` must pass (CI checks the same),
* new behaviour needs a test — preferably one that fails without the fix,
* a change in user-visible behaviour means updating the README in the same PR.

Mind the baseline: `ruff check .` reports ~430 pre-existing findings (mostly
deliberately lazy imports of the hardware layers) and `mypy .` has findings in
the tests. **Do not fix those in passing** — compare against what was there
before your change.
