"""Język interfejsu — teksty, które asystent pokazuje człowiekowi (Faza 10).

Trzy różne „języki" w tym projekcie łatwo pomylić, więc nazwijmy je wprost:

``LANGUAGE`` / ``speech_language``
    język, w którym **model odpowiada** i w którym asystent mówi (Fazy 1–4),
``UI_LANGUAGE``
    język **interfejsu**: napisy na przyciskach, komunikaty w terminalu, opisy
    stanu. To jest ten moduł,
język wypowiedzi użytkownika
    rozpoznawany osobno przy ``LANGUAGE=auto``.

Domyślnie interfejs jest **po angielsku**, tak jak domyślny język odpowiedzi.
Polski zostaje pełnoprawnym wariantem: ``UI_LANGUAGE=pl`` w ``.env``.

Zasady, które ten moduł egzekwuje:

* **angielski jest katalogiem wzorcowym** — brak tłumaczenia w innym języku
  oznacza pokazanie tekstu angielskiego, nigdy pustego napisu ani klucza,
* **brakujący klucz nie wywraca interfejsu** — zwracany jest sam klucz i wpis w
  logu; okno bez jednego napisu jest lepsze niż okno, które się nie otwiera,
* **żadnego stanu globalnego przy każdym wywołaniu**: język ustawia się raz przy
  starcie (``set_ui_language``), a ``t()`` jest tanie i bezpieczne wątkowo.

Tekst dobiera się kluczem (``t("gui.send")``), a nie przez porównywanie
łańcuchów — dzięki temu test sprawdza, że oba katalogi mają ten sam zestaw kluczy
i że żaden napis nie jest pusty.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Final

logger = logging.getLogger(__name__)

# Języki interfejsu, dla których istnieje pełny katalog.
SUPPORTED_UI_LANGUAGES: Final[tuple[str, ...]] = ("en", "pl")
DEFAULT_UI_LANGUAGE: Final[str] = "en"

_lock = threading.RLock()
_current: str = DEFAULT_UI_LANGUAGE
_missing_reported: set[str] = set()


# --------------------------------------------------------------------------- #
# Katalog angielski — wzorzec. Każdy klucz MUSI tu być.
# --------------------------------------------------------------------------- #

_EN: Final[dict[str, str]] = {
    # --- wspólne ---------------------------------------------------------- #
    "common.none": "none",
    "common.unknown": "unknown",
    "common.off": "off",
    "common.on": "on",
    "common.yes": "yes",
    "common.no": "no",
    "common.missing": "missing",
    "common.available": "available",
    "common.disabled": "disabled",
    "common.enabled": "enabled",
    "common.checking": "checking...",
    "common.not_checked": "not checked",
    "common.dash": "—",
    # --- okno: pasek górny i dół ------------------------------------------ #
    "gui.subtitle": "local assistant",
    "gui.window_title": "{name} — local assistant {version}",
    "gui.speech": "Speech",
    "gui.speech_missing": "Speech (unavailable)",
    "gui.new_conversation": "New chat",
    'ollama.remote': (
        'OLLAMA_HOST points at another machine ({host}) — I am not starting'
        ' anything there.'
    ),
    'ollama.missing_binary': 'Ollama is not installed on this machine (no `ollama` in PATH).',
    'ollama.install_hint': (
        'install it: https://ollama.com/download — or point OLLAMA_HOST at '
        'another machine'
    ),
    'ollama.start_failed': 'Could not start `ollama serve`.',
    'ollama.started': 'Ollama was not running — I started it in the background.',
    'ollama.service_hint': 'to have it always on: systemctl enable --now ollama',
    'ollama.exited': '`ollama serve` exited immediately (code {code}) — the port may be taken.',
    'ollama.timeout': '`ollama serve` did not answer within {seconds} s.',
    'ollama.log_hint': 'details: {path}',
    'cli.gui_fallback': (
        'The graphical interface is unavailable ({reason}) — starting in '
        'the terminal.'
    ),
    'deps.compute.name': 'Ollama compute',
    'deps.compute.gpu': 'GPU — {size} of the model in VRAM',
    'deps.compute.cpu_with_gpu': (
        'CPU, although a CUDA GPU is present ({gpu}) — answers are several '
        'times slower'
    ),
    'deps.compute.cpu': 'CPU (no CUDA GPU detected)',
    'deps.compute.unknown': 'not checked — no model is loaded yet',
    'deps.compute.hint_pacman': (
        'sudo pacman -S ollama-cuda   (the `ollama` package is the CPU-only'
        ' build)'
    ),
    'deps.compute.hint_generic': 'install the CUDA build of Ollama: https://ollama.com/download',
    'runtime.wake_ignored': (
        'I heard speech without the wake phrase “{phrase}” — press Listen '
        'or say the phrase.'
    ),
    'deps.whisper.cuda_missing': (
        'CUDA is unusable ({library} is missing) — Whisper runs on the CPU'
    ),
    'deps.whisper.cuda_hint_pacman': (
        'sudo pacman -S cuda cudnn   (or set WHISPER_DEVICE=cpu to stop '
        'trying)'
    ),
    'deps.whisper.cuda_hint_generic': (
        'install the CUDA runtime with cuBLAS and cuDNN, or set '
        'WHISPER_DEVICE=cpu'
    ),
    "gui.settings": "Settings",
    # Warianty na wąskie okno: menedżery kafelkowe potrafią dać okno o połowie
    # oczekiwanej szerokości i wtedy pełne napisy wypychają przyciski za krawędź.
    "gui.new_short": "New",
    "gui.settings_short": "Settings",
    "gui.status_show": "Status",
    "gui.status_hide": "Hide status",
    "gui.listen": "Listen",
    "gui.stop_listening": "Stop listening",
    "gui.send": "Send",
    "gui.interrupt": "Interrupt",
    "gui.input_placeholder": "Type a message or press “Listen”...",
    "gui.new_conversation_notice": (
        "New chat. The window is cleared; facts and notes stay in memory."
    ),
    "gui.closing": "Closing...",
    "gui.session_model": (
        "Model for this session: {model}. To set it permanently, use OLLAMA_MODEL in .env."
    ),
    # --- okno: role w rozmowie -------------------------------------------- #
    "gui.role.user": "You",
    "gui.role.system": "System",
    "gui.role.tool": "Tool",
    "gui.role.error": "Error",
    "gui.role.assistant_fallback": "Assistant",
    "gui.detail.microphone": "microphone",
    "gui.detail.voice_sample": "voice sample",
    # --- okno: panel stanu ------------------------------------------------- #
    "gui.status.title": "Status — {name}",
    "gui.status.model": "Model: {model}",
    "gui.status.language": "Reply language: {language}",
    "gui.status.language_auto": "auto (detected per utterance)",
    "gui.status.language_forced": "{code} (configured)",
    "gui.status.no_model": "none",
    "service.mic": "Microphone",
    "service.wake": "Wake word",
    "service.whisper": "Whisper",
    "service.ollama": "Ollama",
    "service.speech": "Speech",
    "service.memory": "Memory",
    "service.tools": "Tools",
    # --- okno: wskaźnik nasłuchiwania -------------------------------------- #
    "listening.off": "microphone off",
    "listening.idle": "ready",
    "listening.waiting_wake": "waiting for “{phrase}”",
    "listening.waiting_wake_generic": "waiting for the wake word",
    "listening.listening": "listening...",
    "listening.transcribing": "recognising speech...",
    "listening.thinking": "thinking...",
    "listening.speaking": "speaking...",
    # --- okno: ustawienia -------------------------------------------------- #
    "settings.title": "Settings",
    "settings.back": "Back to chat",
    "settings.save": "Save",
    "settings.revert": "Revert",
    "settings.browse": "Browse file...",
    "settings.clear": "Clear",
    "settings.listen_sample": "Play sample",
    "settings.section.assistant": "Assistant",
    "settings.section.voice": "Speech",
    "settings.section.rvc": "Voice conversion (RVC)",
    "settings.reverted": "Restored the values from the settings file.",
    "settings.needs_reload_suffix": "  (requires reloading speech)",
    "settings.dialog_title": "Choose a file: {label}",
    "settings.dialog_unavailable": (
        "The system file dialog is unavailable — type the path by hand."
    ),
    "settings.auto_voice": "(matched to the language)",
    "settings.field.assistant_name": "Assistant name",
    "settings.help.assistant_name": (
        "Window title, headings and the default wake phrase (“hey <name>”)."
    ),
    "settings.placeholder.assistant_name": "e.g. Miku",
    "settings.field.ui_accent_color": "Accent colour",
    "settings.help.ui_accent_color": (
        "The whole theme is derived from this one colour: backgrounds, bubbles, "
        "the listening indicator."
    ),
    "settings.field.personality_traits": "Character traits",
    "settings.help.personality_traits": (
        "Added to the prompt as STYLE. They do not change the rules or grant new "
        "capabilities."
    ),
    "settings.placeholder.personality_traits": "e.g. keeps answers short, likes cycling metaphors",
    "settings.field.voice_engine": "Speech engine",
    "settings.help.voice_engine": (
        "“none” turns speaking off. Changing it requires reloading speech."
    ),
    "settings.field.piper_model": "Piper voice",
    "settings.help.piper_model": (
        "Empty = voice matched to the reply language. Changing it requires reloading speech."
    ),
    "settings.field.speech_language": "Spoken language",
    "settings.help.speech_language": (
        "The language YOU speak — it does not have to match the reply language. "
        "One code (“pl”) forces it. Speak two languages? List them: “pl,en” — the "
        "language is then recognised, but only ever picked from your list. "
        "“auto” follows the assistant's language, “detect” allows any language."
    ),
    "settings.placeholder.speech_language": "pl,en — or auto, detect…",
    "settings.field.rvc_enabled": "RVC voice conversion",
    "settings.help.rvc_enabled": (
        "Stays off without a model file, even when this switch is on."
    ),
    "settings.field.rvc_model_path": "RVC model (.pth)",
    "settings.help.rvc_model_path": (
        "Pick the file with the button — hand-typed paths are the most common source of typos."
    ),
    "settings.field.rvc_index_path": "RVC index (.index)",
    "settings.help.rvc_index_path": (
        "Optional. It improves the timbre, but conversion works without it."
    ),
    "settings.field.rvc_pitch_shift": "Pitch shift (semitones)",
    "settings.help.rvc_pitch_shift": "0 = unchanged. +12 is one octave up.",
    "settings.field.rvc_index_rate": "Index ratio",
    "settings.help.rvc_index_rate": "0 = model only, 1 = maximum influence of the index.",
    "settings.filter.rvc_model": "RVC model",
    "settings.filter.rvc_index": "RVC index",
    "settings.filter.all_files": "All files",
    # --- okno: wyniki zapisu ustawień -------------------------------------- #
    "settings.result.nothing": "Nothing changed.",
    "settings.result.not_saved": "Settings were not saved.",
    "settings.result.saved_reload": (
        "Saved: {fields}. The new speech settings take effect after reloading the "
        "speech engine — doing that now."
    ),
    "settings.result.saved": "Saved: {fields}. The changes are in effect.",
    "settings.problem.color": "The colour must be hexadecimal, e.g. #39C5BB or #3CB.",
    "settings.problem.missing_file": (
        "File not found: {path}. The path will be saved, but speech will not use it."
    ),
    "settings.problem.wrong_suffix": (
        "The file does not end with {suffix} — check whether it is the right file."
    ),
    "settings.problem.rvc_without_model": (
        "RVC is enabled but no model was selected — conversion stays off."
    ),
    "settings.problem.write_failed": "Write error: {error}",
    # --- okno: potwierdzenia ----------------------------------------------- #
    "gui.confirm.allow": "Allow",
    "gui.confirm.cancel": "Cancel",
    # --- okno: komunikaty wątku roboczego ---------------------------------- #
    "runtime.memory_unavailable": (
        "Long-term memory is unavailable ({reason}). The chat works, but nothing "
        "will survive closing the window."
    ),
    "runtime.tools_unavailable": "Tools unavailable ({reason}) — chat only.",
    "runtime.tools_failed": "Could not prepare the tools: {error}",
    "runtime.memory_interrupted": "Saving to memory was interrupted.",
    "runtime.memory_failed": "Could not handle the memory command: {error}",
    "runtime.compacting": "The chat got long — summarising older threads...",
    "runtime.generation_interrupted": "Generation interrupted.",
    "runtime.empty_reply": "The model returned no content.",
    "runtime.unexpected_error": "Unexpected error: {error}",
    "runtime.command_failed": "Could not run the command: {error}",
    "runtime.speech_reloaded": "Speech reloaded: {detail}",
    "runtime.speech_still_off": "Speech stays off ({detail}).",
    "runtime.speech_no_sample": "Nothing to play the sample with ({detail}).",
    "runtime.voice_sample": "Hi, I'm {name}. This is how I sound on this computer.",
    "runtime.speech_error_hint": (
        "Answers stay text-only — you can turn speech back on with the switch."
    ),
    "runtime.mic_unavailable": "Voice mode unavailable: {reason}",
    "runtime.mic_unavailable_hint": "Typing works normally.",
    "runtime.mic_failed": "Could not start the microphone: {error}",
    "runtime.mic_failed_hint": "Details were written to the logs.",
    "runtime.mic_error": "Voice input error: {error}",
    "runtime.mic_error_hint": "Turning the microphone off; typing still works.",
    "runtime.stopped_working": "The assistant stopped working: {error}",
    "runtime.ollama_no_models": "Ollama did not answer ({error})",
    "runtime.thinking": "the model is working on the question...",
    "runtime.confirm_window_closed": "the window was closed",
    "runtime.user_denied": "the user declined",
    "runtime.mic_dummy": "microphone off",
    "runtime.wake_disabled": "off (phrase “{phrase}”)",
    "runtime.wake_pending": "“{phrase}” — engine decided when listening starts",
    "runtime.wake_no_gate": "listening without a gate (phrase “{phrase}”)",
    "runtime.wake_active": "“{phrase}” ({engine})",
    "runtime.speech_muted": "muted",
    "runtime.speech_off": "off",
    "runtime.tools_off": "off",
    # --- uruchamianie GUI -------------------------------------------------- #
    "gui.unavailable": "The graphical interface is unavailable: {reason}",
    "gui.terminal_hint": "The terminal mode works without a GUI: python main.py --terminal",
    "gui.window_failed": "Could not open a window: {error}",
    "gui.no_display_hint": (
        "This usually means no access to a graphical session. Terminal mode: "
        "python main.py --terminal"
    ),
    "gui.crashed": "The graphical interface failed: {error}",
    "gui.missing_tk": "no Tk library for this Python ({error})",
    "gui.missing_toolkit": "the {package} package is missing ({error})",
    "gui.no_session": "no graphical session detected (no DISPLAY and no WAYLAND_DISPLAY)",
    "gui.ready": "CustomTkinter is ready",
    "gui.hint.tk_generic": (
        "install the Tk library for your Python (a system package, not pip)"
    ),
    "gui.hint.tk_windows": (
        "install Python with the “tcl/tk and IDLE” option "
        "(python.org installer → Modify → Optional Features)"
    ),
    # --- sprawdzenia zależności GUI ---------------------------------------- #
    "deps.tk.name": "Tk (system library)",
    "deps.tk.ok": "available",
    "deps.tk.missing": "missing — the GUI will not start, terminal mode works normally",
    "deps.gui.name": "GUI (CustomTkinter)",
    "deps.gui.ok": "available",
    "deps.gui.needs_tk": "the package is installed but unusable without the Tk library",
    "deps.gui.missing": "package missing",
    "deps.gui.purpose": "graphical interface (python main.py --gui)",
    "deps.display.name": "Graphical session",
    "deps.display.ok": "detected",
    "deps.display.missing": (
        "none (headless server, SSH without X11, a service) — use --terminal"
    ),
    "deps.display.hint": "python main.py --terminal",
    'report.system': 'System:      {label} ({machine})',
    'report.wsl': '             (WSL detected)',
    'report.python': 'Python:      {version} — {executable}',
    'report.package_manager': 'Package manager: {manager} → install script: {script}',
    'report.dependencies': 'Dependencies:',
    'report.ok': 'ok  ',
    'report.missing': 'MISSING',
    'report.required': 'required',
    'report.optional': 'optional',
    'report.path': 'path: {path}',
    'report.missing_required': '{count} required items are missing:',
    'report.nothing_automatic': (
        'Nothing is installed automatically. To finish the installation: '
        '{command}'
    ),
    'report.all_present': 'All required dependencies are available.',
    'report.optional_missing': 'Optional items that were not detected:',
    'cli.header.title': ' {name} — local assistant, version {version}',
    'cli.header.model': ' Model:    {model} @ {host}',
    'cli.header.mode': ' Mode:     {mode}',
    'cli.header.system': ' System:   {label} ({machine})',
    'cli.header.gpu': ' GPU:      {detail}',
    'cli.header.logs': ' Logs:     {path}',
    'cli.header.commands': (
        'Commands: /help, /status, /mic, /wake, /voice, /memory, /clear, /reload,'
        ' /deps, /exit'
    ),
    'cli.header.mic_found': 'Microphone detected — /mic switches from typing to talking.',
    'cli.header.wake': 'Wake word: “{phrase}” — change it with /wake phrase <text>.',
    'cli.header.no_voice': 'Voice mode unavailable ({detail}).',
    'cli.help.title': 'Available commands:',
    'cli.help.body': (
        '        /help       — this list\n        /status     — model, chat '
        'history, user settings\n        /mic        — turn talking on/off '
        'instead of typing\n        /mic list   — show the detected microphones\n'
        '        /wake       — wake word status\n        /wake phrase <text> — '
        'change the phrase (saved in user_settings.json)\n        /wake now   — '
        'open the chat window without saying the phrase\n        /wake off   — '
        'listen without the gate until the end of the session\n        /voice'
        '      — turn speaking of answers on/off\n        /voice list — show the '
        'Piper voices that were found\n        /voice test — say a sample '
        'sentence\n        /voice model <name> — change the voice (saved in '
        'user_settings.json)\n        /voice save <file.wav> — write a voice '
        'sample to a file\n        /memory     — what the assistant remembers '
        '(database, facts, summaries)\n        /memory facts              — list '
        'the remembered facts\n        /memory remember k=v       — remember a '
        'fact permanently\n        /memory forget <key>       — delete a fact\n'
        '        /memory note <text>        — save a note\n        /memory search'
        ' <phrase>    — search chats and notes (by words)\n        /memory recall'
        ' <phrase>    — find memories by MEANING\n        /memory reindex'
        '            — recompute embeddings for the whole memory\n        '
        'Remember that ...          — save something permanently (no slash)\n'
        '        Forget that ...            — remove it from memory\n        '
        '/tools      — tools, their risk level and recent calls\n        /clear'
        '      — start a new chat (facts and notes stay)\n        /reload     — '
        're-read config/user_settings.json\n        /deps       — check the '
        'dependencies again\n        /exit       — quit (or Ctrl+D)'
    ),
    'cli.status.model': 'Model:      {model} @ {host}',
    'cli.status.mode': 'Mode:       {mode}',
    'cli.status.history': (
        'History:    {messages}/{max_messages} messages, {chars}/{max_chars} '
        'characters'
    ),
    'cli.status.memory': 'Memory:     {detail}',
    'cli.status.saved': 'Stored:     {detail}',
    'cli.status.semantic': 'Recall:     {detail}',
    'cli.status.summary': 'Summary:    {detail}',
    'cli.status.tools': 'Tools:      {detail}',
    'cli.status.audit': 'Audit:      {detail}',
    'cli.status.mic': 'Microphone: {detail}',
    'cli.status.wake': 'Wake word:  {detail}',
    'cli.status.speech_language': 'Speech lang: {detail}',
    'cli.status.language_auto': 'auto (detected for every utterance)',
    'cli.status.language_forced': '{code} (forced for recognition and answers)',
    'cli.status.assistant': 'Assistant:  {name} (tag {tag})',
    'cli.status.accent': 'GUI colour: {color}',
    'cli.status.speech': 'Speech:     {detail}',
    'cli.status.speech_not_started': 'not started',
    'cli.status.engine': (
        'Engine:     {engine}, voice: {voice}, speed {speed}x, volume {volume} / '
        'RVC: {rvc}'
    ),
    'cli.status.traits': 'Traits:     {detail}',
    'cli.status.tools_off': 'off (TOOLS_ENABLED=false)',
    'cli.status.auto_voice': '(matched to the language)',
    'cli.status.none': '(none)',
    'cli.voice.mic_on': 'on ({detail})',
    'cli.voice.mic_unavailable': 'unavailable — {reason}',
    'cli.voice.mic_off': 'off',
    'cli.voice.wake_source_file': 'wake_word in user_settings.json',
    'cli.voice.wake_source_name': "the assistant's name",
    'cli.voice.wake_off': 'off (phrase “{phrase}”, source: {source})',
    'cli.voice.wake_pending': (
        '“{phrase}” (source: {source}), the engine is chosen when listening '
        'starts'
    ),
    'cli.voice.wake_no_gate': 'unavailable — listening without a gate (phrase “{phrase}”)',
    'cli.voice.wake_awake': 'chat window open',
    'cli.voice.wake_waiting': 'waiting for the phrase',
    'cli.voice.wake_active': '“{phrase}” (source: {source}), {engine}, {state}',
    'cli.voice.give_phrase': 'Give a phrase, e.g. /wake phrase hey aiko',
    'cli.voice.new_phrase': 'New phrase: “{phrase}” (saved in config/user_settings.json)',
    'cli.voice.gate_off': 'Wake gate off until the end of the session — talk without calling.',
    'cli.voice.gate_on': 'Wake gate on.',
    'cli.voice.no_voice_mode': 'Voice mode is not running — use /mic first.',
    'cli.voice.window_open': 'Chat window opened without the wake phrase.',
    'cli.voice.wake_status': 'Wake word: {detail}',
    'cli.voice.wake_usage': 'Usage: /wake [on|off|now|phrase <text>]',
    'cli.voice.hint': '       Hint: {detail}',
    'cli.voice.mode_unavailable': 'Voice mode unavailable: {reason}',
    'cli.voice.install': '        Install the dependencies: {hint}',
    'cli.voice.preparing': 'preparing voice input (the first run loads the model)...',
    'cli.voice.staying_text': 'Staying in text mode.',
    'cli.voice.start_failed': 'Could not start voice mode: {error}',
    'cli.voice.details_in': 'Details were written to {path}',
    'cli.voice.listen_interrupted': (
        'Listening interrupted — voice mode off (/mic turns it back on).'
    ),
    'cli.voice.listen_error': 'Voice input error: {error}',
    'cli.voice.listen_disabled': 'Turning voice mode off; the text chat keeps working.',
    'cli.voice.nothing_heard': 'I heard nothing — you can type instead.',
    'cli.voice.no_audio_packages': 'Audio packages are missing ({error}).',
    'cli.voice.no_input_devices': 'The system reports no input device.',
    'cli.voice.devices': 'Microphones available (part of the name → AUDIO_INPUT_DEVICE):',
    'cli.speech.muted': 'muted for this session (/voice on turns it back)',
    'cli.speech.on': 'on ({detail})',
    'cli.speech.unavailable': 'unavailable — {reason}',
    'cli.speech.off': 'off',
    'cli.speech.speech_unavailable': 'Speech unavailable: {reason}',
    'cli.speech.disabled_in_settings': 'Speech is turned off in the settings ({detail}).',
    'cli.speech.text_only': 'Answers stay text-only.',
    'cli.speech.start_failed': 'Could not start speech: {error}',
    'cli.speech.no_voices': 'I found no voice (.onnx). I looked in:',
    'cli.speech.download_voice': 'Download a voice: python scripts/prepare_offline.py --piper',
    'cli.speech.voices': 'Voices available (name → piper_model in user_settings.json):',
    'cli.speech.selected_marker': '  <- selected',
    'cli.speech.auto_voice_note': (
        'piper_model is empty — the voice is matched to the reply language.'
    ),
    'cli.speech.give_voice': 'Give a voice name, e.g. /voice model pl_PL-darkman-medium',
    'cli.speech.voice_not_found': 'I could not find the voice “{name}”.',
    'cli.speech.list_hint': 'The list is shown by: /voice list',
    'cli.speech.new_voice': 'New voice: {detail} (saved in config/user_settings.json)',
    'cli.speech.saying': 'Saying: {text}',
    'cli.speech.sample': "Hi, I'm {name}. This is how I sound on this computer.",
    'cli.speech.give_file': 'Give a file name, e.g. /voice save sample.wav',
    'cli.speech.sample_saved': 'Voice sample saved: {path}',
    'cli.speech.muted_now': 'Speech muted.',
    'cli.speech.already_off': 'Speech was already off.',
    'cli.speech.already_on': 'Speech is already running.',
    'cli.speech.state': 'Speech: {detail}',
    'cli.speech.usage': 'Usage: /voice [on|off|list|test|model <name>|save <file>]',
    'cli.audio.device': 'Device: {detail}',
    'cli.audio.system_default': 'system default',
    'cli.audio.measuring': 'Measuring the background for {seconds} s — do NOT speak...',
    'cli.audio.no_samples': 'No samples were captured — check whether the microphone is muted.',
    'cli.audio.frames': 'Frames: {frames}, dropped: {dropped}',
    'cli.audio.levels': (
        'Background level: 10th percentile {quiet} dBFS, median {median} dBFS, '
        'peak {peak} dBFS'
    ),
    'cli.audio.thresholds': 'How many background frames would count as speech at each threshold:',
    'cli.audio.suggested': '  <- suggested',
    'cli.audio.frames_share': (
        '        VAD_ENERGY_THRESHOLD_DB={threshold} {share}% of '
        'frames{marker}'
    ),
    'cli.audio.active_vad': 'Active VAD engine: {name}',
    'cli.audio.webrtc_note': 'You are using webrtcvad — the energy threshold is not used at all.',
    'cli.audio.write_env': 'Write this into .env: VAD_ENERGY_THRESHOLD_DB={value}',
    'cli.audio.better': (
        'Even better: pip install webrtcvad-wheels (a speech model instead of an '
        'energy threshold)'
    ),
    'cli.audio.too_loud': (
        'The background is very loud — consider another microphone or a lower '
        'gain.'
    ),
    'cli.audio.recommended': 'Recommended: pip install webrtcvad-wheels',
    'cli.mic.off': 'Voice mode off.',
    'cli.mic.already_off': 'Voice mode was already off.',
    'cli.mic.already_on': 'Voice mode is already running.',
    'cli.mic.on': 'Voice mode on — speak after the [MIC] listening message.',
    'cli.mic.usage': 'Usage: /mic [on|off|list]',
    'cli.mem.state': 'Memory:     {detail}',
    'cli.mem.saved': 'Stored:     {detail}',
    'cli.mem.window': 'Window:     {messages} messages, {pending} waiting to be summarised',
    'cli.mem.semantic': 'Recall:     {detail}',
    'cli.mem.summary': 'Summary:    {detail}',
    'cli.mem.hint': 'Hint:       /help → the memory section; details: /deps',
    'cli.mem.unavailable': 'Long-term memory is unavailable ({detail}).',
    'cli.mem.no_facts': 'I have not remembered any facts yet.',
    'cli.mem.add_fact_hint': 'You can add them with: /memory remember name=Mariusz',
    'cli.mem.facts': 'Remembered facts ({count}):',
    'cli.mem.fact_line': '        - {key}: {value}   (source: {source})',
    'cli.mem.preference_line': '        ~ {key}: {value}   (preference)',
    'cli.mem.remember_usage': 'Usage: /memory remember key=value',
    'cli.mem.remembered': 'Remembered: {key} = {value}',
    'cli.mem.remember_failed': 'Could not save the fact ({error}).',
    'cli.mem.forget_usage': 'Usage: /memory forget <key>',
    'cli.mem.forgotten': 'Forgotten: {key}',
    'cli.mem.no_such_fact': 'I have no such fact: {key}',
    'cli.mem.note_usage': 'Usage: /memory note <text>',
    'cli.mem.note_failed': 'Could not save the note ({error}).',
    'cli.mem.note_saved': 'Note saved (#{id}).',
    'cli.mem.no_notes': 'No notes. You can add them with: /memory note <text>',
    'cli.mem.notes': 'Notes ({count}):',
    'cli.mem.recall_usage': 'Usage: /memory recall <phrase>',
    'cli.mem.semantic_unavailable': 'Semantic memory is unavailable ({detail}).',
    'cli.mem.word_search_works': 'Word search still works: /memory search {phrase}',
    'cli.mem.nothing_recalled': 'Nothing comes to mind for: {phrase}',
    'cli.mem.recalled': 'Associations ({count}):',
    'cli.mem.reindexing': 'Computing embeddings for the whole memory — this may take a while...',
    'cli.mem.reindex_done': 'Done: {count} vectors. {detail}',
    'cli.mem.search_usage': 'Usage: /memory search <phrase>',
    'cli.mem.nothing_found': 'I found nothing for: {phrase}',
    'cli.mem.found': 'Found ({count}):',
    'cli.mem.note_kind': 'note',
    'cli.mem.chat_kind': 'chat',
    'cli.mem.unknown_command': 'I do not know that memory command: {command}',
    'cli.mem.available_commands': (
        'Available: state, facts, remember key=value, forget <key>, note <text>, '
        'notes, search <phrase>, recall <phrase>, reindex'
    ),
    'cli.mem.save_interrupted': 'Saving to memory was interrupted.',
    'cli.mem.handle_failed': 'Could not handle the memory command: {error}',
    'cli.tools.unavailable': 'Tools unavailable ({error}) — chat only.',
    'cli.tools.failed': 'Could not prepare the tools: {error}',
    'cli.tools.disabled': 'Tools are turned off (TOOLS_ENABLED=false).',
    'cli.plugins.list': 'Plugins: {detail}',
    'plugins.reminders.notice': 'Reminder: {text}',
    'cli.tools.list': 'Tools: {detail}',
    'cli.tools.files': 'Files:      {detail}',
    'cli.tools.audit': 'Call audit: {detail}',
    'cli.tools.no_confirm_channel': (
        'No interactive terminal — tools that need consent will be refused.'
    ),
    'cli.reindex.memory_unavailable': 'Long-term memory is unavailable: {error}',
    'cli.reindex.semantic_unavailable': 'Semantic memory is unavailable: {detail}',
    'cli.reindex.details': 'Details: python main.py --check-deps',
    'cli.reindex.computing': 'Computing embeddings — the first run loads the model...',
    'cli.reindex.done': 'Done: {count} vectors in {seconds} s.',
    'cli.run.ollama_down': (
        'Ollama is not responding at {host} — the chat will not work until the '
        'service starts.'
    ),
    'cli.run.ollama_hint': 'After running `ollama serve`, type /deps to check again.',
    'cli.run.model_missing': "The model '{model}' is not pulled — get it with: ollama pull {model}",
    'cli.run.memory_unavailable': 'Long-term memory unavailable: {error}',
    'cli.run.memory_note': 'The chat works normally, but nothing will survive closing the program.',
    'cli.run.voice_on': 'Voice mode on — /mic turns it off at any time.',
    'cli.run.typing': 'Staying with typing.',
    'cli.run.speech_state': 'Speech {detail}. /voice mutes it.',
    'cli.run.speech_unavailable': 'Speech {detail} — answers will be text-only.',
    'cli.run.speech_hint': 'Details: /deps (Phase 4 items), turn on with: /voice on',
    'cli.run.new_chat': 'New chat. The window is cleared; {detail}',
    'cli.run.facts_stay': 'facts and notes stay in memory.',
    'cli.run.memory_off': 'long-term memory is off.',
    'cli.run.reloaded': 'User settings reloaded: assistant {name}, tag {tag}.',
    'cli.run.compacting': 'The chat got long — summarising older threads...',
    'cli.run.compact_interrupted': 'Summarising interrupted.',
    'cli.run.generation_interrupted': 'Generation interrupted.',
    'cli.run.unexpected': 'Unexpected error: {error}',
    'cli.run.empty_reply': 'The model returned no content.',
    'cli.run.goodbye': 'See you!',
    'cli.main.flags_conflict': '--offline and --online are mutually exclusive.',
    'cli.main.gui_conflict': '--gui and --no-gui are mutually exclusive.',
    'cli.main.logging_failed': 'Could not configure logging: {error}',
    'cli.main.deps_failed': 'Could not check the dependencies: {error}',
    'cli.main.speech_report': 'Full speech report: python main.py --check-deps',
    'cli.main.missing_required': 'Required parts of the environment are missing:',
    'cli.main.nothing_automatic': (
        'I install nothing automatically. To finish the installation: '
        '{command}'
    ),
    'cli.main.full_report': 'Full report: python main.py --check-deps',
    'cli.main.starting_anyway': 'Starting with what is available...',
    'cli.arg.description': 'Local assistant — terminal mode and environment diagnostics.',
    'cli.arg.terminal': 'chat in the terminal (default when no other option is given)',
    'cli.arg.gui': 'graphical window instead of the terminal (needs CustomTkinter and Tk)',
    'cli.arg.no_gui': 'force the terminal even when GUI_ENABLED=true',
    'cli.arg.check_deps': 'check the dependencies, print a report and exit',
    'cli.arg.audio_check': (
        'measure the background noise and suggest a VAD threshold for this '
        'microphone'
    ),
    'cli.arg.voice': 'start with voice input on (same as INPUT_MODE=voice)',
    'cli.arg.no_voice': 'force text mode even when INPUT_MODE=voice',
    'cli.arg.no_wake': 'listen without a wake word (same as WAKE_ENABLED=false)',
    'cli.arg.no_tts': 'do not read answers aloud (same as TTS_ENABLED=false)',
    'cli.arg.voice_test': 'say a sample sentence with the selected voice and exit (speech test)',
    'cli.arg.list_voices': 'list the Piper voices that were found and exit',
    'cli.arg.no_memory': (
        'write nothing to disk — the chat lives in memory only '
        '(MEMORY_ENABLED=false)'
    ),
    'cli.arg.no_embeddings': (
        'no semantic memory — do not compute embeddings '
        '(EMBEDDINGS_ENABLED=false)'
    ),
    'cli.arg.reindex_memory': (
        'compute embeddings for the whole memory and exit (after changing the '
        'model)'
    ),
    'cli.arg.no_tools': 'no tools — the model only talks (TOOLS_ENABLED=false)',
    'cli.arg.dry_run_tools': 'tools return a preview instead of acting (SECURITY_DRY_RUN=true)',
    'cli.arg.offline': 'work without the network: no model is downloaded (same as OFFLINE_MODE=on)',
    'cli.arg.online': 'allow downloading a missing Whisper model (same as OFFLINE_MODE=off)',
    'cli.arg.log_level': 'override LOG_LEVEL from .env (DEBUG, INFO, WARNING, ERROR)',
    'cli.arg.ui_lang': 'interface language: en, pl or auto (overrides UI_LANGUAGE from .env)',
    'cli.arg.metavar_text': 'TEXT',
    'deps.mode.name': 'Working mode',
    'deps.mode.hint': 'a full offline set is prepared by: python scripts/prepare_offline.py',
    'deps.python.name': 'Python',
    'deps.python.detail': 'required >= {required}, detected {detected}',
    'deps.package.name': 'package {name}',
    'deps.package.version': 'version {version}',
    'deps.package.version_purpose': 'version {version} — {purpose}',
    'deps.package.missing': 'package not installed',
    'deps.package.missing_purpose': 'package not installed — {purpose}',
    'deps.wheels.name': 'pip wheels (offline)',
    'deps.wheels.present': (
        '{count} packages — installing dependencies will work without the '
        'internet'
    ),
    'deps.wheels.missing': 'no local packages; needed only to install on a machine without network',
    'deps.wheels.hint': 'on a machine with internet: python scripts/prepare_offline.py --wheels',
    'deps.ollama.app': 'Ollama (application)',
    'deps.ollama.in_path': 'found in PATH',
    'deps.ollama.not_in_path': 'not found in PATH (fine if the server runs remotely)',
    'deps.ollama.service': 'Ollama (HTTP service)',
    'deps.ollama.responds': 'responds, version {version}',
    'deps.ollama.no_answer': 'no answer',
    'deps.ollama.start_hint': 'run `ollama serve` ({install})',
    'deps.model.name': 'LLM model ({model})',
    'deps.model.present': 'model pulled',
    'deps.model.absent': 'the model is not pulled; available: {models}',
    'deps.model.no_models': 'no models',
    'deps.model.not_checked': 'not checked — the Ollama service is not responding',
    'deps.model.offline_note': ' — needs the internet, do it beforehand',
    'deps.dir.config': 'Configuration directory',
    'deps.dir.logs': 'Log directory',
    'deps.dir.no_write': 'no write permission: {error}',
    'deps.dir.hint': (
        'grant write permission or point elsewhere with MIKU_CONFIG_DIR / '
        'MIKU_LOGS_DIR'
    ),
    'deps.dir.writable': 'writing is possible',
    'deps.cuda.name': 'CUDA acceleration',
    'deps.cuda.hint': 'no GPU is fine — the models will run on the CPU',
    'deps.gpu.none': 'no CUDA GPU detected — running on the CPU',
    'deps.gpu.nvidia_smi': 'nvidia-smi: {name} (driver {driver})',
    'deps.gpu.torch': 'torch: {name} (CUDA {driver})',
    'deps.gpu.unknown_driver': 'unknown',
    'deps.gpu.unknown_version': 'unknown version',
    'mode.forced_offline': 'offline — forced by OFFLINE_MODE=on (nothing reaches the network)',
    'mode.forced_online': 'online — forced by OFFLINE_MODE=off (models may be downloaded)',
    'mode.auto_offline': 'offline (auto) — the models are local, nothing is downloaded',
    'mode.auto_online': (
        "online (auto) — no Whisper model '{model}' in {path}, so it would be "
        "downloaded on first microphone use"
    ),
    'deps.purpose.pydantic': 'configuration validation',
    'deps.purpose.pydantic_settings': 'reading .env',
    'deps.purpose.dotenv': 'support for the .env file',
    'deps.purpose.httpx': 'HTTP client for Ollama',
    'deps.purpose.numpy': 'audio signal processing',
    'deps.purpose.sounddevice': 'microphone capture (PortAudio)',
    'deps.purpose.faster_whisper': 'speech transcription',
    'deps.purpose.webrtcvad': 'VAD; without it the built-in energy VAD is used',
    'deps.purpose.openwakeword': (
        'wake word via a KWS model; without it the Whisper detector is used'
    ),
    'deps.purpose.piper': (
        'speech synthesis; without the package a `piper` binary in PATH is '
        'enough'
    ),
    'deps.purpose.sentence_transformers': (
        'local embeddings; without the package Ollama computes them'
    ),
    'deps.purpose.faiss': 'fast vector search; without it NumPy does the work',
    'deps.mic.name': 'Microphone',
    'deps.mic.disabled': 'turned off by MIC_ENABLED=false',
    'deps.mic.disabled_hint': 'set MIC_ENABLED=true to use voice mode',
    'deps.mic.no_packages': 'sounddevice or numpy missing — devices were not checked',
    'deps.mic.error_hint': 'voice mode will be off, the text chat works normally',
    'deps.mic.no_devices': 'the system reports no input device',
    'deps.mic.no_devices_hint': 'plug in a microphone or use the text mode',
    'deps.mic.no_match': "no device matching '{wanted}'; available: {devices}",
    'deps.mic.no_match_hint': 'fix AUDIO_INPUT_DEVICE or leave it empty (default device)',
    'deps.mic.ok': '{count} input devices, selected: {selected}',
    'deps.vad.name': 'VAD (speech detection)',
    'deps.whisper.name': 'Faster-Whisper',
    'deps.whisper.detail': 'model {model}, device {device}/{compute}',
    'deps.whisper.missing': 'package not installed — voice mode will be off',
    'deps.whisper.cache_name': 'Whisper model (cache)',
    'deps.whisper.local': "the model '{model}' is on disk — voice mode works without the internet",
    'deps.whisper.absent': "no model '{model}' on disk",
    'deps.whisper.others': '; available locally: {models}',
    'deps.whisper.hint_offline': (
        'python scripts/prepare_offline.py --whisper (needs the internet, '
        'once)'
    ),
    'deps.whisper.hint_online': (
        'it downloads itself on first microphone use; to do it beforehand: python'
        ' scripts/prepare_offline.py --whisper'
    ),
    'deps.wake.name': 'Wake word',
    'deps.wake.source_file': 'wake_word in user_settings.json',
    'deps.wake.source_name': "the assistant's name",
    'deps.wake.disabled': 'off (phrase “{phrase}” from: {source})',
    'deps.wake.disabled_hint': (
        'set WAKE_ENABLED=true to make the assistant react only after being '
        'called'
    ),
    'deps.wake.openwakeword': 'openWakeWord, models: {models}',
    'deps.wake.whisper_detector': "Whisper detector on the model '{model}'",
    'deps.wake.missing_openwakeword': (
        'WAKE_ENGINE=openwakeword, but the package or the model in {path} is '
        'missing'
    ),
    'deps.wake.missing_hint': 'set WAKE_ENGINE=auto (the Whisper detector works with any phrase)',
    'deps.wake.ok': '“{phrase}” (from: {source}) — {engine}',
    'deps.wake.model_name': 'Wake detector model',
    'deps.wake.model_present': "the model '{model}' is on disk",
    'deps.wake.model_absent': "no model '{model}' — the main model will be used",
    'deps.speaker.name': 'Speaker',
    'deps.speaker.no_packages': 'sounddevice or numpy missing — devices were not checked',
    'deps.speaker.error_hint': 'without a speaker the assistant answers with text',
    'deps.speaker.no_devices': 'the system reports no output device',
    'deps.speaker.no_devices_hint': (
        'plug in speakers or headphones; without them answers stay text'
    ),
    'deps.speaker.no_match_hint': 'fix AUDIO_OUTPUT_DEVICE or leave it empty (default device)',
    'deps.speaker.ok': '{count} output devices, selected: {selected}',
    'deps.tts.name': 'Speech synthesis',
    'deps.tts.disabled_env': 'turned off by TTS_ENABLED=false',
    'deps.tts.disabled_engine': 'turned off by voice_engine: "{engine}"',
    'deps.tts.disabled_hint': (
        'set voice_engine: "piper" in config/user_settings.json and '
        'TTS_ENABLED=true'
    ),
    'deps.tts.engine_name': 'Speech engine (Piper)',
    'deps.tts.package': 'the piper-tts package (synthesis inside the assistant process)',
    'deps.tts.binary': 'the piper program',
    'deps.tts.missing': "neither the 'piper-tts' package nor the 'piper' program",
    'deps.tts.missing_hint': (
        'pip install piper-tts (pulls onnxruntime — no wheels for every Python '
        'version) or point to the binary in .env: PIPER_BINARY=...'
    ),
    'deps.voice.name': 'Piper voice (model)',
    'deps.voice.found': '{count} voices: {voices}',
    'deps.voice.selected': ' — selected: {selected}',
    'deps.voice.not_found': "the voice '{wanted}' was not found; available: {voices}",
    'deps.voice.not_found_hint': 'fix piper_model in config/user_settings.json',
    'deps.voice.no_files': 'no .onnx file found in: {directories}',
    'deps.voice.no_files_hint': (
        'python scripts/prepare_offline.py --piper (needs the internet, once) or '
        'point to your own directory: PIPER_VOICES_DIR in .env'
    ),
    'deps.sem.name': 'Semantic memory',
    'deps.sem.disabled': 'off via EMBEDDINGS_ENABLED/EMBEDDING_ENGINE',
    'deps.sem.disabled_hint': (
        'set EMBEDDINGS_ENABLED=true so the assistant can link memories by '
        'meaning'
    ),
    'deps.sem.st_model': 'sentence-transformers, model {model}',
    'deps.sem.on_disk': ' (on disk)',
    'deps.sem.will_download': ' — it will be downloaded on first use',
    'deps.sem.download_hint': 'get it beforehand: python scripts/prepare_offline.py --embeddings',
    'deps.sem.st_missing': 'EMBEDDING_ENGINE=sentence-transformers, but the package is missing',
    'deps.sem.st_missing_hint': '{install}  or switch to EMBEDDING_ENGINE=ollama',
    'deps.sem.ollama_down': (
        'Ollama would compute the embeddings — the service is not responding '
        'now'
    ),
    'deps.sem.ollama_ok': 'Ollama computes the embeddings, model {model}',
    'deps.sem.ollama_missing_model': (
        "Ollama was to compute the embeddings, but the model '{model}' is "
        "missing"
    ),
    'deps.sem.ollama_hint': 'ollama pull {model}  (or: pip install sentence-transformers)',
    'deps.vector.name': 'Vector index',
    'deps.vector.no_numpy': 'numpy is missing — searching by meaning is off',
    'deps.vector.faiss': 'FAISS (fast search)',
    'deps.vector.numpy': (
        'NumPy — enough for ~10⁵ memories; FAISS is faster (pip install faiss-'
        'cpu)'
    ),
    'deps.index.name': 'Memory index',
    'deps.index.no_database': 'the database does not exist yet — the index appears during the chat',
    'deps.index.empty_db': 'no index in the database — it appears on first use (/memory reindex)',
    'deps.index.empty': 'empty — it builds during the chat or with /memory reindex',
    'deps.index.count': '{count} vectors, model {model}',
    'deps.tools.name': 'Tools (tool calling)',
    'deps.tools.disabled': 'off via TOOLS_ENABLED=false',
    'deps.tools.disabled_hint': 'set TOOLS_ENABLED=true so the model can use tools',
    'deps.tools.registry_failed': 'the tool registry did not build: {error}',
    'deps.tools.registry_hint': 'details in logs/errors.log',
    'deps.tools.visible': '{visible} of {total} available to the model: {names}',
    'deps.tools.hidden': ' (hidden from the model: {hidden})',
    'deps.tools.all_disabled': 'all tools are disabled by configuration',
    'deps.perm.name': 'Tool permissions',
    'deps.perm.terminal': '; confirmations in the terminal',
    'deps.perm.no_terminal': '; no interactive terminal — HIGH/CRITICAL will be refused',
    'deps.workspace.name': 'Tool file area',
    'deps.workspace.detail': '{count} directories: {roots}',
    'deps.workspace.missing': ' (they do not exist yet — created on first write)',
    'deps.workspace.hint': 'add your own directories with FS_ALLOWED_ROOTS in .env',
    'deps.host.name': 'System tools (Phase 8)',
    'deps.host.apps': 'applications: {detail}',
    'deps.host.processes': 'processes: {detail}',
    'deps.host.services': 'services: {detail}',
    'deps.host.shell': 'shell: {detail}',
    'deps.host.pdf': 'PDF: {detail}',
    'deps.host.pdf_missing': 'no library (pip install pypdf)',
    'deps.web.name': 'Web tools',
    'deps.web.hint': 'set OFFLINE_MODE=off and WEB_ENABLED=true to allow network access',
    'deps.web.search': 'search: {provider}',
    'deps.web.weather': 'weather: {provider} (no key)',
    'deps.web.news_feeds': 'news: {count} RSS feeds',
    'deps.web.news_search': 'news: search only',
    'deps.web.youtube_key': 'YouTube key: {state}',
    'install.pip': 'python -m pip install -r {requirements}',
    'install.pip_offline': (
        'python -m pip install --no-index --find-links {wheelhouse} -r '
        '{requirements}'
    ),
    'install.pip_prepare': (
        'on a machine with internet: python scripts/prepare_offline.py --wheels, '
        'then python -m pip install --no-index --find-links {wheelhouse} -r '
        '{requirements}'
    ),
    'install.run_script': 'run {script}',
    'net.http_status': 'the server answered with HTTP {code}',
    'net.no_connection': 'no connection ({reason})',
    'net.timeout': 'timed out ({seconds} s)',
    'net.error': 'network error: {error}',
    'deps.vad.webrtc_missing': 'VAD_ENGINE=webrtc, but the webrtcvad package is not installed',
    'deps.vad.energy': 'energy detector (built in, no extra dependencies)',
    'deps.vad.webrtc': 'webrtcvad, aggressiveness {level}',
    'status.tools_off': 'off (TOOLS_ENABLED=false)',
    'status.tools_visible': '{count} available ({names}); {policy}',
    'status.policy.confirm_from': 'confirmations from {level}',
    'status.policy.critical_allowed': 'CRITICAL allowed',
    'status.policy.critical_blocked': 'CRITICAL blocked',
    'status.policy.limit': 'limit {count}/turn',
    'status.policy.dry_run': 'dry-run mode',
    'status.policy.allowed': 'allow list: {names}',
    'status.policy.disabled': 'disabled: {names}',
    'status.audit.off': 'off (SECURITY_AUDIT_ENABLED=false)',
    'status.audit.empty': 'no calls in this session',
    'status.audit.summary': '{count} calls, {denied} refusals ({where})',
    'status.audit.where_db': 'database + log',
    'status.audit.where_log': 'log',
    'status.audit.confirmed': ' (confirmed)',
    'status.audit.line': '{tool} [{risk}] {decision}{confirmed} → {state} in {ms} ms',
    'status.memory.off': 'off (MEMORY_ENABLED=false)',
    'status.memory.ram_only': 'working memory only — {reason}',
    'status.memory.db_unavailable': 'the database is unavailable',
    'status.memory.no_disk': 'nothing written to disk',
    'status.memory.stats_failed': 'unknown (read error)',
    'status.semantic.off': 'off (EMBEDDINGS_ENABLED=false)',
    'status.semantic.no_db': 'off (no database)',
    'status.db.in_memory': 'in process memory',
    'status.db.describe': (
        '{where} (schema v{version}/{latest}, journal {journal}, search '
        '{search})'
    ),
    'status.stats.describe': (
        '{conversations} chats, {messages} messages, {summaries} summaries, '
        '{facts} facts, {preferences} preferences, {notes} notes, {embeddings} '
        'vectors, {audit} audit entries'
    ),
    'status.semantic.no_engine': 'off (no embedding engine)',
    'status.semantic.unavailable': 'unavailable — {reason}',
    'status.semantic.index_off': 'not started',
    'status.semantic.describe': '{provider}, index {index}: {count} vectors',
    'status.semantic.disabled': 'off',
    "deps.ollama.no_model": "the service is running, the model is missing",
    "deps.ollama.unreachable": "the service is not responding",
    # --- tryb bezobsługowy (--headless) ----------------------------------- #
    "cli.arg.headless": (
        "background service: microphone and speech only, no window and no "
        "keyboard (for systemd --user / Task Scheduler)"
    ),
    "cli.main.headless_conflict": "--headless cannot be combined with --gui or --terminal.",
    "cli.main.headless_needs_voice": (
        "--headless has no input other than the microphone, so --no-voice makes no sense."
    ),
    "cli.headless.starting": "{name} is starting as a background service (no window, no keyboard).",
    "cli.headless.no_microphone": (
        "The service has no voice input, and that is its only way of receiving a command."
    ),
    "cli.headless.no_microphone_hint": (
        "check the microphone: python main.py --audio-check — and on Linux make sure the "
        "service can reach the sound session (PipeWire/PulseAudio in the user session)"
    ),
    "cli.headless.no_speech": "No speech output ({detail}) — replies go to the log only.",
    "cli.headless.confirmations_denied": (
        "No one can be asked for confirmation here, so HIGH and CRITICAL actions are "
        "refused automatically."
    ),
    "cli.headless.deny_reason": "background service — no channel to confirm the action",
    "cli.headless.microphone_back": "Voice input is back.",
    "cli.headless.stopped": "Background service stopped.",
}


# --------------------------------------------------------------------------- #
# Katalog polski
# --------------------------------------------------------------------------- #

_PL: Final[dict[str, str]] = {
    "common.none": "brak",
    "common.unknown": "nieznane",
    "common.off": "wyłączone",
    "common.on": "włączone",
    "common.yes": "tak",
    "common.no": "nie",
    "common.missing": "brak",
    "common.available": "dostępne",
    "common.disabled": "wyłączone",
    "common.enabled": "włączone",
    "common.checking": "sprawdzam...",
    "common.not_checked": "nie sprawdzono",
    "common.dash": "—",
    "gui.subtitle": "asystent lokalny",
    "gui.window_title": "{name} — asystent lokalny {version}",
    "gui.speech": "Mowa",
    "gui.speech_missing": "Mowa (brak)",
    "gui.new_conversation": "Nowa rozmowa",
    'ollama.remote': 'OLLAMA_HOST wskazuje inną maszynę ({host}) — niczego tam nie uruchamiam.',
    'ollama.missing_binary': 'Na tej maszynie nie ma Ollamy (brak `ollama` w PATH).',
    'ollama.install_hint': (
        'zainstaluj: https://ollama.com/download — albo wskaż inną maszynę '
        'w OLLAMA_HOST'
    ),
    'ollama.start_failed': 'Nie udało się uruchomić `ollama serve`.',
    'ollama.started': 'Ollama nie działała — uruchomiłam ją w tle.',
    'ollama.service_hint': 'żeby działała zawsze: systemctl enable --now ollama',
    'ollama.exited': '`ollama serve` zakończył się od razu (kod {code}) — port może być zajęty.',
    'ollama.timeout': '`ollama serve` nie odpowiedział w ciągu {seconds} s.',
    'ollama.log_hint': 'szczegóły: {path}',
    'cli.gui_fallback': 'Interfejs graficzny jest niedostępny ({reason}) — startuję w terminalu.',
    'deps.compute.name': 'Obliczenia Ollamy',
    'deps.compute.gpu': 'GPU — {size} modelu w VRAM',
    'deps.compute.cpu_with_gpu': (
        'CPU, mimo obecnego GPU z CUDA ({gpu}) — odpowiedzi są kilka razy '
        'wolniejsze'
    ),
    'deps.compute.cpu': 'CPU (nie wykryto GPU z CUDA)',
    'deps.compute.unknown': 'nie sprawdzono — żaden model nie jest jeszcze załadowany',
    'deps.compute.hint_pacman': (
        'sudo pacman -S ollama-cuda   (pakiet `ollama` to wariant tylko na '
        'CPU)'
    ),
    'deps.compute.hint_generic': 'zainstaluj wariant Ollamy z CUDA: https://ollama.com/download',
    'runtime.wake_ignored': (
        'Usłyszałam mowę bez frazy „{phrase}” — naciśnij Słuchaj albo '
        'powiedz frazę.'
    ),
    'deps.whisper.cuda_missing': 'CUDA jest nieużywalna (brak {library}) — Whisper liczy na CPU',
    'deps.whisper.cuda_hint_pacman': (
        'sudo pacman -S cuda cudnn   (albo ustaw WHISPER_DEVICE=cpu, żeby '
        'nie próbował)'
    ),
    'deps.whisper.cuda_hint_generic': (
        'zainstaluj środowisko CUDA z cuBLAS i cuDNN albo ustaw '
        'WHISPER_DEVICE=cpu'
    ),
    "gui.settings": "Ustawienia",
    "gui.new_short": "Nowa",
    "gui.settings_short": "Ustaw.",
    "gui.status_show": "Stan",
    "gui.status_hide": "Ukryj stan",
    "gui.listen": "Słuchaj",
    "gui.stop_listening": "Nie słuchaj",
    "gui.send": "Wyślij",
    "gui.interrupt": "Przerwij",
    "gui.input_placeholder": "Napisz wiadomość albo naciśnij „Słuchaj”...",
    "gui.new_conversation_notice": (
        "Nowa rozmowa. Okno wyczyszczone; fakty i notatki zostają w pamięci."
    ),
    "gui.closing": "Zamykam...",
    "gui.session_model": (
        "Model na tę sesję: {model}. Na stałe ustawia go OLLAMA_MODEL w pliku .env."
    ),
    "gui.role.user": "Ty",
    "gui.role.system": "System",
    "gui.role.tool": "Narzędzie",
    "gui.role.error": "Błąd",
    "gui.role.assistant_fallback": "Asystent",
    "gui.detail.microphone": "mikrofon",
    "gui.detail.voice_sample": "próbka głosu",
    "gui.status.title": "Stan — {name}",
    "gui.status.model": "Model: {model}",
    "gui.status.language": "Język odpowiedzi: {language}",
    "gui.status.language_auto": "auto (rozpoznawany przy każdej wypowiedzi)",
    "gui.status.language_forced": "{code} (ustawiony)",
    "gui.status.no_model": "brak",
    "service.mic": "Mikrofon",
    "service.wake": "Słowo aktywujące",
    "service.whisper": "Whisper",
    "service.ollama": "Ollama",
    "service.speech": "Mowa",
    "service.memory": "Pamięć",
    "service.tools": "Narzędzia",
    "listening.off": "mikrofon wyłączony",
    "listening.idle": "gotowa",
    "listening.waiting_wake": "czekam na „{phrase}”",
    "listening.waiting_wake_generic": "czekam na frazę",
    "listening.listening": "słucham...",
    "listening.transcribing": "rozpoznaję mowę...",
    "listening.thinking": "myślę...",
    "listening.speaking": "mówię...",
    "settings.title": "Ustawienia",
    "settings.back": "Wróć do rozmowy",
    "settings.save": "Zapisz",
    "settings.revert": "Przywróć",
    "settings.browse": "Przeglądaj plik...",
    "settings.clear": "Wyczyść",
    "settings.listen_sample": "Posłuchaj",
    "settings.section.assistant": "Asystent",
    "settings.section.voice": "Mowa",
    "settings.section.rvc": "Konwersja głosu (RVC)",
    "settings.reverted": "Przywrócono wartości z pliku ustawień.",
    "settings.needs_reload_suffix": "  (wymaga przeładowania mowy)",
    "settings.dialog_title": "Wybierz plik: {label}",
    "settings.dialog_unavailable": (
        "Systemowe okno wyboru pliku jest niedostępne — wpisz ścieżkę ręcznie."
    ),
    "settings.auto_voice": "(dobierany do języka)",
    "settings.field.assistant_name": "Imię asystenta",
    "settings.help.assistant_name": (
        "Tytuł okna, nagłówki i domyślna fraza wybudzająca („hej <imię>”)."
    ),
    "settings.placeholder.assistant_name": "np. Miku",
    "settings.field.ui_accent_color": "Kolor akcentu",
    "settings.help.ui_accent_color": (
        "Z tego jednego koloru liczony jest cały motyw: tła, bąbelki, wskaźnik nasłuchu."
    ),
    "settings.field.personality_traits": "Cechy charakteru",
    "settings.help.personality_traits": (
        "Dopisywane do promptu jako STYL wypowiedzi. Nie zmieniają zasad ani uprawnień."
    ),
    "settings.placeholder.personality_traits": "np. mówi krótko, lubi porównania do kolarstwa",
    "settings.field.voice_engine": "Silnik mowy",
    "settings.help.voice_engine": (
        "„none” wyłącza mówienie. Zmiana wymaga ponownego załadowania mowy."
    ),
    "settings.field.piper_model": "Głos Piper",
    "settings.help.piper_model": (
        "Puste = głos dobierany do języka odpowiedzi. Zmiana wymaga przeładowania mowy."
    ),
    "settings.field.speech_language": "Język, w którym mówisz",
    "settings.help.speech_language": (
        "Język TWOJEJ mowy — nie musi być tym samym co język odpowiedzi. "
        "Jeden kod („pl”) wymusza go na stałe. Mówisz w dwóch? Wypisz oba: „pl,en” — "
        "wtedy język jest rozpoznawany, ale wybierany tylko spośród Twoich. "
        "„auto” idzie za językiem asystenta, „detect” dopuszcza dowolny język."
    ),
    "settings.placeholder.speech_language": "pl,en — albo auto, detect…",
    "settings.field.rvc_enabled": "Konwersja głosu RVC",
    "settings.help.rvc_enabled": (
        "Bez wskazanego modelu pozostaje wyłączona, nawet gdy przełącznik jest włączony."
    ),
    "settings.field.rvc_model_path": "Model RVC (.pth)",
    "settings.help.rvc_model_path": (
        "Wybierz plik przyciskiem — ścieżki wpisywane ręcznie to najczęstsze źródło literówek."
    ),
    "settings.field.rvc_index_path": "Indeks RVC (.index)",
    "settings.help.rvc_index_path": (
        "Opcjonalny. Poprawia barwę głosu, ale bez niego konwersja też działa."
    ),
    "settings.field.rvc_pitch_shift": "Zmiana wysokości (półtony)",
    "settings.help.rvc_pitch_shift": "0 = bez zmiany. +12 to oktawa w górę.",
    "settings.field.rvc_index_rate": "Udział indeksu",
    "settings.help.rvc_index_rate": "0 = sam model, 1 = maksymalny wpływ indeksu.",
    "settings.filter.rvc_model": "Model RVC",
    "settings.filter.rvc_index": "Indeks RVC",
    "settings.filter.all_files": "Wszystkie pliki",
    "settings.result.nothing": "Nic nie zmieniono.",
    "settings.result.not_saved": "Nie zapisano ustawień.",
    "settings.result.saved_reload": (
        "Zapisano: {fields}. Nowe ustawienia mowy zaczną obowiązywać po "
        "przeładowaniu silnika mowy — zrobię to teraz."
    ),
    "settings.result.saved": "Zapisano: {fields}. Zmiany działają od razu.",
    "settings.problem.color": "Kolor musi być zapisem szesnastkowym, np. #39C5BB albo #3CB.",
    "settings.problem.missing_file": (
        "Nie znalazłam pliku: {path}. Zapiszę ścieżkę, ale mowa jej nie użyje."
    ),
    "settings.problem.wrong_suffix": (
        "Plik nie ma rozszerzenia {suffix} — sprawdź, czy to właściwy plik."
    ),
    "settings.problem.rvc_without_model": (
        "RVC jest włączone, ale nie wskazano modelu — konwersja pozostanie wyłączona."
    ),
    "settings.problem.write_failed": "Błąd zapisu: {error}",
    "gui.confirm.allow": "Zezwól",
    "gui.confirm.cancel": "Anuluj",
    "runtime.memory_unavailable": (
        "Pamięć długoterminowa jest niedostępna ({reason}). Rozmowa działa, ale nic "
        "nie przetrwa zamknięcia okna."
    ),
    "runtime.tools_unavailable": "Narzędzia niedostępne ({reason}) — zostaje sama rozmowa.",
    "runtime.tools_failed": "Nie udało się przygotować narzędzi: {error}",
    "runtime.memory_interrupted": "Przerwano zapisywanie do pamięci.",
    "runtime.memory_failed": "Nie udało się obsłużyć polecenia pamięci: {error}",
    "runtime.compacting": "Rozmowa się wydłużyła — streszczam starsze wątki...",
    "runtime.generation_interrupted": "Przerwano generowanie odpowiedzi.",
    "runtime.empty_reply": "Model nie zwrócił żadnej treści.",
    "runtime.unexpected_error": "Nieoczekiwany błąd: {error}",
    "runtime.command_failed": "Nie udało się wykonać polecenia: {error}",
    "runtime.speech_reloaded": "Mowa przeładowana: {detail}",
    "runtime.speech_still_off": "Mowa pozostaje wyłączona ({detail}).",
    "runtime.speech_no_sample": "Nie mam czym powiedzieć próbki ({detail}).",
    "runtime.voice_sample": "Cześć, jestem {name}. Tak brzmię na tym komputerze.",
    "runtime.speech_error_hint": (
        "Odpowiedzi zostają tekstowe — mowę włączysz ponownie przełącznikiem."
    ),
    "runtime.mic_unavailable": "Tryb głosowy niedostępny: {reason}",
    "runtime.mic_unavailable_hint": "Pisanie działa normalnie.",
    "runtime.mic_failed": "Nie udało się uruchomić mikrofonu: {error}",
    "runtime.mic_failed_hint": "Szczegóły zapisano w logach.",
    "runtime.mic_error": "Błąd wejścia głosowego: {error}",
    "runtime.mic_error_hint": "Wyłączam mikrofon, pisanie działa dalej.",
    "runtime.stopped_working": "Asystent przestał działać: {error}",
    "runtime.ollama_no_models": "Ollama nie odpowiedziała ({error})",
    "runtime.thinking": "model analizuje pytanie...",
    "runtime.confirm_window_closed": "okno zostało zamknięte",
    "runtime.user_denied": "użytkownik odmówił",
    "runtime.mic_dummy": "mikrofon wyłączony",
    "runtime.wake_disabled": "wyłączone (fraza „{phrase}”)",
    "runtime.wake_pending": "„{phrase}” — silnik ustali się przy starcie nasłuchu",
    "runtime.wake_no_gate": "nasłuch bez bramki (fraza „{phrase}”)",
    "runtime.wake_active": "„{phrase}” ({engine})",
    "runtime.speech_muted": "wyciszona",
    "runtime.speech_off": "wyłączona",
    "runtime.tools_off": "wyłączone",
    "gui.unavailable": "Interfejs graficzny jest niedostępny: {reason}",
    "gui.terminal_hint": "Tryb terminalowy działa bez GUI: python main.py --terminal",
    "gui.window_failed": "Nie udało się otworzyć okna: {error}",
    "gui.no_display_hint": (
        "Zwykle znaczy to brak dostępu do sesji graficznej. Tryb terminalowy: "
        "python main.py --terminal"
    ),
    "gui.crashed": "Interfejs graficzny zakończył się błędem: {error}",
    "gui.missing_tk": "brak biblioteki Tk dla tego Pythona ({error})",
    "gui.missing_toolkit": "brak pakietu {package} ({error})",
    "gui.no_session": "nie wykryto sesji graficznej (brak DISPLAY i WAYLAND_DISPLAY)",
    "gui.ready": "CustomTkinter gotowy",
    "gui.hint.tk_generic": (
        "zainstaluj bibliotekę Tk dla swojego Pythona (pakiet systemowy, nie pip)"
    ),
    "gui.hint.tk_windows": (
        "zainstaluj Pythona z opcją „tcl/tk and IDLE” "
        "(instalator python.org → Modify → Optional Features)"
    ),
    "deps.tk.name": "Tk (biblioteka systemowa)",
    "deps.tk.ok": "dostępna",
    "deps.tk.missing": "brak — GUI nie wystartuje, tryb terminalowy działa normalnie",
    "deps.gui.name": "GUI (CustomTkinter)",
    "deps.gui.ok": "dostępny",
    "deps.gui.needs_tk": "pakiet jest, ale nie da się go użyć bez biblioteki Tk",
    "deps.gui.missing": "brak pakietu",
    "deps.gui.purpose": "interfejs graficzny (python main.py --gui)",
    "deps.display.name": "Sesja graficzna",
    "deps.display.ok": "wykryta",
    "deps.display.missing": (
        "brak (serwer bez pulpitu, SSH bez X11, usługa) — użyj --terminal"
    ),
    "deps.display.hint": "python main.py --terminal",
    'report.system': 'System:      {label} ({machine})',
    'report.wsl': '             (wykryto WSL)',
    'report.python': 'Python:      {version} — {executable}',
    'report.package_manager': 'Menedżer pakietów: {manager} → skrypt instalacyjny: {script}',
    'report.dependencies': 'Zależności:',
    'report.ok': 'ok  ',
    'report.missing': 'BRAK',
    'report.required': 'wymagane',
    'report.optional': 'opcjonalne',
    'report.path': 'ścieżka: {path}',
    'report.missing_required': 'Brakuje {count} wymaganych elementów:',
    'report.nothing_automatic': (
        'Nic nie jest instalowane automatycznie. Aby dokończyć instalację: '
        '{command}'
    ),
    'report.all_present': 'Wszystkie wymagane zależności są dostępne.',
    'report.optional_missing': 'Elementy opcjonalne, których nie wykryto:',
    'cli.header.title': ' {name} — asystent lokalny, wersja {version}',
    'cli.header.model': ' Model:    {model} @ {host}',
    'cli.header.mode': ' Tryb:     {mode}',
    'cli.header.system': ' System:   {label} ({machine})',
    'cli.header.gpu': ' GPU:      {detail}',
    'cli.header.logs': ' Logi:     {path}',
    'cli.header.commands': (
        'Komendy: /pomoc, /status, /mic, /wake, /glos, /pamiec, /nowa, /reload, '
        '/deps, /wyjscie'
    ),
    'cli.header.mic_found': 'Mikrofon wykryty — /mic przełącza na mówienie zamiast pisania.',
    'cli.header.wake': 'Słowo aktywujące: „{phrase}” — zmienisz je komendą /wake fraza <tekst>.',
    'cli.header.no_voice': 'Tryb głosowy niedostępny ({detail}).',
    'cli.help.title': 'Dostępne komendy:',
    'cli.help.body': (
        '        /pomoc      — ta lista\n        /status     — model, historia '
        'rozmowy, ustawienia użytkownika\n        /mic        — włącz/wyłącz '
        'mówienie zamiast pisania\n        /mic lista  — pokaż wykryte '
        'mikrofony\n        /wake       — stan słowa aktywującego\n        /wake '
        'fraza <tekst> — zmień frazę (zapis w user_settings.json)\n        /wake '
        'teraz — otwórz okno rozmowy bez wypowiadania frazy\n        /wake off   '
        '— nasłuch bez bramki do końca sesji\n        /glos       — włącz/wycisz '
        'mówienie odpowiedzi\n        /glos lista — pokaż znalezione głosy '
        'Pipera\n        /glos test  — powiedz zdanie próbne\n        /glos model'
        ' <nazwa> — zmień głos (zapis w user_settings.json)\n        /glos zapisz'
        ' <plik.wav> — zapisz próbkę głosu do pliku\n        /pamiec     — co '
        'asystent pamięta (baza, fakty, streszczenia)\n        /pamiec fakty'
        '              — lista zapamiętanych faktów\n        /pamiec zapamietaj '
        'k=w     — zapamiętaj fakt na stałe\n        /pamiec zapomnij <klucz>   —'
        ' usuń fakt\n        /pamiec notatka <tekst>    — zapisz notatkę\n'
        '        /pamiec szukaj <fraza>     — przeszukaj rozmowy i notatki (po '
        'słowach)\n        /pamiec przypomnij <fraza> — znajdź wspomnienia po '
        'ZNACZENIU\n        /pamiec reindeks           — przelicz embeddingi '
        'całej pamięci\n        Zapamiętaj, że ...         — zapisz coś na stałe '
        '(bez ukośnika)\n        Zapomnij, że ...           — usuń to z pamięci\n'
        '        /narzedzia  — narzędzia, ich ryzyko i ostatnie wywołania\n'
        '        /nowa       — zacznij nową rozmowę (fakty i notatki zostają)\n'
        '        /reload     — wczytaj ponownie config/user_settings.json\n'
        '        /deps       — sprawdź ponownie zależności\n        /wyjscie    —'
        ' zakończ (albo Ctrl+D)'
    ),
    'cli.status.model': 'Model:      {model} @ {host}',
    'cli.status.mode': 'Tryb:       {mode}',
    'cli.status.history': (
        'Historia:   {messages}/{max_messages} wiadomości, {chars}/{max_chars} '
        'znaków'
    ),
    'cli.status.memory': 'Pamięć:     {detail}',
    'cli.status.saved': 'Zapisano:   {detail}',
    'cli.status.semantic': 'Skojarz.:   {detail}',
    'cli.status.summary': 'Streszcz.:  {detail}',
    'cli.status.tools': 'Narzędzia:  {detail}',
    'cli.status.audit': 'Audyt:      {detail}',
    'cli.status.mic': 'Mikrofon:   {detail}',
    'cli.status.wake': 'Wake word:  {detail}',
    'cli.status.speech_language': 'Język mowy: {detail}',
    'cli.status.language_auto': 'auto (rozpoznawany przy każdej wypowiedzi)',
    'cli.status.language_forced': '{code} (wymuszony w rozpoznawaniu i odpowiedziach)',
    'cli.status.assistant': 'Asystent:   {name} (tag {tag})',
    'cli.status.accent': 'Kolor GUI:  {color}',
    'cli.status.speech': 'Mowa:       {detail}',
    'cli.status.speech_not_started': 'nie uruchomiono',
    'cli.status.engine': (
        'Silnik:     {engine}, model: {voice}, tempo {speed}x, głośność {volume} '
        '/ RVC: {rvc}'
    ),
    'cli.status.traits': 'Cechy:      {detail}',
    'cli.status.tools_off': 'wyłączone (TOOLS_ENABLED=false)',
    'cli.status.auto_voice': '(dobierany do języka)',
    'cli.status.none': '(brak)',
    'cli.voice.mic_on': 'włączony ({detail})',
    'cli.voice.mic_unavailable': 'niedostępny — {reason}',
    'cli.voice.mic_off': 'wyłączony',
    'cli.voice.wake_source_file': 'wake_word w user_settings.json',
    'cli.voice.wake_source_name': 'imię asystenta',
    'cli.voice.wake_off': 'wyłączona (fraza „{phrase}”, źródło: {source})',
    'cli.voice.wake_pending': (
        '„{phrase}” (źródło: {source}), silnik ustali się przy starcie '
        'nasłuchu'
    ),
    'cli.voice.wake_no_gate': 'niedostępna — nasłuch bez bramki (fraza „{phrase}”)',
    'cli.voice.wake_awake': 'okno rozmowy otwarte',
    'cli.voice.wake_waiting': 'czeka na frazę',
    'cli.voice.wake_active': '„{phrase}” (źródło: {source}), {engine}, {state}',
    'cli.voice.give_phrase': 'Podaj frazę, np. /wake fraza hej aiko',
    'cli.voice.new_phrase': 'Nowa fraza: „{phrase}” (zapisano w config/user_settings.json)',
    'cli.voice.gate_off': 'Bramka frazy wyłączona do końca sesji — mów bez zawołania.',
    'cli.voice.gate_on': 'Bramka frazy włączona.',
    'cli.voice.no_voice_mode': 'Tryb głosowy nie działa — najpierw /mic.',
    'cli.voice.window_open': 'Okno rozmowy otwarte bez zawołania.',
    'cli.voice.wake_status': 'Słowo aktywujące: {detail}',
    'cli.voice.wake_usage': 'Użycie: /wake [on|off|teraz|fraza <tekst>]',
    'cli.voice.hint': '       Podpowiedź: {detail}',
    'cli.voice.mode_unavailable': 'Tryb głosowy niedostępny: {reason}',
    'cli.voice.install': '        Zainstaluj zależności: {hint}',
    'cli.voice.preparing': 'przygotowuję wejście głosowe (pierwsze uruchomienie ładuje model)...',
    'cli.voice.staying_text': 'Zostaję w trybie tekstowym.',
    'cli.voice.start_failed': 'Nie udało się uruchomić trybu głosowego: {error}',
    'cli.voice.details_in': 'Szczegóły zapisano w {path}',
    'cli.voice.listen_interrupted': (
        'Przerwano nasłuch — tryb głosowy wyłączony (/mic włącza ponownie).'
    ),
    'cli.voice.listen_error': 'Błąd wejścia głosowego: {error}',
    'cli.voice.listen_disabled': 'Wyłączam tryb głosowy, czat tekstowy działa dalej.',
    'cli.voice.nothing_heard': 'Nic nie usłyszałam — możesz wpisać tekst.',
    'cli.voice.no_audio_packages': 'Brak pakietów audio ({error}).',
    'cli.voice.no_input_devices': 'System nie zgłasza żadnego urządzenia wejściowego.',
    'cli.voice.devices': 'Dostępne mikrofony (fragment nazwy → AUDIO_INPUT_DEVICE):',
    'cli.speech.muted': 'wyciszona na czas sesji (/glos on włącza)',
    'cli.speech.on': 'włączona ({detail})',
    'cli.speech.unavailable': 'niedostępna — {reason}',
    'cli.speech.off': 'wyłączona',
    'cli.speech.speech_unavailable': 'Mowa niedostępna: {reason}',
    'cli.speech.disabled_in_settings': 'Mowa wyłączona w ustawieniach ({detail}).',
    'cli.speech.text_only': 'Odpowiedzi zostają tekstowe.',
    'cli.speech.start_failed': 'Nie udało się uruchomić mowy: {error}',
    'cli.speech.no_voices': 'Nie znalazłam żadnego głosu (.onnx). Szukałam w:',
    'cli.speech.download_voice': 'Pobierz głos: python scripts/prepare_offline.py --piper',
    'cli.speech.voices': 'Dostępne głosy (nazwa → piper_model w user_settings.json):',
    'cli.speech.selected_marker': '  <- wybrany',
    'cli.speech.auto_voice_note': 'piper_model jest puste — głos dobiera się do języka odpowiedzi.',
    'cli.speech.give_voice': 'Podaj nazwę głosu, np. /glos model pl_PL-darkman-medium',
    'cli.speech.voice_not_found': 'Nie znalazłam głosu „{name}”.',
    'cli.speech.list_hint': 'Listę pokazuje: /glos lista',
    'cli.speech.new_voice': 'Nowy głos: {detail} (zapisano w config/user_settings.json)',
    'cli.speech.saying': 'Mówię: {text}',
    'cli.speech.sample': 'Cześć, jestem {name}. Tak brzmię na tym komputerze.',
    'cli.speech.give_file': 'Podaj nazwę pliku, np. /glos zapisz proba.wav',
    'cli.speech.sample_saved': 'Zapisano próbkę głosu: {path}',
    'cli.speech.muted_now': 'Mowa wyciszona.',
    'cli.speech.already_off': 'Mowa już była wyłączona.',
    'cli.speech.already_on': 'Mowa już działa.',
    'cli.speech.state': 'Mowa: {detail}',
    'cli.speech.usage': 'Użycie: /glos [on|off|lista|test|model <nazwa>|zapisz <plik>]',
    'cli.audio.device': 'Urządzenie: {detail}',
    'cli.audio.system_default': 'domyślne systemowe',
    'cli.audio.measuring': 'Mierzę tło przez {seconds} s — NIE mów w tym czasie...',
    'cli.audio.no_samples': (
        'Nie odebrano żadnych próbek — sprawdź, czy mikrofon nie jest '
        'wyciszony.'
    ),
    'cli.audio.frames': 'Ramek: {frames}, odrzuconych: {dropped}',
    'cli.audio.levels': (
        'Poziom tła: 10. percentyl {quiet} dBFS, mediana {median} dBFS, szczyt '
        '{peak} dBFS'
    ),
    'cli.audio.thresholds': 'Ile ramek tła zostałoby uznane za mowę przy różnych progach:',
    'cli.audio.suggested': '  <- proponowany',
    'cli.audio.frames_share': '        VAD_ENERGY_THRESHOLD_DB={threshold} {share}% ramek{marker}',
    'cli.audio.active_vad': 'Aktywny silnik VAD: {name}',
    'cli.audio.webrtc_note': 'Używasz webrtcvad — próg energetyczny nie jest w ogóle stosowany.',
    'cli.audio.write_env': 'Wpisz do .env: VAD_ENERGY_THRESHOLD_DB={value}',
    'cli.audio.better': (
        'Jeszcze lepiej: pip install webrtcvad-wheels (model mowy zamiast progu '
        'energii)'
    ),
    'cli.audio.too_loud': 'Tło jest bardzo głośne — rozważ inny mikrofon albo niższy gain.',
    'cli.audio.recommended': 'Zalecane: pip install webrtcvad-wheels',
    'cli.mic.off': 'Tryb głosowy wyłączony.',
    'cli.mic.already_off': 'Tryb głosowy już był wyłączony.',
    'cli.mic.already_on': 'Tryb głosowy już działa.',
    'cli.mic.on': 'Tryb głosowy włączony — mów po komunikacie [MIC] słucham.',
    'cli.mic.usage': 'Użycie: /mic [on|off|lista]',
    'cli.mem.state': 'Pamięć:     {detail}',
    'cli.mem.saved': 'Zapisano:   {detail}',
    'cli.mem.window': 'Okno:       {messages} wiadomości, {pending} czeka na streszczenie',
    'cli.mem.semantic': 'Skojarz.:   {detail}',
    'cli.mem.summary': 'Streszcz.:  {detail}',
    'cli.mem.hint': 'Podpowiedź: /pomoc → sekcja pamięci; szczegóły: /deps',
    'cli.mem.unavailable': 'Pamięć trwała jest niedostępna ({detail}).',
    'cli.mem.no_facts': 'Nie zapamiętałam jeszcze żadnych faktów.',
    'cli.mem.add_fact_hint': 'Dodasz je komendą: /pamiec zapamietaj imie=Mariusz',
    'cli.mem.facts': 'Zapamiętane fakty ({count}):',
    'cli.mem.fact_line': '        - {key}: {value}   (źródło: {source})',
    'cli.mem.preference_line': '        ~ {key}: {value}   (preferencja)',
    'cli.mem.remember_usage': 'Użycie: /pamiec zapamietaj klucz=wartość',
    'cli.mem.remembered': 'Zapamiętane: {key} = {value}',
    'cli.mem.remember_failed': 'Nie udało się zapisać faktu ({error}).',
    'cli.mem.forget_usage': 'Użycie: /pamiec zapomnij <klucz>',
    'cli.mem.forgotten': 'Zapomniane: {key}',
    'cli.mem.no_such_fact': 'Nie mam takiego faktu: {key}',
    'cli.mem.note_usage': 'Użycie: /pamiec notatka <treść>',
    'cli.mem.note_failed': 'Nie udało się zapisać notatki ({error}).',
    'cli.mem.note_saved': 'Notatka zapisana (#{id}).',
    'cli.mem.no_notes': 'Brak notatek. Dodasz je: /pamiec notatka <treść>',
    'cli.mem.notes': 'Notatki ({count}):',
    'cli.mem.recall_usage': 'Użycie: /pamiec przypomnij <fraza>',
    'cli.mem.semantic_unavailable': 'Pamięć semantyczna jest niedostępna ({detail}).',
    'cli.mem.word_search_works': 'Szukanie po słowach nadal działa: /pamiec szukaj {phrase}',
    'cli.mem.nothing_recalled': 'Nic mi się nie kojarzy z: {phrase}',
    'cli.mem.recalled': 'Skojarzenia ({count}):',
    'cli.mem.reindexing': 'Liczę embeddingi dla całej pamięci — to może chwilę potrwać...',
    'cli.mem.reindex_done': 'Gotowe: {count} wektorów. {detail}',
    'cli.mem.search_usage': 'Użycie: /pamiec szukaj <fraza>',
    'cli.mem.nothing_found': 'Nic nie znalazłam dla: {phrase}',
    'cli.mem.found': 'Znalezione ({count}):',
    'cli.mem.note_kind': 'notatka',
    'cli.mem.chat_kind': 'rozmowa',
    'cli.mem.unknown_command': 'Nie znam takiego polecenia pamięci: {command}',
    'cli.mem.available_commands': (
        'Dostępne: stan, fakty, zapamietaj klucz=wartość, zapomnij <klucz>, '
        'notatka <tekst>, notatki, szukaj <fraza>, przypomnij <fraza>, reindeks'
    ),
    'cli.mem.save_interrupted': 'Przerwano zapisywanie do pamięci.',
    'cli.mem.handle_failed': 'Nie udało się obsłużyć polecenia pamięci: {error}',
    'cli.tools.unavailable': 'Narzędzia niedostępne ({error}) — zostaje sama rozmowa.',
    'cli.tools.failed': 'Nie udało się przygotować narzędzi: {error}',
    'cli.tools.disabled': 'Narzędzia są wyłączone (TOOLS_ENABLED=false).',
    'cli.plugins.list': 'Pluginy: {detail}',
    'plugins.reminders.notice': 'Przypomnienie: {text}',
    'cli.tools.list': 'Narzędzia: {detail}',
    'cli.tools.files': 'Pliki:      {detail}',
    'cli.tools.audit': 'Audyt wywołań: {detail}',
    'cli.tools.no_confirm_channel': (
        'Brak interaktywnego terminala — narzędzia wymagające zgody będą '
        'odrzucane.'
    ),
    'cli.reindex.memory_unavailable': 'Pamięć trwała jest niedostępna: {error}',
    'cli.reindex.semantic_unavailable': 'Pamięć semantyczna jest niedostępna: {detail}',
    'cli.reindex.details': 'Szczegóły: python main.py --check-deps',
    'cli.reindex.computing': 'Liczę embeddingi — pierwsze uruchomienie ładuje model...',
    'cli.reindex.done': 'Gotowe: {count} wektorów w {seconds} s.',
    'cli.run.ollama_down': (
        'Ollama nie odpowiada pod {host} — rozmowa nie zadziała, dopóki usługa '
        'nie wystartuje.'
    ),
    'cli.run.ollama_hint': 'Po uruchomieniu `ollama serve` wpisz /deps, żeby sprawdzić ponownie.',
    'cli.run.model_missing': (
        "Model '{model}' nie jest pobrany — pobierz go poleceniem: ollama pull "
        "{model}"
    ),
    'cli.run.memory_unavailable': 'Pamięć długoterminowa niedostępna: {error}',
    'cli.run.memory_note': 'Rozmowa działa normalnie, ale nic nie przetrwa zamknięcia programu.',
    'cli.run.voice_on': 'Tryb głosowy włączony — /mic wyłącza go w każdej chwili.',
    'cli.run.typing': 'Pozostaję przy wpisywaniu tekstu.',
    'cli.run.speech_state': 'Mowa {detail}. /glos wycisza.',
    'cli.run.speech_unavailable': 'Mowa {detail} — odpowiedzi będą tekstowe.',
    'cli.run.speech_hint': 'Szczegóły: /deps (pozycje Fazy 4), włączenie: /glos on',
    'cli.run.new_chat': 'Nowa rozmowa. Okno wyczyszczone; {detail}',
    'cli.run.facts_stay': 'fakty i notatki zostają w pamięci.',
    'cli.run.memory_off': 'pamięć trwała jest wyłączona.',
    'cli.run.reloaded': 'Przeładowano ustawienia użytkownika: asystent {name}, tag {tag}.',
    'cli.run.compacting': 'Rozmowa się wydłużyła — streszczam starsze wątki...',
    'cli.run.compact_interrupted': 'Przerwano streszczanie.',
    'cli.run.generation_interrupted': 'Przerwano generowanie odpowiedzi.',
    'cli.run.unexpected': 'Nieoczekiwany błąd: {error}',
    'cli.run.empty_reply': 'Model nie zwrócił żadnej treści.',
    'cli.run.goodbye': 'Do zobaczenia!',
    'cli.main.flags_conflict': '--offline i --online wykluczają się wzajemnie.',
    'cli.main.gui_conflict': '--gui i --no-gui wykluczają się wzajemnie.',
    'cli.main.logging_failed': 'Nie udało się skonfigurować logowania: {error}',
    'cli.main.deps_failed': 'Nie udało się sprawdzić zależności: {error}',
    'cli.main.speech_report': 'Pełny raport mowy: python main.py --check-deps',
    'cli.main.missing_required': 'Brakuje wymaganych elementów środowiska:',
    'cli.main.nothing_automatic': (
        'Nie instaluję niczego automatycznie. Aby dokończyć instalację: '
        '{command}'
    ),
    'cli.main.full_report': 'Pełny raport: python main.py --check-deps',
    'cli.main.starting_anyway': 'Startuję z tym, co jest dostępne...',
    'cli.arg.description': 'Lokalny asystent — tryb terminalowy i diagnostyka środowiska.',
    'cli.arg.terminal': 'tryb rozmowy w terminalu (domyślny, gdy nie podano innej opcji)',
    'cli.arg.gui': 'okno graficzne zamiast terminala (wymaga CustomTkintera i Tk)',
    'cli.arg.no_gui': 'wymuś terminal, nawet gdy GUI_ENABLED=true',
    'cli.arg.check_deps': 'sprawdź zależności, wypisz raport i zakończ',
    'cli.arg.audio_check': 'zmierz tło akustyczne i zaproponuj próg VAD dla tego mikrofonu',
    'cli.arg.voice': 'startuj z włączonym wejściem głosowym (równoważne INPUT_MODE=voice)',
    'cli.arg.no_voice': 'wymuś tryb tekstowy, nawet gdy INPUT_MODE=voice',
    'cli.arg.no_wake': 'nasłuch bez słowa aktywującego (równoważne WAKE_ENABLED=false)',
    'cli.arg.no_tts': 'nie mów odpowiedzi na głos (równoważne TTS_ENABLED=false)',
    'cli.arg.voice_test': 'powiedz zdanie próbne wybranym głosem i zakończ (test mowy)',
    'cli.arg.list_voices': 'wypisz znalezione głosy Pipera i zakończ',
    'cli.arg.no_memory': (
        'nie zapisuj niczego na dysk — rozmowa tylko w pamięci '
        '(MEMORY_ENABLED=false)'
    ),
    'cli.arg.no_embeddings': (
        'bez pamięci semantycznej — nie licz embeddingów '
        '(EMBEDDINGS_ENABLED=false)'
    ),
    'cli.arg.reindex_memory': 'policz embeddingi dla całej pamięci i zakończ (po zmianie modelu)',
    'cli.arg.no_tools': 'bez narzędzi — model tylko rozmawia (TOOLS_ENABLED=false)',
    'cli.arg.dry_run_tools': 'narzędzia zwracają podgląd zamiast działać (SECURITY_DRY_RUN=true)',
    'cli.arg.offline': (
        'praca bez sieci: żaden model nie jest pobierany (równoważne '
        'OFFLINE_MODE=on)'
    ),
    'cli.arg.online': 'zezwól na pobranie brakującego modelu Whisper (równoważne OFFLINE_MODE=off)',
    'cli.arg.log_level': 'nadpisz LOG_LEVEL z .env (DEBUG, INFO, WARNING, ERROR)',
    'cli.arg.ui_lang': 'język interfejsu: en, pl albo auto (nadpisuje UI_LANGUAGE z .env)',
    'cli.arg.metavar_text': 'TEKST',
    'deps.mode.name': 'Tryb pracy',
    'deps.mode.hint': 'komplet do pracy bez sieci przygotuje: python scripts/prepare_offline.py',
    'deps.python.name': 'Python',
    'deps.python.detail': 'wymagane >= {required}, wykryto {detected}',
    'deps.package.name': 'pakiet {name}',
    'deps.package.version': 'wersja {version}',
    'deps.package.version_purpose': 'wersja {version} — {purpose}',
    'deps.package.missing': 'pakiet niezainstalowany',
    'deps.package.missing_purpose': 'pakiet niezainstalowany — {purpose}',
    'deps.wheels.name': 'Koła pip (offline)',
    'deps.wheels.present': '{count} pakietów — instalacja zależności zadziała bez internetu',
    'deps.wheels.missing': (
        'brak lokalnych pakietów; potrzebne tylko do instalacji na maszynie bez '
        'sieci'
    ),
    'deps.wheels.hint': 'na maszynie z internetem: python scripts/prepare_offline.py --wheels',
    'deps.ollama.app': 'Ollama (aplikacja)',
    'deps.ollama.in_path': 'znaleziono w PATH',
    'deps.ollama.not_in_path': (
        'nie znaleziono w PATH (to nie problem, jeśli serwer działa zdalnie)'
    ),
    'deps.ollama.service': 'Ollama (usługa HTTP)',
    'deps.ollama.responds': 'odpowiada, wersja {version}',
    'deps.ollama.no_answer': 'brak odpowiedzi',
    'deps.ollama.start_hint': 'uruchom `ollama serve` ({install})',
    'deps.model.name': 'Model LLM ({model})',
    'deps.model.present': 'model pobrany',
    'deps.model.absent': 'model nie jest pobrany; dostępne: {models}',
    'deps.model.no_models': 'brak modeli',
    'deps.model.not_checked': 'nie sprawdzono — usługa Ollama nie odpowiada',
    'deps.model.offline_note': ' — wymaga internetu, zrób to zawczasu',
    'deps.dir.config': 'Katalog konfiguracji',
    'deps.dir.logs': 'Katalog logów',
    'deps.dir.no_write': 'brak prawa zapisu: {error}',
    'deps.dir.hint': (
        'nadaj uprawnienia zapisu albo wskaż inny katalog zmienną MIKU_CONFIG_DIR'
        ' / MIKU_LOGS_DIR'
    ),
    'deps.dir.writable': 'zapis możliwy',
    'deps.cuda.name': 'Akceleracja CUDA',
    'deps.cuda.hint': 'brak GPU jest w porządku — modele pójdą na CPU',
    'deps.gpu.none': 'nie wykryto GPU z CUDA — praca na CPU',
    'deps.gpu.nvidia_smi': 'nvidia-smi: {name} (sterownik {driver})',
    'deps.gpu.torch': 'torch: {name} (CUDA {driver})',
    'deps.gpu.unknown_driver': 'nieznany',
    'deps.gpu.unknown_version': 'nieznana wersja',
    'mode.forced_offline': 'offline — wymuszony przez OFFLINE_MODE=on (nic nie sięga do sieci)',
    'mode.forced_online': 'online — wymuszony przez OFFLINE_MODE=off (wolno pobierać modele)',
    'mode.auto_offline': 'offline (auto) — modele są lokalnie, nic nie jest pobierane',
    'mode.auto_online': (
        "online (auto) — brak modelu Whisper '{model}' w {path}, więc pobrałby "
        "się przy pierwszym użyciu mikrofonu"
    ),
    'deps.purpose.pydantic': 'walidacja konfiguracji',
    'deps.purpose.pydantic_settings': 'odczyt .env',
    'deps.purpose.dotenv': 'obsługa pliku .env',
    'deps.purpose.httpx': 'klient HTTP do Ollamy',
    'deps.purpose.numpy': 'obróbka sygnału audio',
    'deps.purpose.sounddevice': 'nasłuch mikrofonu (PortAudio)',
    'deps.purpose.faster_whisper': 'transkrypcja mowy',
    'deps.purpose.webrtcvad': 'VAD; bez niego działa wbudowany VAD energetyczny',
    'deps.purpose.openwakeword': (
        'wykrywanie frazy modelem KWS; bez niego działa detektor whisperowy'
    ),
    'deps.purpose.piper': 'synteza mowy; bez pakietu wystarczy binarka `piper` w PATH',
    'deps.purpose.sentence_transformers': 'lokalne embeddingi; bez pakietu policzy je Ollama',
    'deps.purpose.faiss': 'szybkie wyszukiwanie wektorów; bez niego liczy NumPy',
    'deps.mic.name': 'Mikrofon',
    'deps.mic.disabled': 'wyłączony ustawieniem MIC_ENABLED=false',
    'deps.mic.disabled_hint': 'ustaw MIC_ENABLED=true, aby korzystać z trybu głosowego',
    'deps.mic.no_packages': 'brak pakietu sounddevice lub numpy — nie sprawdzono urządzeń',
    'deps.mic.error_hint': 'tryb głosowy będzie wyłączony, czat tekstowy działa normalnie',
    'deps.mic.no_devices': 'system nie zgłasza żadnego urządzenia wejściowego',
    'deps.mic.no_devices_hint': 'podłącz mikrofon albo korzystaj z trybu tekstowego',
    'deps.mic.no_match': "nie znaleziono urządzenia pasującego do '{wanted}'; dostępne: {devices}",
    'deps.mic.no_match_hint': 'popraw AUDIO_INPUT_DEVICE albo zostaw puste (urządzenie domyślne)',
    'deps.mic.ok': '{count} urządzeń wejściowych, wybrane: {selected}',
    'deps.vad.name': 'VAD (detekcja mowy)',
    'deps.whisper.name': 'Faster-Whisper',
    'deps.whisper.detail': 'model {model}, urządzenie {device}/{compute}',
    'deps.whisper.missing': 'pakiet niezainstalowany — tryb głosowy będzie wyłączony',
    'deps.whisper.cache_name': 'Model Whisper (cache)',
    'deps.whisper.local': "model '{model}' jest na dysku — tryb głosowy działa bez internetu",
    'deps.whisper.absent': "brak modelu '{model}' na dysku",
    'deps.whisper.others': '; dostępne lokalnie: {models}',
    'deps.whisper.hint_offline': (
        'python scripts/prepare_offline.py --whisper (wymaga internetu, raz)'
    ),
    'deps.whisper.hint_online': (
        'pobierze się sam przy pierwszym użyciu mikrofonu; żeby zrobić to '
        'zawczasu: python scripts/prepare_offline.py --whisper'
    ),
    'deps.wake.name': 'Słowo aktywujące',
    'deps.wake.source_file': 'wake_word w user_settings.json',
    'deps.wake.source_name': 'imię asystenta',
    'deps.wake.disabled': 'wyłączone (fraza „{phrase}” z: {source})',
    'deps.wake.disabled_hint': (
        'ustaw WAKE_ENABLED=true, aby asystent reagował dopiero na zawołanie'
    ),
    'deps.wake.openwakeword': 'openWakeWord, modele: {models}',
    'deps.wake.whisper_detector': "detektor whisperowy na modelu '{model}'",
    'deps.wake.missing_openwakeword': (
        'WAKE_ENGINE=openwakeword, ale brakuje pakietu albo modelu w {path}'
    ),
    'deps.wake.missing_hint': 'ustaw WAKE_ENGINE=auto (detektor whisperowy działa z dowolną frazą)',
    'deps.wake.ok': '„{phrase}” (z: {source}) — {engine}',
    'deps.wake.model_name': 'Model detektora frazy',
    'deps.wake.model_present': "model '{model}' jest na dysku",
    'deps.wake.model_absent': "brak modelu '{model}' — użyty zostanie model główny",
    'deps.speaker.name': 'Głośnik',
    'deps.speaker.no_packages': 'brak pakietu sounddevice lub numpy — nie sprawdzono urządzeń',
    'deps.speaker.error_hint': 'bez głośnika asystent odpowiada tekstem',
    'deps.speaker.no_devices': 'system nie zgłasza żadnego urządzenia wyjściowego',
    'deps.speaker.no_devices_hint': (
        'podłącz głośnik albo słuchawki; bez nich odpowiedzi zostają tekstem'
    ),
    'deps.speaker.no_match_hint': (
        'popraw AUDIO_OUTPUT_DEVICE albo zostaw puste (urządzenie domyślne)'
    ),
    'deps.speaker.ok': '{count} urządzeń wyjściowych, wybrane: {selected}',
    'deps.tts.name': 'Synteza mowy',
    'deps.tts.disabled_env': 'wyłączona ustawieniem TTS_ENABLED=false',
    'deps.tts.disabled_engine': 'wyłączona ustawieniem voice_engine: "{engine}"',
    'deps.tts.disabled_hint': (
        'ustaw voice_engine: "piper" w config/user_settings.json oraz '
        'TTS_ENABLED=true'
    ),
    'deps.tts.engine_name': 'Silnik mowy (Piper)',
    'deps.tts.package': 'pakiet piper-tts (liczenie w procesie asystenta)',
    'deps.tts.binary': 'program piper',
    'deps.tts.missing': "brak pakietu 'piper-tts' i programu 'piper'",
    'deps.tts.missing_hint': (
        'pip install piper-tts (ciągnie onnxruntime — nie ma kół dla każdej '
        'wersji Pythona) albo wskaż binarkę w .env: PIPER_BINARY=...'
    ),
    'deps.voice.name': 'Głos Piper (model)',
    'deps.voice.found': '{count} głosów: {voices}',
    'deps.voice.selected': ' — wybrany: {selected}',
    'deps.voice.not_found': "nie znaleziono głosu '{wanted}'; dostępne: {voices}",
    'deps.voice.not_found_hint': 'popraw piper_model w config/user_settings.json',
    'deps.voice.no_files': 'nie znaleziono żadnego pliku .onnx w: {directories}',
    'deps.voice.no_files_hint': (
        'python scripts/prepare_offline.py --piper (wymaga internetu, raz) albo '
        'wskaż własny katalog: PIPER_VOICES_DIR w .env'
    ),
    'deps.sem.name': 'Pamięć semantyczna',
    'deps.sem.disabled': 'wyłączona ustawieniem EMBEDDINGS_ENABLED/EMBEDDING_ENGINE',
    'deps.sem.disabled_hint': (
        'ustaw EMBEDDINGS_ENABLED=true, aby asystent kojarzył wspomnienia po '
        'znaczeniu'
    ),
    'deps.sem.st_model': 'sentence-transformers, model {model}',
    'deps.sem.on_disk': ' (na dysku)',
    'deps.sem.will_download': ' — zostanie pobrany przy pierwszym użyciu',
    'deps.sem.download_hint': 'pobierz zawczasu: python scripts/prepare_offline.py --embeddings',
    'deps.sem.st_missing': 'EMBEDDING_ENGINE=sentence-transformers, ale pakietu nie ma',
    'deps.sem.st_missing_hint': '{install}  albo przełącz na EMBEDDING_ENGINE=ollama',
    'deps.sem.ollama_down': 'embeddingi policzy Ollama — usługa teraz nie odpowiada',
    'deps.sem.ollama_ok': 'embeddingi policzy Ollama, model {model}',
    'deps.sem.ollama_missing_model': (
        "embeddingi miałaby liczyć Ollama, ale modelu '{model}' nie ma"
    ),
    'deps.sem.ollama_hint': 'ollama pull {model}  (albo: pip install sentence-transformers)',
    'deps.vector.name': 'Indeks wektorowy',
    'deps.vector.no_numpy': 'brak pakietu numpy — wyszukiwanie po znaczeniu jest wyłączone',
    'deps.vector.faiss': 'FAISS (szybkie wyszukiwanie)',
    'deps.vector.numpy': (
        'NumPy — wystarcza do ~10⁵ wspomnień; FAISS przyspiesza (pip install '
        'faiss-cpu)'
    ),
    'deps.index.name': 'Indeks wspomnień',
    'deps.index.no_database': 'baza jeszcze nie istnieje — indeks powstanie w trakcie rozmowy',
    'deps.index.empty_db': (
        'brak indeksu w bazie — powstanie przy pierwszym użyciu (/pamiec '
        'reindeks)'
    ),
    'deps.index.empty': 'pusty — zbuduje się w trakcie rozmowy albo poleceniem /pamiec reindeks',
    'deps.index.count': '{count} wektorów, model {model}',
    'deps.tools.name': 'Narzędzia (tool calling)',
    'deps.tools.disabled': 'wyłączone ustawieniem TOOLS_ENABLED=false',
    'deps.tools.disabled_hint': 'ustaw TOOLS_ENABLED=true, aby model mógł korzystać z narzędzi',
    'deps.tools.registry_failed': 'rejestr narzędzi nie zbudował się: {error}',
    'deps.tools.registry_hint': 'szczegóły w logs/errors.log',
    'deps.tools.visible': '{visible} z {total} dostępnych dla modelu: {names}',
    'deps.tools.hidden': ' (ukryte przed modelem: {hidden})',
    'deps.tools.all_disabled': 'wszystkie narzędzia są wyłączone konfiguracją',
    'deps.perm.name': 'Uprawnienia narzędzi',
    'deps.perm.terminal': '; potwierdzenia w terminalu',
    'deps.perm.no_terminal': '; brak interaktywnego terminala — HIGH/CRITICAL będą odrzucane',
    'deps.workspace.name': 'Obszar plików narzędzi',
    'deps.workspace.detail': '{count} katalogów: {roots}',
    'deps.workspace.missing': ' (jeszcze nie istnieją — powstaną przy pierwszym zapisie)',
    'deps.workspace.hint': 'dostęp do własnych katalogów dodasz przez FS_ALLOWED_ROOTS w .env',
    'deps.host.name': 'Narzędzia systemowe (Faza 8)',
    'deps.host.apps': 'aplikacje: {detail}',
    'deps.host.processes': 'procesy: {detail}',
    'deps.host.services': 'usługi: {detail}',
    'deps.host.shell': 'powłoka: {detail}',
    'deps.host.pdf': 'PDF: {detail}',
    'deps.host.pdf_missing': 'brak biblioteki (pip install pypdf)',
    'deps.web.name': 'Narzędzia sieciowe',
    'deps.web.hint': 'ustaw OFFLINE_MODE=off i WEB_ENABLED=true, aby pozwolić na dostęp do sieci',
    'deps.web.search': 'wyszukiwanie: {provider}',
    'deps.web.weather': 'pogoda: {provider} (bez klucza)',
    'deps.web.news_feeds': 'wiadomości: {count} kanałów RSS',
    'deps.web.news_search': 'wiadomości: tylko szukanie',
    'deps.web.youtube_key': 'klucz YouTube: {state}',
    'install.pip': 'python -m pip install -r {requirements}',
    'install.pip_offline': (
        'python -m pip install --no-index --find-links {wheelhouse} -r '
        '{requirements}'
    ),
    'install.pip_prepare': (
        'na maszynie z internetem: python scripts/prepare_offline.py --wheels, '
        'potem python -m pip install --no-index --find-links {wheelhouse} -r '
        '{requirements}'
    ),
    'install.run_script': 'uruchom {script}',
    'net.http_status': 'serwer odpowiedział kodem HTTP {code}',
    'net.no_connection': 'brak połączenia ({reason})',
    'net.timeout': 'przekroczono limit czasu ({seconds} s)',
    'net.error': 'błąd sieci: {error}',
    'deps.vad.webrtc_missing': 'VAD_ENGINE=webrtc, ale pakiet webrtcvad nie jest zainstalowany',
    'deps.vad.energy': 'detektor energetyczny (wbudowany, bez dodatkowych zależności)',
    'deps.vad.webrtc': 'webrtcvad, agresywność {level}',
    'status.tools_off': 'wyłączone (TOOLS_ENABLED=false)',
    'status.tools_visible': '{count} dostępnych ({names}); {policy}',
    'status.policy.confirm_from': 'potwierdzenia od {level}',
    'status.policy.critical_allowed': 'CRITICAL dozwolone',
    'status.policy.critical_blocked': 'CRITICAL zablokowane',
    'status.policy.limit': 'limit {count}/turę',
    'status.policy.dry_run': 'tryb próbny (dry-run)',
    'status.policy.allowed': 'lista dozwolonych: {names}',
    'status.policy.disabled': 'wyłączone: {names}',
    'status.audit.off': 'wyłączony (SECURITY_AUDIT_ENABLED=false)',
    'status.audit.empty': 'brak wywołań w tej sesji',
    'status.audit.summary': '{count} wywołań, {denied} odmów ({where})',
    'status.audit.where_db': 'baza + log',
    'status.audit.where_log': 'log',
    'status.audit.confirmed': ' (potwierdzone)',
    'status.audit.line': '{tool} [{risk}] {decision}{confirmed} → {state} w {ms} ms',
    'status.memory.off': 'wyłączona (MEMORY_ENABLED=false)',
    'status.memory.ram_only': 'tylko pamięć robocza — {reason}',
    'status.memory.db_unavailable': 'baza niedostępna',
    'status.memory.no_disk': 'brak zapisu na dysku',
    'status.memory.stats_failed': 'nieznane (błąd odczytu)',
    'status.semantic.off': 'wyłączona (EMBEDDINGS_ENABLED=false)',
    'status.semantic.no_db': 'wyłączona (brak bazy)',
    'status.db.in_memory': 'w pamięci procesu',
    'status.db.describe': (
        '{where} (schemat v{version}/{latest}, dziennik {journal}, wyszukiwanie '
        '{search})'
    ),
    'status.stats.describe': (
        '{conversations} rozmów, {messages} wiadomości, {summaries} streszczeń, '
        '{facts} faktów, {preferences} preferencji, {notes} notatek, {embeddings}'
        ' wektorów, {audit} wpisów audytu'
    ),
    'status.semantic.no_engine': 'wyłączona (brak silnika embeddingów)',
    'status.semantic.unavailable': 'niedostępna — {reason}',
    'status.semantic.index_off': 'nieuruchomiony',
    'status.semantic.describe': '{provider}, indeks {index}: {count} wektorów',
    'status.semantic.disabled': 'wyłączona',
    "deps.ollama.no_model": "usługa działa, brak modelu",
    "deps.ollama.unreachable": "usługa nie odpowiada",
    # --- tryb bezobsługowy (--headless) ----------------------------------- #
    "cli.arg.headless": (
        "usługa w tle: tylko mikrofon i mowa, bez okna i bez klawiatury "
        "(pod systemd --user / Harmonogram zadań)"
    ),
    "cli.main.headless_conflict": "--headless wyklucza się z --gui i z --terminal.",
    "cli.main.headless_needs_voice": (
        "--headless nie ma innego wejścia niż mikrofon, więc --no-voice nie ma sensu."
    ),
    "cli.headless.starting": "{name} startuje jako usługa w tle (bez okna, bez klawiatury).",
    "cli.headless.no_microphone": (
        "Usługa nie ma wejścia głosowego, a to jej jedyny sposób na przyjęcie polecenia."
    ),
    "cli.headless.no_microphone_hint": (
        "sprawdź mikrofon: python main.py --audio-check — a na Linuksie upewnij się, że "
        "usługa widzi sesję dźwiękową (PipeWire/PulseAudio w sesji użytkownika)"
    ),
    "cli.headless.no_speech": "Brak wyjścia głosowego ({detail}) — odpowiedzi trafiają tylko do logu.",
    "cli.headless.confirmations_denied": (
        "Nie ma tu kogo pytać o zgodę, więc akcje HIGH i CRITICAL są odrzucane automatycznie."
    ),
    "cli.headless.deny_reason": "usługa w tle — brak kanału do potwierdzenia akcji",
    "cli.headless.microphone_back": "Wejście głosowe wróciło.",
    "cli.headless.stopped": "Usługa w tle zatrzymana.",
}

_CATALOGS: Final[dict[str, dict[str, str]]] = {"en": _EN, "pl": _PL}


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def normalize_ui_language(code: str | None, *, reply_language: str | None = None) -> str:
    """Sprowadź ustawienie do obsługiwanego kodu języka interfejsu.

    ``auto`` znaczy „idź za językiem odpowiedzi" — dzięki temu ktoś, kto ustawił
    ``LANGUAGE=pl``, dostaje też polski interfejs bez drugiego wpisu w ``.env``.
    Nieznany kod schodzi do angielskiego, bo to katalog wzorcowy.
    """
    wanted = (code or "").strip().lower()
    if wanted in ("", "auto"):
        wanted = (reply_language or "").strip().lower()
    wanted = wanted.replace("_", "-").split("-", 1)[0][:2]
    if wanted in SUPPORTED_UI_LANGUAGES:
        return wanted
    return DEFAULT_UI_LANGUAGE


def set_ui_language(code: str | None, *, reply_language: str | None = None) -> str:
    """Ustaw język interfejsu na czas działania programu. Zwraca użyty kod."""
    global _current
    resolved = normalize_ui_language(code, reply_language=reply_language)
    with _lock:
        _current = resolved
    return resolved


def ui_language() -> str:
    """Aktualny język interfejsu."""
    with _lock:
        return _current


def t(key: str, /, *, _lang: str | None = None, **params: Any) -> str:
    """Tekst interfejsu dla klucza.

    Kolejność szukania: wybrany język → angielski → sam klucz. Brakujące
    tłumaczenie nigdy nie kończy się wyjątkiem ani pustym napisem: użytkownik
    zobaczy tekst angielski, a w logu pojawi się jednorazowe ostrzeżenie.

    ``_lang`` wymusza język JEDNEGO tekstu (rzeczy **mówione** idą językiem
    odpowiedzi, nie interfejsu). Podkreślenie w nazwie nie jest ozdobą: parametr
    dzieli przestrzeń nazw ze wstawkami, a „language" jest zbyt naturalną nazwą
    wstawki — regresja z prawdziwego uruchomienia, gdzie panel stanu pokazał
    „Język odpowiedzi: {language}" zamiast wartości.
    """
    code = normalize_ui_language(_lang) if _lang else ui_language()
    catalog = _CATALOGS.get(code, _EN)
    template = catalog.get(key)
    if template is None:
        template = _EN.get(key)
    if template is None:
        if key not in _missing_reported:
            _missing_reported.add(key)
            logger.warning("Brak tekstu interfejsu dla klucza %r", key)
        return key
    if not params:
        return template
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):  # pragma: no cover - zła liczba parametrów
        logger.warning("Nie udało się wstawić wartości do tekstu %r", key)
        return template


def has_key(key: str) -> bool:
    """Czy katalog zna taki klucz? (Angielski jest wzorcem.)"""
    return key in _EN


def translate_or_text(value: str, **params: Any) -> str:
    """Przetłumacz, jeśli to klucz katalogu; w przeciwnym razie oddaj tekst bez zmian.

    Potrzebne tam, gdzie ta sama pozycja bywa wypełniana przez rdzeń programu
    (kluczem) i przez kod z zewnątrz — np. ``PackageRequirement.purpose``
    dopisywany przez kolejne fazy albo wtyczki.
    """
    if has_key(value):
        return t(value, **params)
    return value


def catalog(code: str) -> dict[str, str]:
    """Kopia katalogu danego języka (do testów i diagnostyki)."""
    return dict(_CATALOGS.get(normalize_ui_language(code), _EN))


def register_texts(language: str, texts: dict[str, str]) -> None:
    """Dopisz teksty do katalogu (dla kolejnych faz i wtyczek).

    Nie nadpisuje kluczy już istniejących — tłumaczenie z rdzenia programu ma
    pierwszeństwo przed tym, co dołoży wtyczka.
    """
    target = _CATALOGS.setdefault(normalize_ui_language(language), {})
    for key, value in texts.items():
        target.setdefault(key, value)


__all__ = [
    "DEFAULT_UI_LANGUAGE",
    "SUPPORTED_UI_LANGUAGES",
    "catalog",
    "has_key",
    "normalize_ui_language",
    "register_texts",
    "set_ui_language",
    "t",
    "translate_or_text",
    "ui_language",
]
