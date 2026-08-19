---
name: Bug report
about: Something does not work the way the README describes
title: ''
labels: bug
assignees: ''
---

## What happens

<!-- One or two sentences. What you did, and what happened instead of what you expected. -->

## How to reproduce it

1.
2.
3.

**What I expected:**
**What happened:**

## Environment

<!-- PASTE THE OUTPUT OF THIS COMMAND. It is one line that describes the whole
     environment (system, Python, packages, GPU, microphone, models, Ollama) and
     saves us both a round of questions. -->

```
$ python main.py --check-deps


```

## Logs

<!-- Full tracebacks land in logs/errors.log, not on the screen.
     Paste the last dozen or so lines around the error.
     In service mode: journalctl --user -u miku-assistant -n 50 -->

```


```

## Before you send — three things

- [ ] I have read the **Limitations / Known limitations** section of the README.
      Hallucinations from a small model, speech-recognition mistakes in noise,
      being asked for consent on every HIGH action, and RVC not working are
      **documented properties**, not bugs.
- [ ] I have checked that what I am pasting contains no API keys, tokens or
      paths carrying my account name.
- [ ] The problem occurs even when `python main.py --check-deps` reports no
      errors (or I am attaching the report showing what is missing).
