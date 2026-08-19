---
name: Feature request
about: An idea for a new feature, or a change to an existing one
title: ''
labels: enhancement
assignees: ''
---

## The problem

<!-- Start from the problem, not the solution. What are you trying to do, and
     what gets in the way? Without that it is hard to judge whether the proposed
     solution is the best one available. -->

## The proposal

<!-- How it would work from the user's point of view. -->

## What this concerns

- [ ] a new **tool** (something the model can call)
- [ ] a new **plugin**
- [ ] speech recognition / the wake word
- [ ] speech synthesis
- [ ] memory (long-term or semantic)
- [ ] the interface (window, terminal, service mode)
- [ ] installation and configuration
- [ ] something else:

## If this is a new tool

<!-- Fill this in if it concerns something the model should be able to call. -->

**Risk level** — SAFE (read only) / MEDIUM (reversible change) /
HIGH (consequences cannot be undone) / CRITICAL (can break the system):

**Does it need anything absent from a clean installation** (a system program,
an API key, hardware)?

## The project's boundaries

Check whether the idea runs into something that is a **deliberate decision**
here rather than an omission — the full list is in the Limitations section of
the README:

- [ ] The model **never** executes anything outside the defined tools.
      There will be no "write and run a script" and no arbitrary shell.
- [ ] HIGH and CRITICAL **always** require confirmation. There will be no
      "trust me, stop asking" mode.
- [ ] The architecture is **single-user and single-machine**: no accounts, no
      cloud, no synchronisation between devices, no remote access.
- [ ] Everything is computed **locally**. Proposals to send conversation content
      or memory to an external API are out of scope.

<!-- If your idea requires moving one of these boundaries — say so explicitly and
     explain why. That does not mean "no", it means "let us talk". -->
