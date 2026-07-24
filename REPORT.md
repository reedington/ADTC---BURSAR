# Bursa — ADTC 2026 Report

> **Status: DRAFT skeleton.** Every benchmark number here is TODO and must be *measured on
> i5-class hardware at 4 threads* before it appears (execution-plan Prime Directive 3 / Hard
> Rule 5). No unmeasured number is written into this report.

## Problem & African context
_TODO_ — Fragmented school-fee payment records in Nigerian/African schools: shared bank
accounts, parent-named transfers, nicknames/abbreviations, sibling and instalment payments,
offline/low-connectivity operation. (See `docs/PROJECT_CONTEXT.md`, PRD §2.)

## Design decisions & alternatives
_TODO_ — Hybrid deterministic rules + local LLM + financial constraint engine; charge-level
append-only ledger; Qwen3-1.7B → Bursa-Recon-1.7B at Q4_K_M on `llama.cpp`; calibrated
abstention. Alternatives considered (0.6B, Q5_K_M, student-level ledger) and why.

## Constraints
_TODO_ — 7 GB RAM ceiling, 4 vCPU audit profile, fully offline, no thermal throttle;
integer minor units; the LLM never writes to the ledger.

## Measured benchmarks
_TODO_ — Fill from the execution-plan §9 tables: baselines (4 models), quantization
(Q4_K_M vs Q5_K_M vs merged), thread sweep, before/after fine-tune, 10-run participant-mode
summary (peak RSS, tps, max temp, throttle=0, crashes=0).

## Best Integration Award argument
_TODO_ — Local LLM + financial constraint solving as the load-bearing cross-disciplinary
pairing; neither half reconciles messy real payments alone.

## Best Localisation Award argument
_TODO_ — English + Nigerian Pidgin (+ one reviewed major language if a native reviewer is
confirmed), naira-native handling, offline local-context data.
