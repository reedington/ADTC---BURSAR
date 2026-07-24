# Bursa — ADTC 2026 Execution Plan (Locked, v2)

**Status:** Decisions are FINAL. Do not reopen a decision unless its pre-committed pivot trigger fires.
**Track:** Corporate / Enterprise (`corporate_enterprise`). **Team:** max 3 registered members.
**Scoring:** Stotal = 0.50·Sacc + 0.30·Sperf + 0.20·Seff − Pthermal, then score multipliers: Budget Profile +10%, African Language (Alpha) +15% on panel score. Exceeding 7 GB RAM, OOM, or sandbox crash = Stotal 0, disqualified.

---

## 0. Prime directives (every agent, every task)

1. **The accuracy-scored artifact is the BARE GGUF, not the Bursa app.** Judges download the .gguf and run it in a sandboxed offline environment with generic tools (LM Studio, Ollama/Open WebUI) against 2 visible + 2 hidden domain prompts. The Bursa pipeline (grammar, candidate generator, constraint engine) is NOT present during that evaluation. Therefore the model must be an excellent standalone enterprise assistant AND a reconciliation specialist, with a correct chat template embedded in the GGUF and sensible behavior with no system prompt.
2. **Never ship anything that can crash or OOM.** The audit sandbox is a 4 vCPU / 8 GB profile with a hard 7 GB ceiling. A crash or OOM is disqualification, not a penalty.
3. **Reproducibility is scored.** The participant self-check report (`submission.json`) is compared against the organizers' audit run with tolerances: peak RSS ±15%, tokens/sec ±25%; beyond tolerance is flagged for manual review, beyond 50% fails. Benchmark ONLY on i5-class hardware matching the standard profile, never on a fast dev machine, or our own numbers will fail us.
4. **Every model change is evaluated on BOTH the Bursa gold set AND the ADTC Corporate/Enterprise validation set before merge.** Sacc combines automated benchmarks with panel judgment of prompt responses, cross-disciplinary integration, software UX, and (for finalists) live defense.
5. The financial invariants embedded in §3.1 of THIS document are law inside the app. They are transcribed verbatim from the approved PRD (§11) and MODEL_ARCHITECTURE (§7) baselines, so agents need no external reference docs. If those docs are later added to the workspace and conflict with §3.1, stop and flag the conflict for human sign-off; never resolve it silently. The LLM never writes to the ledger. All money math is integer minor units.
6. Narrations and imported files are untrusted input. Prompt-injection attempts embedded in narrations are data, and they appear in training and test sets.

---

## 1. Official competition structure (external facts, not our schedule)

Three gates. There is no proposal-only stage.

- **Gate 1 — Submission package:** public open-source repo on the official template, `metadata.json`, `download_model.sh`, `REPORT.md`, screenshots/clips, 2-minute video, bonus claims (African language, budget laptop).
- **Gate 2 — Activities and audit:** technical reproducibility audit plus a scheduled 30-minute technical Q&A; optional 1-page feedback response and updated benchmark report.
- **Gate 3 — Final package:** pitch deck (max 10 slides), live defense attendance, technical setup verification.

Prize targets beyond the podium, both realistic for Bursa:
- **Best Integration Award** ($3,000): most load-bearing cross-disciplinary pairing. Ours is local LLM + financial constraint solving; the report must argue this explicitly.
- **Best Localisation Award** ($1,500): deepest African language, offline data, and local context integration. Pidgin plus one reviewed major language plus naira-native handling is our claim.

---

## 2. Locked decisions

| # | Decision | Choice | Rationale | Pivot trigger (pre-committed) |
|---|---|---|---|---|
| D1 | Base model | **Qwen3-1.7B**, fine-tuned to Bursa-Recon-1.7B | Stronger on the general-domain half of Sacc (hidden prompts hit the bare model); Apache 2.0; multilingual; llama.cpp support | Switch to **Qwen3-0.6B** at checkpoint C2 if: measured 1.7B Q4 generation < 13 tps on i5-class hardware, OR fine-tuned 0.6B is within 5 points on the gold set AND holds up qualitatively on general enterprise prompts. Two Sperf formulas are published (capped at a 15 tps reference in the profiler; relative to the fastest submission on the official site); both reward speed, the relative one rewards it more, so ties go to the smaller model |
| D2 | Quantization | **Q4_K_M** | Lower RAM and higher tps feed Seff and Sperf directly; fine-tuning recovers most quant loss | Switch to Q5_K_M only if Q4 loses > 3 points exact-allocation accuracy vs the unquantized checkpoint |
| D3 | Runtime config | llama.cpp `llama-server` inside the app; GBNF grammar for app inference only; temperature 0; Qwen3 thinking disabled; context cap **2048**; KV cache q8_0; prompt caching ON. The shipped GGUF carries a correct chat template and must behave well in plain LM Studio/Ollama with no system prompt | The judged artifact is the bare GGUF; the grammar exists only inside Bursa | None. Fixed. |
| D4 | CPU threads | Tune for **4 threads** (audit sandbox is a 4 vCPU profile); confirm with a thread sweep on i5-class hardware | Optimizing for 8 laptop threads misleads us about audit numbers | Adjust only from sweep data on matching hardware |
| D5 | Fine-tuning | **LoRA r=16, alpha=32, via Unsloth** on Udutech GPU credits; merge adapter → GGUF → quantize | Fast, low-VRAM, documented path to GGUF | Higher rank or full fine-tune only if LoRA plateaus below checkpoint C3 accuracy |
| D6 | Training mix | **70% reconciliation / 30% general enterprise instruction data** (summarization, drafting, analysis). Reconciliation examples in TWO formats: app-style constrained JSON behind the app system prompt, AND chat-style worked reconciliation answers with no system prompt | The hidden prompts hit a bare chat model; it must reason about reconciliation conversationally, not only emit JSON | Raise general share to 40% if ADTC validation regression > 2 points |
| D7 | App stack | **Python FastAPI + SQLite (WAL) + Jinja2/HTMX**, single process, llama-server as managed subprocess | Minimal RAM, no node chain, one process, fast for agents | None. Fixed. |
| D8 | OCR | **Tesseract**, ONE path: clear transfer screenshot → fields → link to statement row. Built in Phase 3 only if the core pipeline is stable | Demo and UX value for the qualitative score | **Cut entirely** at the Phase 3 ship-or-cut review if not demo-reliable |
| D9 | Natural-language reporting | **CUT** | No scored value; adds RAM and crash surface | None. Do not build |
| D10 | Languages | **English + Nigerian Pidgin (`pcm`), plus ONE reviewed major language (Yoruba, Hausa, or Igbo) if a competent native reviewer is confirmed before dataset construction** | The African Alpha bonus is a **+15% multiplier on panel score**, far more valuable than previously assumed; meaningful functionality in one African language earns it. Pidgin likely qualifies; one reviewed major language makes the claim unambiguous and feeds the Best Localisation prize | Claim `african_alpha_claim: true` only for languages with reviewed, demonstrable functionality |
| D11 | Confidence | Logistic-regression calibrator over observable features; bands start 0.90 / 0.65, recalibrated to **zero false auto-posts** | Model self-confidence is not trusted | Thresholds move; the zero-false-auto-post rule never does |
| D12 | Money | Naira display, integer minor units internally. No floats near money | Locked in PRD | None |
| D13 | Packaging | Fork the official template. `metadata.json` fully filled (see §7), `download_model.sh` idempotent and credential-free to `model/`, weights public on Hugging Face, `*.gguf` and `model/` in `.gitignore`, REPORT.md 1–3 pages | The evaluator downloads weights fresh via the script; an LLM-based audit system also reads REPORT.md, so it must be clear, factual, specific | None |
| D14 | Test prompts | Exactly 2 prompts in `metadata.json`, **fully self-contained** so the bare model shines in a chat UI: 1) a reconciliation scenario with the transaction, narration, and 3–4 candidate students with balances embedded in the prompt, asking for the correct split with reasoning; 2) a noisy Pidgin narration scenario where the correct answer is to flag for human review, demonstrating calibrated abstention | Judges paste these into LM Studio/Ollama; the prompt must carry its own context because the app's candidate generator is absent. Organizers add 2 hidden domain prompts to test overfitting | None |

---

## 3. Architecture (frozen, from approved docs)

```
CSV / screenshot → Import + OCR → Normalize + dedupe → Deterministic matcher
  → unambiguous → Constraint engine
  → ambiguous  → Candidate generator (max 5) → Bursa-Recon LLM (GBNF JSON) → Schema validation → Constraint engine
Constraint engine → Calibrated confidence
  ≥ 0.90 → proposed auto-match (reversible)   0.65–0.89 → review queue   < 0.65 → unmatched
All approved postings → append-only ledger (source, evidence, actor, decision path)
```

Boundaries that must never blur: OCR transcribes, it does not reconcile. The LLM ranks and explains, it does not post. The constraint engine owns all arithmetic and validation. The bursar owns uncertainty. The bank statement is the source of truth for received electronic money.

### 3.1 Financial invariants v1 (embedded, authoritative for agents)

The reference docs may not exist in every agent workspace, so the invariant set lives here in full. Source: MODEL_ARCHITECTURE §7 (ledger invariants) and PRD §11 (business rules) of the approved baselines. Editing any line below requires human sign-off.

**Ledger invariants — the constraint engine enforces each one, and each needs a test proving it blocks posting:**

- **INV-01** Allocations cannot exceed the transaction amount.
- **INV-02** A transaction reference cannot be posted twice.
- **INV-03** Amounts are integer minor units; floating-point arithmetic is prohibited anywhere near money.
- **INV-04** Allocations may only reference student and fee IDs supplied to the pipeline.
- **INV-05** A student balance cannot become negative unless the excess is explicitly recorded as credit.
- **INV-06** Unallocated money remains visible as an unapplied amount.
- **INV-07** Overpayments enter a credit account and are never silently discarded.
- **INV-08** Reversals create compensating entries; financial history is never destructively edited or deleted.
- **INV-09** Every ledger posting records its source, evidence, actor, timestamp, and decision path.
- **INV-10** A failed invariant routes the transaction to review and prevents posting.

**Business rules — product behavior on top of the ledger:**

- **BR-01** A bank credit transaction represents received money; a receipt image alone is evidence until linked to a statement transaction or explicitly classified by the bursar.
- **BR-02** A payment may have zero, one, or several student allocations and may remain partly or fully unapplied.
- **BR-03** A student's fee status (`outstanding`, `part_paid`, `cleared`, `credit`) derives from ledger totals, never from a manually edited status field.
- **BR-04** An automatic match is a reversible proposal; medium- and low-confidence results cannot post without human action.
- **BR-05** Model or OCR failure can never post a financial record.

---

## 4. Agent work packages

### Agent F — Financial Core
Build: canonical data model; CSV importers with column mapping and row-level errors; idempotent re-import; duplicate detection; deterministic matcher with reason codes; constraint engine enforcing INV-01 through INV-10; append-only ledger with reversals.
**Accept when:** re-importing the same statement twice produces zero new transactions; every invariant INV-01 through INV-10 has a failing test that blocks posting; BR-01 through BR-05 each have behavior tests; balances reconstruct purely from ledger events; property-based tests pass on allocation sums.

### Agent M — Model & Inference
Build: llama-server lifecycle management; GBNF grammar from the reconciliation schema with IDs restricted to supplied candidates; prompt builder (≤ 1,300 tokens); schema validation rejecting unknown IDs; confidence calibrator; **chat-template verification: the shipped GGUF must load and converse correctly in stock Ollama and LM Studio with no configuration**.
**Accept when:** 1,000 consecutive app inferences give ≥ 99.5% valid JSON and 0% unknown-ID outputs; the bare GGUF answers general enterprise prompts and the two test prompts well in a clean Ollama install.

### Agent D — Data & Evaluation
Build: gold set of ~150 team-authored canonical cases across all 13 scenario families; ~4,000 synthetic training examples (OCR corruption, Nigerian name/abbreviation variants, Pidgin, injection attempts) in BOTH formats per D6; 25–30% abstention labels; near-duplicate detection; splits by guardian family and template family, 70/15/15; frozen hidden test set; evaluation harness for all MODEL_ARCHITECTURE §12 metrics; ADTC validation-set runner; **a bare-model prompt suite simulating hidden-prompt conditions (generic enterprise prompts run through plain llama.cpp chat, no system prompt)**.
**Accept when:** one command emits the full scorecard; no family/template leakage; baselines exist for Qwen3-0.6B, Qwen3-1.7B, Gemma 3 1B, Llama 3.2 3B on the gold set, the ADTC validation set, and the bare-model suite.

### Agent T — Training
Build: Unsloth LoRA pipeline per D5/D6; adapter merge; GGUF conversion with correct chat template; Q4_K_M and Q5_K_M quantization; automated post-training eval on all three suites.
**Accept when:** one script goes dataset → quantized GGUF → scorecard; checkpoint C3 criteria met.

### Agent P — Platform & Profiling
Build: llama.cpp toolchain including **llama-bench on PATH** (the profiler requires it); `adtc-profiler` installed and run in participant mode against our template-structured repo producing a valid `submission.json` with `measured_on: participant_laptop`; repeatable 10-run benchmark logging peak RSS, tps, temperature, throttle flags; 4-thread sweep on i5-class hardware; offline verification with networking disabled; `download_model.sh`; clean-machine install test; Docker parity check against the profiler's audit container settings.
**Accept when:** 10 consecutive participant-mode runs on i5-class hardware succeed with peak model RSS ≤ 1.8 GB, generation ≥ 15 tps, temp < 85°C, zero throttling, zero crashes; numbers are ones we are confident the audit will reproduce within ±15% RAM and ±25% tps.

### Agent X — Docs, Demo & Defense
Build: REPORT.md (1–3 pages: problem and African context, design decisions and alternatives, constraints, measured benchmarks), written knowing an LLM-based audit system reads it; README with scorecard; 2-minute video per PRD §17 including a real-bursar clip and live profiler readout; demo fixtures for all 13 scenarios; **Gate 2 prep: reproducibility notes and a Q&A brief covering every design decision and its rationale; Gate 3 prep: 10-slide pitch deck skeleton**; explicit Best Integration and Best Localisation arguments in the report.
**Accept when:** the report contains only measured numbers; the video shows the model running; bonus claims are backed by demonstrable functionality.

---

## 5. Phases and checkpoints (internal; distinct from the official Gates)

Phases run strictly in order. A phase starts only when the previous checkpoint passes.

**Phase 0 — Prerequisites**
- Register team of ≤ 3 on Devpost. Apply for Udutech GPU credits immediately.
- Copy the approved PRD.md and MODEL_ARCHITECTURE.md into the repo's `docs/` directory so every agent workspace has the reference docs; reconcile §3.1 against them and flag any mismatch for human sign-off.
- Fork the submission template; install adtc-profiler and llama.cpp (with llama-bench); a hello-world GGUF passes participant mode on i5-class hardware.
- Obtain the Corporate/Enterprise validation set; run all four baseline models; produce the baseline table.
- Confirm the native reviewer for the third language, or lock D10 to English + Pidgin.

**Phase 1 — Financial core**
- Agent F ships data model, imports, constraint engine, ledger, deterministic matcher.

**CHECKPOINT C1:** Financial core tests green. Baseline table exists. Participant-mode profiler run is clean.

**Phase 2 — Model integration and evaluation**
- Agent M ships inference integration, grammar, prompt builder, calibrator v1, chat-template checks.
- Agent D ships gold set v1, synthetic generator, harness, splits, bare-model suite.
- End-to-end run: statement in → queues → approved postings → balanced ledger.
- Measure 1.7B Q4 and 0.6B Q4 on i5-class hardware: tps at 4 threads, RSS, temp, zero-shot accuracy on all three suites.

**CHECKPOINT C2:** Execute the D1 pivot rule on measured numbers. Model, quant, context, threads FROZEN.

**Phase 3 — Fine-tuning and calibration**
- Agent T trains, quantizes, evaluates on all three suites.
- Calibrate confidence to zero false auto-posts.
- Agent P runs thermal and thread sweeps, offline tests, submission.json generation.
- Agent X drafts REPORT.md with real numbers.
- OCR built only if all green; ship-or-cut review decides.

**CHECKPOINT C3:** Fine-tuned model beats zero-shot on the gold set; ADTC validation regression ≤ 2 points; bare-model suite quality holds; JSON validity ≥ 99.5%; zero incorrect auto-posts. If C3 fails, ship the best zero-shot baseline and say so honestly.

**Phase 4 — Freeze, hardening, Gate 1 submission**
- Model freeze. Run the frozen hidden test set once; record.
- Hardening: kill/resume, idempotency, 10x profiler runs.
- Verify the bare GGUF in clean Ollama and LM Studio installs, including both test prompts.
- Finalize metadata.json, REPORT.md, video. Clean-machine dry run including `bash download_model.sh`.
- Submit with buffer before the official deadline. Note: submissions are pinned to the git commit hash captured at submission time; no post-submission edits count.

**CHECKPOINT C4:** Submission checklist (§8) 100% complete.

**Phase 5 — Gate 2 and Gate 3 readiness (post-submission)**
- Keep the exact benchmark environment reproducible for the audit window.
- Rehearse the 30-minute technical Q&A from the Agent X brief.
- Build the 10-slide deck; rehearse the live defense with the working demo.

---

## 6. Output contract (v1, app-internal, do not extend)

```json
{
  "transaction_id": "<from prompt>",
  "interpretation": {
    "payer_name": "", "student_mentions": [], "term": "",
    "fee_types": [], "payment_intent": ""
  },
  "candidate_allocations": [
    { "student_id": "<must be in candidates>", "amount_minor": 0, "reason_codes": [] }
  ],
  "recommended_action": "auto|review|unmatched",
  "explanation": "",
  "ambiguities": []
}
```

Grammar restricts `student_id` and `reason_codes` to prompt-supplied values. Output failing validation becomes `review`, logged, retried at most once, never posted. This contract governs the app path only; the bare model conversational behavior is governed by D6 training.

---

## 7. metadata.json (fill exactly, no placeholders)

- `team_id`: registered ADTF team ID (a mismatched team ID fails the audit comparison)
- `domain`: `corporate_enterprise`
- `language_scope`: `["en", "pcm"]` plus the BCP-47 code of the third language if D10 confirms it
- `african_alpha_claim`: `true` (backed by demonstrable Pidgin functionality at minimum)
- `budget_laptop_claim`: `true`
- `submitter`: name, email, and GitHub handle of the registered submitting member
- `cross_disciplinary_pairing`: discipline `finance_accounting`, `load_bearing: true`, description naming the constraint-solving integration
- `test_prompts`: exactly 2, per D14
- `model`: name `Bursa-Recon-1.7B-Q4_K_M` (or 0.6B per C2), runtime `llama.cpp`, quantization `GGUF Q4_K_M`, honest `parameters_estimate`, packaging `binary_bundle`
- `_runtime.model_path`: `model/<file>.gguf`, exactly matching what `download_model.sh` produces

---

## 8. Submission checklist (Gate 1)

- [ ] Repo is public, forked from the official template, structure intact
- [ ] `metadata.json` complete per §7, zero placeholders, exactly 2 test prompts
- [ ] `bash download_model.sh` idempotent, credential-free, downloads the GGUF to the exact `_runtime.model_path`
- [ ] `*.gguf` and `model/` in `.gitignore`; no weights, no student PII in git
- [ ] `adtc-profiler run --mode participant` produces a valid `submission.json` on i5-class hardware; committed alongside the report
- [ ] Bare GGUF verified in clean Ollama AND LM Studio: correct chat template, good answers to both test prompts and generic enterprise prompts
- [ ] App completes all core workflows fully offline
- [ ] REPORT.md 1–3 pages, measured numbers only, explicit Best Integration and Best Localisation arguments
- [ ] 2-minute video: problem → offline import → fast path → ambiguous sibling split → safety routing → balances and audit trail → live profiler readout; real-bursar clip included
- [ ] 13 demo scenarios have repeatable fixtures
- [ ] Team of ≤ 3 registered; Devpost rules page re-checked before submitting

---

## 9. Benchmark tables to fill (REPORT.md copies these)

1. Baselines: 4 models × {gold-set accuracy, ADTC validation score, bare-model suite quality, tps at 4 threads, peak RSS, temp}.
2. Quantization: Q4_K_M vs Q5_K_M vs merged checkpoint.
3. Thread sweep at i5-class: threads × {tps, sustained temp}.
4. Before/after fine-tuning on all three suites.
5. Final 10-run participant-mode summary: peak RSS, tps, max temp, throttle events (0), crashes (0).

---

## 10. Risk tripwires

| Tripwire | Automatic response |
|---|---|
| GPU credits not approved before Phase 3 needs them | Train on Colab/rented GPU; budget owner decides immediately |
| 1.7B Q4 < 13 tps at 4 threads on i5-class hardware | D1 pivot fires at C2 |
| ADTC validation regression > 2 points after fine-tune | Raise general mix to 40%, retrain once; else ship zero-shot |
| Bare model answers generic prompts with reconciliation JSON | Training bug: rebalance D6 dual-format data, retrain; blocks C3 |
| Self-check hardware differs from audit-class hardware | Stop; re-benchmark on i5-class machine; tolerance failure risks a failed audit verdict |
| Any profiler crash or OOM in Phase 4 | All agents stop feature work; root-cause first |
| Peak model RSS > 1.8 GB | Drop context to 1536, verify KV q8_0, confirm nothing else loads during profiling |
| Sustained temp ≥ 82°C | Reduce threads by one; re-sweep |
| OCR flaky at ship-or-cut review | Cut it; report lists it as future work |
| Gold set < 25% abstention labels | Data agent adds cases before any training run |
| §3.1 conflicts with PRD.md or MODEL_ARCHITECTURE.md once present in the workspace | Stop; flag both versions for human sign-off; never silently pick one |
| Third-language reviewer unavailable | Ship English + Pidgin; keep `african_alpha_claim` honest |

---

## 11. What we are NOT building

Payment APIs, virtual accounts, refunds, parent portal, SMS/email, natural-language reporting, unrestricted SQL, handwriting OCR, cloud anything, general chatbot product surface, full ERP. Each costs RAM, reliability, or time and earns zero points.
