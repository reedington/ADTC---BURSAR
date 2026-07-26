# Bursa — ADTC 2026 Technical Report

**Track:** Corporate / Enterprise
**Runtime:** `llama.cpp` with GGUF weights
**Status:** Engineering prototype; target-hardware measurements are not yet claimed

## 1. Problem and African context

Many African schools receive fees into one shared bank account. Transfer records often contain a
parent's name rather than a student's name, nicknames and spelling errors, several instalments,
one payment for multiple siblings, or one payment covering multiple fee items. Bursars reconcile
these records manually against student spreadsheets. The result is slow posting, disputed
balances, unapplied money, duplicate posting risk, and weak audit trails.

Bursa is an offline reconciliation copilot for this workflow. It imports student, fee, and bank
records; identifies clear matches deterministically; uses a local language model only for
ambiguous narration; validates every proposed allocation; and maintains an append-only
student-fee ledger. It is designed for the ordinary 8 GB school laptop rather than a cloud API.
Naira is displayed to the user and represented internally as integer minor units.

The initial context is Nigerian school-fee reconciliation, including Nigerian names, naira,
shared-account workflows, English, and Nigerian Pidgin examples. The architecture generalizes to
other African schools without requiring a specific bank or payment provider.

## 2. Load-bearing cross-disciplinary integration

Bursa combines local language-model inference with formal financial constraints:

1. Deterministic matching handles exact student IDs and other unambiguous evidence.
2. For ambiguous payments, a candidate generator supplies at most five known students.
3. A local GGUF model interprets the narration, ranks candidates, explains evidence, and may
   abstain.
4. A GBNF grammar and schema validator restrict outputs to the supplied candidate IDs.
5. A deterministic constraint engine owns all arithmetic and is the only route to the ledger.
6. Uncertain, invalid, unavailable, or unsafe model results route to human review.

This pairing is load-bearing. Rules alone cannot reliably interpret nicknames, sibling intent,
Pidgin, or noisy narration. A language model alone cannot safely own money arithmetic, known-ID
validation, duplicate protection, or irreversible posting. The hybrid design uses each component
only where it is strongest.

The ledger enforces ten invariants, including: allocations cannot exceed the received
transaction; references cannot post twice; only known IDs may be used; unapplied money remains
visible; overpayment becomes explicit credit; reversals preserve history; and every event records
source, evidence, actor, time, and decision path. The model never writes to the ledger.

## 3. Model and system design decisions

The current honest baseline is Qwen3-1.7B in GGUF Q4_K_M format, served by `llama.cpp`. A
Qwen3-0.6B Q4_K_M baseline is included in the C2 comparison plan. The 1.7B model was selected as
the initial candidate because it offers a stronger capacity margin for enterprise and
reconciliation prompts while remaining plausible under the target memory budget. Q4_K_M is the
initial quantization because speed and memory efficiency contribute directly to the challenge
score.

The local runtime is configured for four CPU threads, a 2,048-token context, deterministic
generation, and q8_0 KV caches. App inference uses a bounded prompt and constrained JSON. The
bare-model evaluation uses the chat template embedded in the GGUF, matching the judge-visible
artifact rather than the app's grammar.

The project will retain the zero-shot model if domain fine-tuning does not improve reconciliation
without degrading general enterprise behaviour. If fine-tuning is used, the plan is LoRA
adaptation with a 70% reconciliation / 30% general-enterprise mix, followed by GGUF conversion
and Q4_K_M quantization. Every candidate must be compared against zero-shot on the Bursa gold
set, an ADTC/proxy suite, and a bare-model prompt suite.

Alternatives rejected or deferred:

- Cloud inference was rejected because it violates offline evaluation and creates recurring cost,
  connectivity, and privacy dependencies.
- A model-only ledger was rejected because language models are not trusted for financial
  arithmetic or posting authority.
- A rules-only system remains the fast path but cannot resolve the intended ambiguity classes.
- Larger models and Q5 quantization remain conditional on measured target-hardware trade-offs.
- OCR is deferred and will be cut from Gate 1 unless one printed-transfer workflow is reliable.
- Natural-language reporting, payment APIs, virtual accounts, parent portals, and a full ERP are
  outside the hackathon scope.

## 4. Data and evaluation

The committed authored gold set covers all fourteen evaluation families: exact/name matching,
guardian surname differences, nicknames, sibling splits, instalments, underpayment, overpayment,
fee-item splits, duplicate references, known payers, ambiguous candidates, no-candidate cases,
OCR-style corruption, and narration prompt injection. It includes English and Nigerian Pidgin,
explicit abstention cases, and deterministic financial validation.

Synthetic data generation is deterministic and leak-aware. Training examples can be rendered in
two formats: constrained app JSON and bare conversational answers. Splits are grouped by guardian
and template family to reduce leakage. Public data must be fictional, de-identified, and
license-verified.

One scorecard command runs:

- the Bursa gold suite through the real reconciliation seams;
- a general-capability ADTC/proxy adapter through the GGUF's embedded chat template;
- bare-model enterprise and submission prompts;
- two money-safety gates: zero incorrect automatic posts and 100% duplicate blocking.

Fake-backend CI proves the harness mechanics only. It is not model-quality or performance
evidence. Final claims require real GGUF runs and the official profiler on matching hardware.

## 5. Constraints and reproducibility

The audit target is Ubuntu 22.04, four vCPUs, 8 GB physical RAM, integrated graphics, and a hard
7 GB peak-memory ceiling. Inference must run with no outbound network access. An out-of-memory
condition or sandbox crash disqualifies the submission, and temperatures above 85°C or throttling
incur a penalty.

`download_model.sh` downloads the public primary GGUF without credentials to the exact path in
`metadata.json`. Its optional `--baselines` mode also downloads the 0.6B comparison model. Model
weights are excluded from Git. Run artifacts record the Git commit, model path, model SHA-256,
backend, case records, scorecard, and profiler-supplied performance data.

## 6. Measured benchmarks

No target-hardware number is claimed yet. The following table must be populated only from the
consolidated i5-class, four-thread session:

| Artifact | Gold accuracy | ADTC/proxy | Bare quality | Tokens/s | Peak RSS | Peak temp | Crashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B Q4_K_M zero-shot | Not yet measured | Not yet measured | Not yet reviewed | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Qwen3-0.6B Q4_K_M zero-shot | Not yet measured | Not yet measured | Not yet reviewed | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Final submission GGUF | Not yet selected | Not yet selected | Not yet selected | Not yet selected | Not yet selected | Not yet selected | Not yet selected |

Gate 1 will not be marked ready until ten consecutive participant-mode runs complete with peak
memory below the limit, no thermal throttle, no crash, and reproducible throughput.

## 7. Localisation and limitations

Bursa is naira-native, offline, and grounded in Nigerian school-fee workflows. The authored suite
includes a Nigerian Pidgin ambiguity case where the correct result is to abstain rather than
guess. The African-language bonus is not currently claimed in `metadata.json`; it will be enabled
only after a competent reviewer confirms meaningful model behaviour and the final GGUF
demonstrates it.

Current limitations are explicit: the final GGUF has not been selected, target-hardware
benchmarks are pending, OCR is not shipped, the confidence calibrator remains conservative, and
the local web interface is a demonstration surface rather than a production multi-user system.
