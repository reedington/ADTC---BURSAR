# ADTC validation data — provenance

## Official set (access-gated)
The ADTC Corporate/Enterprise validation subset is **judge-distributed** (the profiler's
`accuracy.py` scores against the hidden 30% set). It is **not** committed here. When obtained,
drop it in as `data/adtc/official/*.jsonl` using the same shape as the proxy:
`{id, prompt, expected, choices?}`. The adapter loads it unchanged.

## Proxy set (interim C3 forgetting-detector)
The proxy is a **forgetting detector, not a domain-fit measure**. C3 uses the **relative delta**
(post − pre fine-tune) via `regression_delta()`; it never compares absolute proxy scores.

Each proxy dataset's **real license MUST be verified before its rows are committed** (same rule as
the training 30%). Record one row per dataset:

| dataset | source URL | license | verified | commit |
|---|---|---|---|---|
| sample (placeholder) | hand-authored (2 rows) | project-owned | n/a | this repo |

> The `proxy/sample.jsonl` file is a placeholder to keep the adapter runnable in CI. Replace it
> on the i5 with a license-verified general-capability subset (e.g. an MMLU/ARC slice) and fill in
> the row above before using its numbers for C3.
