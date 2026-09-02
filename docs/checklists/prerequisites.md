# Bursa competition prerequisites

This checklist deliberately keeps external identity, hardware, credit, and reviewer evidence out
of source code until a registered team member supplies or confirms it.

## Registration and people

- [ ] Registered ADTF team ID added to `metadata.json`.
- [ ] Registered submitter name added to `metadata.json`.
- [ ] Registered submitter email added to `metadata.json`.
- [ ] Submitter GitHub handle verified against the registered member.
- [ ] Team has no more than three registered members.
- [ ] Udutech GPU-credit status recorded: **status not supplied**.
- [x] Competent Yoruba reviewer secured (confirmed by the project owner).
- [ ] Reviewer sign-off recorded for every Yoruba gold, training, bare-prompt, and final-GGUF
      artifact. Keep the reviewer's private contact details outside the public repository.

## Toolchain and hardware

- [x] Official ADTC Profiler pinned; see `toolchain.lock.json`.
- [x] `llama-server` and `llama-bench` installed; build pinned in `toolchain.lock.json`.
- [x] Qwen3 1.7B, 0.6B, and tokenizer upstream revisions and SHA-256 values pinned.
- [x] Gemma 3 1B and Llama 3.2 3B evaluation artifacts pinned by revision, size, checksum,
      and license; weights are not redistributed.
- [x] Internal MMLU enterprise proxy frozen at 200 rows and explicitly labelled non-official.
- [ ] At least 12 GiB is free on the i5 laptop (current implementation preflight blocks below it).
- [ ] At least 80 GiB is free on the rented 24 GB NVIDIA training host.
- [ ] Participant-mode profiler run completed on the physical i5-class laptop at four threads.
- [ ] Ten-run i5 profiling series completed with no crash, OOM, or thermal throttle.
- [ ] Corporate/Enterprise official validation set obtained. No public set is currently
      available; hidden organizer prompts must not be represented by the internal proxy.

Development verification on Apple ARM64 is useful for functionality only and must never be copied
into the target-hardware benchmark tables.
