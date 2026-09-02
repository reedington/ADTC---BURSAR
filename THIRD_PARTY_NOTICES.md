# Third-party notices

Bursa is distributed under GPL-3.0-only. Third-party components and model weights retain their
own licenses. Inclusion here is attribution, not a relicensing of those components.

| Component | Purpose | Upstream | License |
|---|---|---|---|
| ADTC 2026 submission template | Required repository structure | https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template | GPL-3.0 |
| Qwen3-1.7B / Qwen3-0.6B | Local language-model baselines | https://huggingface.co/Qwen | Apache-2.0 |
| Gemma 3 1B IT GGUF | Evaluation control only; not redistributed | https://huggingface.co/unsloth/gemma-3-1b-it-GGUF | Gemma Terms |
| Llama 3.2 3B Instruct GGUF | Evaluation control only; not redistributed | https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF | Llama 3.2 Community License |
| OpenAssistant/oasst1 | Pinned general-enterprise training pairs | https://huggingface.co/datasets/OpenAssistant/oasst1 | Apache-2.0 |
| cais/mmlu | Internal enterprise regression proxy only | https://huggingface.co/datasets/cais/mmlu | MIT |
| Unsloth | Reproducible QLoRA, merge, and GGUF export | https://github.com/unslothai/unsloth | Apache-2.0 |
| llama.cpp | GGUF CPU inference runtime | https://github.com/ggml-org/llama.cpp | MIT |
| Pydantic | Runtime schema validation | https://github.com/pydantic/pydantic | MIT |
| PyYAML | Gold-case YAML loading | https://github.com/yaml/pyyaml | MIT |
| FastAPI | Offline local web interface | https://github.com/fastapi/fastapi | MIT |
| Jinja | HTML templates | https://github.com/pallets/jinja | BSD-3-Clause |
| Uvicorn | Local ASGI server | https://github.com/encode/uvicorn | BSD-3-Clause |
| pytest | Test runner | https://github.com/pytest-dev/pytest | MIT |
| Hypothesis | Property-based tests | https://github.com/HypothesisWorks/hypothesis | MPL-2.0 |

Exact revisions are recorded in `toolchain.lock.json` and generated data/model manifests. The
final model card must be added before Gate 1. Data with unclear or incompatible licensing must
not be used.
