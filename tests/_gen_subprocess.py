import json
from bursa_eval.synth.generate import generate

cases = generate(base_seed=123, n=20)
print(json.dumps([c.model_dump() for c in cases], sort_keys=True, default=str))
