#!/usr/bin/env python3
"""One-off extraction of AgentDojo benign ground-truth tool outputs (run in a venv with agentdojo==0.1.35, NOT a repo dependency).
Produces datasets/sources/agentdojo_v1.2.2_extract.json, which aase_eval/build_benign_set.py consumes. Re-run only to refresh that pin.
  python -m venv /tmp/venv_ad && /tmp/venv_ad/bin/pip install agentdojo==0.1.35 && cd datasets/sources && /tmp/venv_ad/bin/python ../../aase_eval/extract_agentdojo.py
"""
import json, warnings, re, sys, datetime, hashlib
warnings.filterwarnings("ignore")
from pydantic import BaseModel
import agentdojo
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
SUITE_VERSION = "v1.2.2"
def to_plain(o):
    if isinstance(o, BaseModel): return to_plain(o.model_dump())
    if isinstance(o, dict): return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_plain(x) for x in o]
    if isinstance(o, (datetime.datetime, datetime.date)): return o.isoformat()
    if hasattr(o, "value"): return o.value  # enums
    return o
rows = []
for name, suite in get_suites(SUITE_VERSION).items():
    env = suite.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(suite.tools)
    seen = set()
    for tid, task in suite.user_tasks.items():
        e2 = env.model_copy(deep=True)
        for c in task.ground_truth(env.model_copy(deep=True)):
            r, err = runtime.run_function(e2, c.function, c.args)
            if err is not None or r is None or str(r) == "": continue
            key = (c.function, json.dumps(c.args, default=str, sort_keys=True))
            if key in seen: continue
            seen.add(key)
            plain = to_plain(r)
            text = plain if isinstance(plain, str) else repr(plain)   # dict/list -> Python repr, like InjecAgent's tool responses
            rows.append(dict(suite=name, user_task=tid, user_instruction=task.PROMPT, tool=c.function,
                             args=json.loads(json.dumps(c.args, default=str)), tool_response=text,
                             raw_repr=str(r)[:200], n_chars=len(text)))
json.dump(dict(agentdojo_version=agentdojo.__version__ if hasattr(agentdojo,'__version__') else "0.1.35", suite_version=SUITE_VERSION, rows=rows),
          open("agentdojo_v1.2.2_extract.json", "w"), indent=1)
print("rows", len(rows))
over = [r for r in rows if r["n_chars"] > 4000]; print("over 4000 chars:", [(r["suite"], r["tool"], r["n_chars"]) for r in over])
pw = [r for r in rows if re.search(r"password", r["tool_response"], re.I)]
print("password-mentioning:", [(r["suite"], r["tool"], json.dumps(r["args"])[:50], r["n_chars"]) for r in pw])
