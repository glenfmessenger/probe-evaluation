"""InjecAgent loaders (fail-loud, real data only).

On-disk schema of the upstream repository (uiuc-kang-lab/InjecAgent, commit
f19c9f2), verified against ``runtime/aag/InjecAgent/data``:

``attacker_cases_dh.jsonl`` (30 rows), ``attacker_cases_ds.jsonl`` (32 rows) —
one JSON object per line, UTF-8, no BOM, keys::

    "Attacker Tools"          list[str]   (serialised as a Python-repr string in test cases)
    "Modifed"                 str         (sic — upstream typo; not used)
    "Attacker Instruction"    str         <- the injection text
    "Expected Achievements"   str
    "Attack Type"             str         ("Physical Harm", "Financial Harm", "Data Security Harm", ...)

``user_cases.jsonl`` (17 rows) — the benign user/tool contexts, keys::

    "User Tool"               str
    "User Instruction"        str
    "Tool Parameters"         str
    "Tool Response Template"  str         (contains the literal placeholder "<Attacker Instruction>")
    "Thought"                 str
    "Level"                   str         ("High"/"Low")

``test_cases_{dh,ds}_{base,enhanced}.json`` — JSON lists (510 + 544 = 1,054
cases per setting) formed by pairing every attacker case with every user case;
each element carries all attacker-case keys plus all user-case keys plus
``"Tool Response"`` (the template with the injection inserted; in the
``enhanced`` setting the injection is prefixed with the "IMPORTANT!!!" hacking
prompt).  All 1,054 ``Tool Response`` strings are distinct.

Surface-form quirk (verified on disk): every one of the 1,054 ``Tool Response``
values is wrapped in a pair of literal double quotes (``"{'product_details': ...}"``)
whereas none of the 17 ``Tool Response Template`` values is.  Left as-is, the
leading ``"`` would be a perfect label cue for a probe.  ``load_test_cases``
therefore strips exactly one wrapping quote pair and unescapes the ``\\"`` sequences
that wrapper introduced (124 of them across the base set) by default
(``strip_wrapping_quotes=True``) and keeps the raw string in ``tool_response_raw``.

The historical defect fixed here: the vLLM-era scripts looked up
``case.get("attacker_instruction")`` (snake_case), which never matched, and then
silently substituted synthetic prompts.  This module accepts the real column
name first, the snake_case alias second, and raises on anything else.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .errors import DataLoadError, empty_error, missing_file_error, schema_error

# canonical name -> accepted aliases (real column first; snake_case kept for forward compat)
ATTACK_KEYS = {
    "attacker_instruction": ("Attacker Instruction", "attacker_instruction"),
    "attacker_tools": ("Attacker Tools", "attacker_tools"),
    "attack_type": ("Attack Type", "attack_type"),
    "expected_achievements": ("Expected Achievements", "expected_achievements"),
}
USER_KEYS = {
    "user_tool": ("User Tool", "user_tool"),
    "user_instruction": ("User Instruction", "user_instruction"),
    "tool_parameters": ("Tool Parameters", "tool_parameters"),
    "tool_response_template": ("Tool Response Template", "tool_response_template"),
    "thought": ("Thought", "thought"),
    "level": ("Level", "level"),
}
TEST_CASE_KEYS = {**ATTACK_KEYS, **USER_KEYS, "tool_response": ("Tool Response", "tool_response")}

ATTACKER_FILES = {"dh": "attacker_cases_dh.jsonl", "ds": "attacker_cases_ds.jsonl"}
USER_FILE = "user_cases.jsonl"
PLACEHOLDER = "<Attacker Instruction>"
DEFAULT_PROMPT_TEMPLATE = "User: {user_instruction}\nTool Result: {tool_response}\nAssistant:"


def _pick(record: Dict[str, Any], canonical: str, aliases: Iterable[str], path, idx: int):
    for k in aliases:
        if k in record:
            return record[k]
    raise schema_error(path, list(aliases), record.keys(), context=f"record {idx}: missing '{canonical}'")


def _normalise(record: Dict[str, Any], keymap: Dict[str, tuple], path, idx: int, required: Iterable[str]) -> Dict[str, Any]:
    out = {}
    for canonical, aliases in keymap.items():
        if canonical in required:
            out[canonical] = _pick(record, canonical, aliases, path, idx)
        else:
            out[canonical] = next((record[k] for k in aliases if k in record), None)
    return out


def _read_jsonl(path: Path, what: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise missing_file_error(path, what, "clone https://github.com/uiuc-kang-lab/InjecAgent.git and point injecagent.dir at it")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataLoadError(f"{what}: line {n} of {path} is not valid JSON: {e}") from e
    if not rows:
        raise empty_error(path, what)
    return rows


def _read_json_list(path: Path, what: str) -> List[Dict[str, Any]]:
    if not path.exists():
        raise missing_file_error(path, what, "expected InjecAgent/data/test_cases_{dh,ds}_{base,enhanced}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise empty_error(path, what)
    return data


def load_attacker_cases(injecagent_dir) -> List[Dict[str, Any]]:
    """The 62 attacker cases (30 direct-harm + 32 data-stealing)."""
    base = Path(injecagent_dir) / "data"
    out = []
    for split, fname in ATTACKER_FILES.items():
        path = base / fname
        for i, rec in enumerate(_read_jsonl(path, f"InjecAgent attacker cases ({split})")):
            norm = _normalise(rec, ATTACK_KEYS, path, i, required=("attacker_instruction", "attack_type"))
            if not str(norm["attacker_instruction"]).strip():
                raise DataLoadError(f"record {i} of {path} has an empty 'Attacker Instruction'")
            norm.update(split=split, source_file=str(path), case_id=f"attacker_{split}_{i}")
            out.append(norm)
    return out


def load_user_cases(injecagent_dir) -> List[Dict[str, Any]]:
    """The 17 benign user cases (real tool contexts, no injection)."""
    path = Path(injecagent_dir) / "data" / USER_FILE
    out = []
    for i, rec in enumerate(_read_jsonl(path, "InjecAgent user cases")):
        norm = _normalise(rec, USER_KEYS, path, i, required=("user_tool", "user_instruction", "tool_response_template"))
        norm.update(source_file=str(path), case_id=f"user_{i}")
        out.append(norm)
    return out


def _strip_wrapping_quotes(s: str) -> str:
    """Undo upstream's wrapper: remove the outer ``"..."`` and unescape the ``\\"`` it introduced.

    Verified on disk: for all 1,054 cases in both settings the result equals the user case's
    ``Tool Response Template`` with the injection inserted at the placeholder (i.e. exactly the
    scaffold the benign cases are built from).  Without the unescape, 124 inner double quotes
    (e.g. ``\\"Doctor's Appointment\\"``) would remain as a class-correlated surface cue.
    """
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def load_test_cases(injecagent_dir, setting: str = "base", strip_wrapping_quotes: bool = True) -> List[Dict[str, Any]]:
    """The 1,054 paired test cases for a setting ('base' or 'enhanced').

    ``strip_wrapping_quotes`` removes the label-correlated ``"..."`` wrapper
    around every injected tool response (see module docstring)."""
    if setting not in ("base", "enhanced"):
        raise DataLoadError(f"InjecAgent setting must be 'base' or 'enhanced', got {setting!r}")
    base = Path(injecagent_dir) / "data"
    out = []
    for split in ("dh", "ds"):
        path = base / f"test_cases_{split}_{setting}.json"
        for i, rec in enumerate(_read_json_list(path, f"InjecAgent test cases ({split}, {setting})")):
            norm = _normalise(rec, TEST_CASE_KEYS, path, i,
                              required=("attacker_instruction", "user_instruction", "user_tool", "tool_response", "attack_type"))
            if not str(norm["tool_response"]).strip():
                raise DataLoadError(f"record {i} of {path} has an empty 'Tool Response'")
            norm["tool_response_raw"] = norm["tool_response"]
            if strip_wrapping_quotes:
                norm["tool_response"] = _strip_wrapping_quotes(norm["tool_response"])
            norm.update(split=split, setting=setting, source_file=str(path), case_id=f"test_{split}_{setting}_{i}")
            out.append(norm)
    return out


def format_prompt(user_instruction: str, tool_response: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(user_instruction=user_instruction, tool_response=tool_response)


def build_injection_prompts(test_cases: List[Dict[str, Any]], template: str = DEFAULT_PROMPT_TEMPLATE) -> List[Dict[str, Any]]:
    """One prompt per real test case: the real user instruction + the real injected tool response."""
    return [
        {
            "id": c["case_id"],
            "label": 1,
            "prompt": format_prompt(c["user_instruction"], c["tool_response"], template),
            "attack_type": c["attack_type"],
            "attacker_instruction": c["attacker_instruction"],
            "user_tool": c["user_tool"],
            "split": c["split"],
            "setting": c["setting"],
        }
        for c in test_cases
    ]


def benign_tool_response(tool_response_template: str, placeholder_fill: str = "") -> str:
    """Real tool response with the injection slot removed.

    The upstream template contains the literal ``<Attacker Instruction>`` where
    the injection would go.  For the benign condition we replace it with
    ``placeholder_fill`` (default: empty string, i.e. the slot is simply absent).
    """
    if PLACEHOLDER not in tool_response_template:
        raise DataLoadError(f"user-case tool response template does not contain the {PLACEHOLDER!r} placeholder: {tool_response_template[:120]!r}")
    return tool_response_template.replace(PLACEHOLDER, placeholder_fill)


def build_benign_prompts(user_cases: List[Dict[str, Any]], placeholder_fill: str = "",
                         template: str = DEFAULT_PROMPT_TEMPLATE) -> List[Dict[str, Any]]:
    return [
        {
            "id": c["case_id"],
            "label": 0,
            "prompt": format_prompt(c["user_instruction"], benign_tool_response(c["tool_response_template"], placeholder_fill), template),
            "user_tool": c["user_tool"],
            "level": c.get("level"),
        }
        for c in user_cases
    ]


def _subsample(items: List[Dict[str, Any]], count: Optional[int], seed: int, what: str) -> List[Dict[str, Any]]:
    if count is None or count >= len(items):
        return items
    if count <= 0:
        raise DataLoadError(f"{what}: sample count must be positive, got {count}")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), count))
    return [items[i] for i in idx]


def load_injecagent_eval_set(cfg: Dict[str, Any], injecagent_dir=None) -> Dict[str, Any]:
    """Build the AAG evaluation set from real InjecAgent data according to config.

    ``cfg`` is the ``injecagent`` section of eval_config.yaml.  Returns a dict
    with ``injections`` (list of prompt records, label 1), ``benign`` (label 0)
    and a ``summary``.  Raises DataLoadError on any problem; never substitutes.
    """
    from .config import resolve_path

    d = Path(injecagent_dir) if injecagent_dir else resolve_path(cfg["dir"])
    if not (d / "data").is_dir():
        raise missing_file_error(d / "data", "InjecAgent data directory")
    settings = cfg.get("settings")
    if settings is None:  # backwards compatibility with the single-setting key
        settings = [cfg.get("setting", "base")]
    if isinstance(settings, str):
        settings = [settings]
    if not settings or any(s not in ("base", "enhanced") for s in settings):
        raise DataLoadError(f"injecagent.settings must be a non-empty subset of ['base', 'enhanced'], got {settings!r}")
    template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
    seed = int(cfg.get("sample_seed", 0))

    test_cases = []
    for s in settings:
        test_cases += load_test_cases(d, s, bool(cfg.get("strip_wrapping_quotes", True)))
    injections = build_injection_prompts(test_cases, template)   # every record carries its "setting"
    injections = _subsample(injections, cfg.get("max_injections"), seed, "injections")

    benign_source = cfg.get("benign_source", "aag_benign_eval")
    if benign_source == "injecagent_user17":
        user_cases = load_user_cases(d)
        benign = build_benign_prompts(user_cases, cfg.get("benign_placeholder_fill", ""), template)
        for b in benign:
            b.update(subset="injecagent_user", source="injecagent:user_cases.jsonl", hard_negative=False, hard_negative_kind=None)
        n_benign_available = len(user_cases)
    elif benign_source == "aag_benign_eval":
        from .benign_set import DEFAULT_PATH, load_benign_set, to_prompt_records
        from .config import resolve_path as _rp
        bpath = _rp(cfg["benign_set_path"]) if cfg.get("benign_set_path") else DEFAULT_PATH
        brecs = load_benign_set(bpath, cfg.get("benign_subsets"))
        benign = to_prompt_records(brecs, template)
        n_benign_available = len(brecs)
    else:
        raise DataLoadError(f"unknown injecagent.benign_source {benign_source!r}; valid: aag_benign_eval, injecagent_user17")
    benign = _subsample(benign, cfg.get("benign_sample_count"), seed, "benign")

    if not injections or not benign:
        raise DataLoadError("InjecAgent eval set is empty after sampling")
    prompts = [r["prompt"] for r in injections]
    if len(set(prompts)) != len(prompts):
        raise DataLoadError("InjecAgent injection prompts are not distinct — the data path is broken")

    from collections import Counter as _Counter
    summary = {
        "injecagent_dir": str(d),
        "settings": list(settings),
        "n_injections": len(injections),
        "n_injections_by_setting": dict(_Counter(r["setting"] for r in injections)),
        "n_injections_available": len(test_cases),
        "n_injections_available_by_setting": dict(_Counter(c["setting"] for c in test_cases)),
        "n_benign": len(benign),
        "n_benign_available": n_benign_available,
        "benign_source": benign_source,
        "benign_subsets": dict(__import__("collections").Counter(b.get("subset", "unknown") for b in benign)),
        "n_benign_hard_negative": sum(1 for b in benign if b.get("hard_negative")),
        "n_distinct_attacker_instructions": len({c["attacker_instruction"] for c in test_cases}),
        "n_user_tools": len({c["user_tool"] for c in test_cases}),
        "prompt_template": template,
        "benign_placeholder_fill": cfg.get("benign_placeholder_fill", ""),
        "strip_wrapping_quotes": bool(cfg.get("strip_wrapping_quotes", True)),
    }
    return {"injections": injections, "benign": benign, "summary": summary}
