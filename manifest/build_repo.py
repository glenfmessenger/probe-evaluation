#!/usr/bin/env python3
"""Consolidate AASE sources from ~/Downloads into ~/projects/aase. Copy-only; never deletes sources."""
import os, re, sys, csv, json, shutil, hashlib, time
from collections import defaultdict

DL   = os.path.expanduser('~/Downloads')
REPO = os.path.expanduser('~/projects/aase')
SP   = sys.argv[1]
ZIP15 = os.path.join(SP, 'zip15', 'aase')   # extracted files (15)/aase_complete.zip

SKIP_NAMES = {'.DS_Store'}
SKIP_EXT   = ('.pyc',)
prov = []      # provenance rows
conflicts = [] # conflict log entries
dupes = []     # exact-duplicate skips
placed_hash = {}  # sha256 -> repo relpath (first placement)

def sha(p):
    m = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): m.update(c)
    return m.hexdigest()

def mt(p): return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))

def put(src, dest_rel, action, note='', source_tag=''):
    dest = os.path.join(REPO, dest_rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)
    h = sha(src)
    placed_hash.setdefault(h, dest_rel)
    prov.append(dict(repo_path=dest_rel, source=os.path.relpath(src, DL), source_tag=source_tag,
                     sha256=h, size=os.path.getsize(src), source_mtime=mt(src), action=action, note=note))
    return dest_rel

def skip_dup(src, dup_of, source_tag, note=''):
    dupes.append(dict(source=os.path.relpath(src, DL), source_tag=source_tag, sha256=sha(src),
                      identical_to=dup_of, note=note))

def walk(base):
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d not in ('__pycache__',)]
        for f in sorted(fn):
            if f in SKIP_NAMES or f.endswith(SKIP_EXT): continue
            yield os.path.join(dp, f)

if os.path.exists(REPO) and os.listdir(REPO):
    print("refusing: repo dir not empty", file=sys.stderr); sys.exit(1)
os.makedirs(REPO, exist_ok=True)

# ---------------------------------------------------------------- 1. home/glen -> runtime/
HOME = os.path.join(DL, 'home', 'glen')
EXCL_HOME = ('.git-credentials', '.docker/', '.ssh/', '.config/', '.nv/')
excluded_home = []
for p in walk(HOME):
    rel = os.path.relpath(p, HOME)
    if any(rel == e.rstrip('/') or rel.startswith(e) for e in EXCL_HOME):
        excluded_home.append(rel); continue
    if rel.startswith('aag/InjecAgent/.git/'):
        continue
    put(p, os.path.join('runtime', rel), 'copy', 'canonical runtime (GPU VM home)', 'home')
# record InjecAgent upstream
with open(os.path.join(REPO, 'runtime/aag/InjecAgent/UPSTREAM.txt'), 'w') as f:
    f.write("Third-party clone (working tree only, .git/ dropped).\n"
            "origin: https://github.com/uiuc-kang-lab/InjecAgent.git\n"
            "commit: f19c9f2c79a41046eb13c03c51a24c567a8ffa07 (refs/heads/main)\n"
            "cloned on the GPU VM by `git clone` (see runtime/.bash_history line 346).\n")

# ---------------------------------------------------------------- 2. aase-release -> release/ (intact)
REL = os.path.join(DL, 'testingaase', 'aase-release')
for p in walk(REL):
    put(p, os.path.join('release', 'aase-release', os.path.relpath(p, REL)), 'copy', 'packaged release, preserved intact', 'release')
put(os.path.join(DL, 'testingaase', 'aase-release.tar.gz'), 'release/aase-release.tar.gz', 'copy', 'original tarball of the release dir', 'release')

# ---------------------------------------------------------------- 3a. files (15) -> package/
F15 = os.path.join(DL, 'files (15)')
zip_hashes = {}
for p in walk(ZIP15):
    rel = os.path.relpath(p, ZIP15)
    zip_hashes[sha(p)] = rel
    put(p, os.path.join('package', 'aase', rel), 'copy', 'restored from files (15)/aase_complete.zip (structured layout)', 'files15')
put(os.path.join(F15, 'aase_complete.zip'), 'package/aase_complete.zip', 'copy', 'original zip', 'files15')
for f in sorted(os.listdir(F15)):
    p = os.path.join(F15, f)
    if not os.path.isfile(p) or f in SKIP_NAMES or f.endswith(SKIP_EXT) or f == 'aase_complete.zip': continue
    h = sha(p)
    if h in zip_hashes:
        skip_dup(p, 'package/aase/' + zip_hashes[h], 'files15', 'flattened export of a zip member')
    else:
        put(p, os.path.join('package', f), 'copy', 'working-dir script shipped alongside the package (not in zip)', 'files15')
for p in walk(os.path.join(F15, 'mnt')):
    h = sha(p)
    if h in zip_hashes: skip_dup(p, 'package/aase/' + zip_hashes[h], 'files15', 'mnt/user-data/outputs mirror of a zip member')
    else: put(p, os.path.join('package', os.path.relpath(p, F15)), 'copy', 'mnt/ file not in zip', 'files15')

# ---------------------------------------------------------------- 3b. aase 2 (+ aase-vllm-0.1.1) -> vllm/
A2  = os.path.join(DL, 'aase 2')
V011 = os.path.join(DL, 'aase-vllm-0.1.1')
for p in walk(A2):
    rel = os.path.relpath(p, A2)
    if rel == 'setup.py':
        put(p, 'archive/superseded/aase_2/setup.py', 'archive', 'version=0.1.0; superseded by aase-vllm-0.1.1/setup.py (version bump, later mtime)', 'aase2')
        continue
    put(p, os.path.join('vllm', rel), 'copy', 'aase 2 is a superset of aase-vllm-0.1.1', 'aase2')
put(os.path.join(V011, 'setup.py'), 'vllm/setup.py', 'copy', 'version=0.1.1, only diff vs aase 2/setup.py is the version line; later mtime', 'vllm011')
for p in walk(V011):
    rel = os.path.relpath(p, V011)
    if rel == 'setup.py': continue
    h = sha(p)
    a2p = os.path.join(A2, rel)
    if os.path.exists(a2p) and sha(a2p) == h:
        skip_dup(p, os.path.join('vllm', rel), 'vllm011', 'identical to aase 2 copy')
    else:
        put(p, os.path.join('archive/superseded/aase-vllm-0.1.1', rel), 'archive', 'differs from aase 2 copy; aase 2 version is newer/larger', 'vllm011')
        conflicts.append(dict(family=rel, winner=os.path.join('vllm', rel), winner_src='aase 2/' + rel,
                              losers=['aase-vllm-0.1.1/' + rel],
                              resolution='aase 2 copy kept (newer mtime, multi-model ROC script / no __MACOSX junk); 0.1.1 copy archived',
                              evidence='diff + mtime; not in bash history (post-VM work)'))
conflicts.append(dict(family='setup.py (vLLM package)', winner='vllm/setup.py', winner_src='aase-vllm-0.1.1/setup.py',
                      losers=['aase 2/setup.py'], resolution='0.1.1 file kept: identical except version "0.1.1" vs "0.1.0", and 19 min newer',
                      evidence='diff + mtime; not in bash history'))

# ---------------------------------------------------------------- 3c. loose Downloads root scripts
def norm(b): return re.sub(r' \(\d+\)(?=\.\w+$)', '', b)
FAMILY = [
    (r'^(benchmark_comparison|defense_comparison|llama_guard3_defense_comparison|mlcommons_benchmark|benchmark_latency|generate_roc_curves)', 'scripts/benchmarks'),
    (r'^(phase1_validation|gemma7b_validation|llama3_validation|small_model_validation|quantization_validation|run_validation_llama)', 'scripts/validation'),
    (r'^af_', 'scripts/af'),
    (r'^(ahd_|simpleqa_ahd|layer_sweep_methodology)', 'scripts/ahd'),
    (r'^apc_', 'scripts/apc'),
    (r'^(agent_gating|phase1_live_agent|injecagent)', 'scripts/aag'),
    (r'^(ihe_)', 'scripts/ihe'),
    (r'^(pretrain_|recalibrate_|retrain_|tune_|package_release|fix_aag|eval_harmbench)', 'scripts/package_tools'),
    (r'^request_a100_quota', 'scripts/infra'),
    (r'\.ipynb$', 'notebooks'),
]
UNRELATED = {'btc_simulation.py': 'crypto price simulation, not AASE',
             'w3.py': 'web3/Base-chain script with an embedded API key, not AASE (not copied)'}
def family(b):
    for rx, d in FAMILY:
        if re.search(rx, b): return d
    return 'scripts/misc'

# index of repo files by normalized basename (for name conflicts)
repo_index = defaultdict(list)
for r in prov:
    repo_index[norm(os.path.basename(r['repo_path']))].append(r)

loose = [f for f in sorted(os.listdir(DL)) if os.path.isfile(os.path.join(DL, f)) and f.endswith(('.py', '.sh', '.ipynb'))]
groups = defaultdict(list)
for f in loose: groups[norm(f)].append(f)
unrelated_log = []
for nb, fs in sorted(groups.items()):
    if nb in UNRELATED:
        for f in fs: unrelated_log.append((f, UNRELATED[nb]))
        continue
    fs_info = [(f, sha(os.path.join(DL, f)), os.path.getmtime(os.path.join(DL, f))) for f in fs]
    canon = [r for r in repo_index.get(nb, []) if r['source_tag'] in ('home', 'files15', 'aase2', 'vllm011') and r['action'] == 'copy']
    if canon:
        # a canonical copy exists in runtime/ or package/ or vllm/
        win = canon[0]
        losers = []
        for f, h, m in fs_info:
            if h == win['sha256'] or h in placed_hash:
                skip_dup(os.path.join(DL, f), placed_hash.get(h, win['repo_path']), 'loose', 'identical to canonical copy')
            else:
                put(os.path.join(DL, f), os.path.join('archive/superseded/loose', f), 'archive',
                    f'differs from canonical {win["repo_path"]}', 'loose')
                losers.append(f)
        if losers:
            conflicts.append(dict(family=nb, winner=win['repo_path'], winner_src=win['source'], losers=losers,
                                  resolution='canonical copy wins (higher-priority source); loose drafts archived', evidence='__AUTO__'))
        continue
    # no canonical copy: pick latest mtime among loose variants
    fs_info.sort(key=lambda t: t[2])
    winner = fs_info[-1]
    dest_dir = family(nb)
    put(os.path.join(DL, winner[0]), os.path.join(dest_dir, nb), 'copy',
        'loose script; ' + ('latest of %d variants by mtime' % len(fs_info) if len(fs_info) > 1 else 'only copy'), 'loose')
    if len(fs_info) > 1:
        losers = []
        for f, h, m in fs_info[:-1]:
            if h == winner[1]:
                skip_dup(os.path.join(DL, f), os.path.join(dest_dir, nb), 'loose', 'identical to chosen variant')
            else:
                put(os.path.join(DL, f), os.path.join('archive/superseded/loose', f), 'archive', 'earlier download variant', 'loose'); losers.append(f)
        conflicts.append(dict(family=nb, winner=os.path.join(dest_dir, nb), winner_src=winner[0], losers=losers,
                              resolution='latest mtime among browser-download variants wins (no canonical copy, not in bash history)',
                              evidence='mtime only'))

# ---------------------------------------------------------------- write provenance
os.makedirs(os.path.join(REPO, 'manifest'), exist_ok=True)
with open(os.path.join(REPO, 'manifest/provenance.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(prov[0].keys())); w.writeheader(); w.writerows(sorted(prov, key=lambda r: r['repo_path']))
with open(os.path.join(REPO, 'manifest/deduplicated.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(dupes[0].keys())); w.writeheader(); w.writerows(dupes)
shutil.copy2(os.path.join(SP, 'manifest.csv'), os.path.join(REPO, 'manifest/sources_sha256.csv'))
json.dump(dict(conflicts=conflicts, excluded_home=excluded_home, unrelated=unrelated_log), open(os.path.join(SP, 'build_result.json'), 'w'), indent=1)
print(f"placed {len(prov)} files, skipped {len(dupes)} exact duplicates, {len(conflicts)} conflicts, excluded from home: {excluded_home}, unrelated: {unrelated_log}")
