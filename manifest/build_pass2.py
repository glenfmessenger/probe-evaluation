#!/usr/bin/env python3
"""Second pass: copy AASE material from the pass-two sources into ~/projects/aase (branch second-pass).
Copy-only, hash-deduped against manifest/provenance.csv. Never touches ~/Downloads."""
import os, sys, csv, re, time, shutil, hashlib, glob
DL=os.path.expanduser('~/Downloads'); REPO=os.path.expanduser('~/projects/aase'); SP=sys.argv[1]
ZIPS=os.path.join(SP,'p2zips')
prov_path=os.path.join(REPO,'manifest/provenance.csv')
prov=list(csv.DictReader(open(prov_path)))
have={r['sha256']:r['repo_path'] for r in prov}
def sha(p):
    m=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): m.update(c)
    return m.hexdigest()
def mt(p): return time.strftime('%Y-%m-%d %H:%M',time.localtime(os.path.getmtime(p)))
new=[]; skipped=[]
def src_label(p):
    if p.startswith(ZIPS): return '[zip]'+os.path.relpath(p,ZIPS).replace('_nested/','!/')
    return os.path.relpath(p,DL)
def put(src,dest_rel,tag,note=''):
    h=sha(src)
    if h in have:
        skipped.append(dict(source=src_label(src),source_tag=tag,sha256=h,identical_to=have[h],note=note)); return None
    dest=os.path.join(REPO,dest_rel); os.makedirs(os.path.dirname(dest),exist_ok=True)
    if os.path.exists(dest): raise SystemExit('dest exists: '+dest_rel)
    shutil.copy2(src,dest); have[h]=dest_rel
    new.append(dict(repo_path=dest_rel,source=src_label(src),source_tag='pass2:'+tag,sha256=h,size=os.path.getsize(src),source_mtime=mt(src),action='copy',note=note))
    return dest_rel
def walk(base):
    for dp,dn,fn in os.walk(base):
        dn[:]=[d for d in dn if d!='__pycache__']
        for f in sorted(fn):
            if f=='.DS_Store' or f.endswith('.pyc'): continue
            yield os.path.join(dp,f)
def stamp(p): return time.strftime('%Y-%m-%d_%H%M',time.localtime(os.path.getmtime(p)))

# ---- paper-src/final : the submitted revision (AASE PeerJ Response, 2026-04-17) + bib + figures + code supplement
PR=os.path.join(DL,'AASE PeerJ Response'); PC=os.path.join(DL,'AASE PeerJ CS Response'); FIG=os.path.join(DL,'AASE FOR PEERJ CS')
put(os.path.join(PR,'aase_revised.tex'),'paper-src/final/aase_revised.tex','peerj-response','latest revision (2026-04-17); differs from 04-08 copy only in the AI-use statement')
put(os.path.join(PR,'aase_revised.pdf'),'paper-src/final/aase_revised.pdf','peerj-response','compiled PDF of the final revision')
put(os.path.join(PR,'AASE_AI_Code_Supplemental.pdf'),'paper-src/final/AASE_AI_Code_Supplemental.pdf','peerj-response','supplemental (Google Docs export)')
put(os.path.join(PR,'aase_revised.tex.zip'),'paper-src/final/aase_revised.tex.zip','peerj-response','zip of the same tex (kept as submitted artefact)')
put(os.path.join(PC,'aase.bib'),'paper-src/final/aase.bib','peerj-cs-response','bib shipped with the 04-08 response bundle (tex uses embedded thebibliography)')
for f in ['figure1_layer_sweeps.pdf','figure3_latency.pdf','figure4_roc_curves.pdf']:
    put(os.path.join(FIG,f),'paper-src/final/'+f,'aase-for-peerj-cs','figure bundle whose hashes match the includegraphics targets')
for f in ['README.md','LICENSE','requirements.txt','probe_training.py','vllm_integration.py','datasets/aag_training_data.json','datasets/af_training_data.json','datasets/apc_training_data.json']:
    put(os.path.join(PC,f),'paper-src/final/code-supplement/'+f,'peerj-cs-response','code+data supplement accompanying the 04-08 response')
# ---- paper-src/revisions : every earlier AASE-lineage tex, date-stamped
lineage=[('AASE PeerJ CS Response/aase_revised.tex','peerj-cs-response'),('aase_revised.tex','root'),('aase_peerj.tex','root'),('aase_paper_mdpi_v0.2.tex','root'),
 ('files (32)/aase_paper_mdpi_v0.32 crap.tex','files32'),('activation_safety_paper_v0.31.tex','root'),('activation_safety_paper_v0.30.tex','root'),('activation_safety_paper_v0.29.tex','root'),('activation_safety_paper_v0.28.tex','root'),
 ('activation safety paper/activation_safety_paper_revised.tex','activation-safety-paper'),('files (20)/activation_safety_paper_revised.tex','files20'),('files (20)/aase_workshop_paper.tex','files20'),
 ('activation_safety_paper_revised (1).tex','root'),('activation_safety_paper_revised.tex','root'),('aase_paper_bundle/activation_safety_paper.tex','aase_paper_bundle'),('LATEST/activation_safety_paper v0.26.tex','LATEST'),
 ('activation_safety_paper (7).tex','root'),('activation_safety_paper (6).tex','root'),('files (10)/activation_safety_paper.tex','files10'),('activation_safety_paper (5).tex','root'),('activation_safety_paper (4).tex','root'),
 ('activation_safety_paper (3).tex','root'),('activation_safety_paper (2).tex','root'),('activation_safety_paper (1).tex','root'),('activation_safety_paper.tex','root'),('cognitive_depth_paper (1).tex','root'),('cognitive_depth_paper.tex','root'),
 ('AHD_Paper_v2.tex','root'),('ahd_package/ahd_paper.tex','ahd_package')]
for rel,tag in lineage:
    p=os.path.join(DL,rel); put(p,f'paper-src/revisions/{stamp(p)}__{os.path.basename(rel).replace(" ","_")}',tag,'AASE-lineage tex, date-stamped from source mtime')
# ---- paper-src/figures : every distinct AASE figure file (referenced names only) + AHD paper figures
figs=glob.glob(os.path.join(DL,'figure1_layer_sweeps*'))+glob.glob(os.path.join(DL,'figure2_ahd_scale*'))+glob.glob(os.path.join(DL,'figure3_latency*'))+glob.glob(os.path.join(DL,'figure4_roc_curves*'))+ \
     [os.path.join(DL,f) for f in ['fig1_layer_sweep.png','fig2_score_distributions.png','fig3_auc_comparison.png','fig4_architecture.png','fig3-layer-sweep.svg']]+glob.glob(os.path.join(DL,'files (8)','*.png'))+ \
     glob.glob(os.path.join(DL,'files (11)','*.pdf'))+glob.glob(os.path.join(DL,'aase_paper_bundle','figure*'))+glob.glob(os.path.join(DL,'AASE Conolidated Data','figure*'))+glob.glob(os.path.join(DL,'files (32)','figure*'))+[os.path.join(DL,'paper','cognitive_depth_figure.pdf'),os.path.join(DL,'paper','cognitive_depth_figure.png'),os.path.join(DL,'paper','cognitive_depth_auc.png')]
for p in sorted(set(figs),key=lambda x:os.path.getmtime(x)):
    if not os.path.isfile(p) or 'activation_safety_paper' in p: continue
    b=os.path.basename(p); b=re.sub(r' \(\d+\)','',b)
    put(p,f'paper-src/figures/{stamp(p)}__{b}','figures','figure variant, date-stamped; identical copies elsewhere skipped')
# ---- paper-src/pdfs : compiled paper PDFs
for rel in ['files (10)/activation_safety_paper.pdf','files (11)/activation_safety_paper.pdf','aase_paper_bundle/activation_safety_paper.pdf','files (8)/AHD_Paper_Final.pdf','ahd_package/ahd_paper.pdf','files (32)/aase_paper_mdpi_v0.32 crap.pdf','AASE PeerJ CS Response/aase_revised.pdf','files (20)/aase_vllm_prd.docx','files (20)/DELIVERABLES_README.md']:
    p=os.path.join(DL,rel)
    if os.path.exists(p): put(p,f'paper-src/pdfs/{stamp(p)}__{os.path.basename(rel).replace(" ","_")}','pdfs','compiled/exported paper artefact')
# ---- paper-src/ahd_package : the AHD paper bundle
for p in walk(os.path.join(DL,'ahd_package')):
    if p.endswith(('.tex','.pdf')): continue
    put(p,'paper-src/ahd_package/'+os.path.relpath(p,os.path.join(DL,'ahd_package')),'ahd_package','AHD paper code bundle')
# ---- experiments/paper-scripts : ~/Downloads/paper (layer sweeps, quantization, round_2, llama)
for p in walk(os.path.join(DL,'paper')):
    if os.path.basename(p) in ('cognitive_depth_figure.pdf','cognitive_depth_figure.png','cognitive_depth_auc.png'): continue
    put(p,'experiments/paper-scripts/'+os.path.relpath(p,os.path.join(DL,'paper')),'paper','paper-ready experiment scripts and sweep results (2026-01-01)')
# ---- experiments/consolidated-data : AASE Conolidated Data (zips extracted, deduped)
CD=os.path.join(DL,'AASE Conolidated Data')
for p in walk(CD):
    if p.endswith('.zip') or os.path.basename(p).startswith('figure'): continue
    put(p,'experiments/consolidated-data/'+os.path.basename(p),'consolidated-data','context prompt')
for zdir,dest in [('AASE_Conolidated_Data_aase_unified_package','experiments/consolidated-data/aase_unified'),('AASE_Conolidated_Data_AAG_PoC','experiments/consolidated-data/AAG_PoC'),('AASE_Conolidated_Data_AF_PoC','experiments/consolidated-data/AF_PoC'),('AASE_Conolidated_Data_APC_PoC','experiments/consolidated-data/APC_PoC')]:
    base=os.path.join(ZIPS,zdir)
    for p in walk(base):
        if p.endswith('.zip'): continue
        rel=os.path.relpath(p,base).replace('aase_unified/','',1).replace('aag_consolidated_nested/aag_consolidated/','aag_consolidated/')
        put(p,dest+'/'+rel,'consolidated-data','extracted from '+zdir.replace('AASE_Conolidated_Data_','')+'.zip')
for z in glob.glob(os.path.join(CD,'*.zip')): put(z,'experiments/consolidated-data/_zips/'+os.path.basename(z),'consolidated-data','original zip (contents extracted alongside)')
# ---- experiments/vllm-prototype : files (13)
for p in walk(os.path.join(DL,'files (13)')): put(p,'experiments/vllm-prototype/'+os.path.relpath(p,os.path.join(DL,'files (13)')),'files13','vLLM hook feasibility scripts 01–04')
# ---- packaging : AASE packaging
PK=os.path.join(DL,'AASE packaging')
for p in walk(PK):
    if p.endswith('.zip'): put(p,'packaging/upload/_zips/'+os.path.basename(p),'aase-packaging','original upload zip'); continue
    put(p,'packaging/'+os.path.relpath(p,PK),'aase-packaging','packaging-era script / upload staging')
for zdir in ['AASE_packaging_upload_aase','AASE_packaging_upload_aase_calibrate','AASE_packaging_upload_scripts']:
    base=os.path.join(ZIPS,zdir)
    for p in walk(base): put(p,'packaging/upload/_zips/'+zdir.replace('AASE_packaging_upload_','')+'/'+os.path.relpath(p,base),'aase-packaging','extracted zip member')
# ---- AASE/ dir (all 12 already in repo -> will all be skipped, logged)
for p in walk(os.path.join(DL,'AASE')): put(p,'experiments/AASE-snapshot/'+os.path.relpath(p,os.path.join(DL,'AASE')),'AASE','PoC snapshot dir')
# ---- docs/product : root docs + new prompt
docs=[f for f in os.listdir(DL) if os.path.isfile(os.path.join(DL,f)) and re.match(r'^(AASE_|AF_|APC_|AHD_|aase_faq|INTEGRATION_GUIDE|IDF_v4_content|aag_llmd_compatibility|af_context_prompt|af_edge_mobile|af_lessons|af_technical|af_architecture_diagram|activation_fingerprinting_diagram|activation_fingerprinting_economic|aase_ams_gtm)',f) and f.endswith(('.md','.docx','.html')) and not f.startswith('AASE_Architecture')] + ['AASE_Architecture_Diagram.html','AASE FAQ.pdf']
for f in sorted(set(docs)):
    p=os.path.join(DL,f)
    if os.path.exists(p): put(p,'docs/product/'+f.replace(' ','_'),'root-docs','product/positioning doc from Downloads root')
NP=os.path.join(DL,'new prompt')
for p in walk(NP): put(p,'docs/product/new_prompt/'+os.path.basename(p),'new-prompt','context-prompt bundle (2025-12-27)')
# ---- write provenance
with open(prov_path,'a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(prov[0].keys())); w.writerows(new)
with open(os.path.join(REPO,'manifest/pass2_skipped_duplicates.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['source','source_tag','sha256','identical_to','note']); w.writeheader(); w.writerows(skipped)
print(f'copied {len(new)} new files; skipped {len(skipped)} hash-duplicates')
from collections import Counter; print(Counter(r["source_tag"] for r in new))
