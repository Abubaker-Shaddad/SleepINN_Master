"""Build self-contained HTML viewers from the public databank and answer files."""
import base64
import gzip
import json
from pathlib import Path
import pandas as pd
from .io import project_root, read_jsonl, answer_frame

STYLE = '''body{font:16px system-ui;background:#f4f7fa;color:#142b3b;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:32px}a{color:#176082}nav{display:flex;gap:24px}label{display:inline-block;margin:8px 15px 8px 0}select,input{padding:9px;border:1px solid #abc;border-radius:6px;max-width:95%}input{width:360px}article{background:white;border:1px solid #dce4ea;border-radius:12px;padding:24px;margin:18px 0}pre{white-space:pre-wrap;font:inherit;line-height:1.55}.muted{color:#526875}button{padding:10px 18px;background:#176082;color:white;border:0;border-radius:6px;cursor:pointer}details{margin:15px 0}.badge{font-size:13px;color:#176082}.filters{background:#e6eef3;padding:15px;border-radius:12px}summary{cursor:pointer}footer{padding:20px 0;color:#526875}'''
JS = r'''
const $=s=>document.querySelector(s);let rows=[],position=0,filtered=[];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function choices(key,label){const values=[...new Set(rows.map(r=>r[key]).filter(Boolean))].sort();return `<label>${label}<br><select data-key="${key}"><option value="">All</option>${values.map(v=>`<option>${esc(v)}</option>`).join('')}</select></label>`}
function filter(){const selects=[...document.querySelectorAll('select')],query=$('#search').value.toLowerCase();filtered=rows.filter(r=>selects.every(s=>!s.value||r[s.dataset.key]===s.value)&&(!query||JSON.stringify(r).toLowerCase().includes(query)));position=0;render()}
function render(){const portion=filtered.slice(position,position+20);$('#count').textContent=`${filtered.length.toLocaleString()} matches · showing ${filtered.length?position+1:0}–${Math.min(position+20,filtered.length)}`;$('#cards').innerHTML=portion.map(r=>`<article><span class="badge">${esc(r.item_id)} · ${esc(r.item_type)} · ${esc(r.source_pdf)}</span><h2>${esc(r.question)}</h2>${r.options?`<pre>${esc(Object.entries(r.options).map(([k,v])=>k+'. '+v).join('\n'))}</pre>`:''}<details ${MODE==='dataset'?'open':''}><summary>Reference answer</summary><pre>${esc(Array.isArray(r.reference)?r.reference.join('\n'):r.reference)}</pre></details>${MODE==='models'?`<p><b>${esc(r.model_id)}</b> · ${esc(r.precision)} · ${esc(r.mode)}</p><pre>${esc(r.answer)}</pre><p class="muted">${esc(r.assessment)}</p>`:`<p class="muted">Topic: ${esc(r.topic||r.section_label||'Not assigned')} · Stored source pages: ${esc(r.page_start)}–${esc(r.page_end)}. ${r.item_type==='clinical_case'?'Clinical pages are one-based.':'Knowledge pages are zero-based PDF indices.'}</p>`}</article>`).join('');$('#prev').disabled=position===0;$('#next').disabled=position+20>=filtered.length}
async function start(){try{const binary=Uint8Array.from(atob($('#payload').textContent.trim()),c=>c.charCodeAt(0));const text=await new Response(new Blob([binary]).stream().pipeThrough(new DecompressionStream('gzip'))).text();rows=JSON.parse(text);$('#filters').innerHTML=choices('domain','Dataset')+choices('item_type','Question type')+choices('source_pdf','Source')+(MODE==='models'?choices('model_id','Model')+choices('precision','Precision')+choices('mode','Condition'):choices('topic','Topic'));document.querySelectorAll('select').forEach(s=>s.addEventListener('change',filter));$('#search').addEventListener('input',filter);$('#prev').onclick=()=>{position=Math.max(0,position-20);render()};$('#next').onclick=()=>{position+=20;render()};filter()}catch(e){$('#count').textContent='This viewer needs a modern browser with gzip decompression support. Use current Chrome, Edge or Firefox.'}}
start();
'''


def page(title, description, records, mode):
    raw = json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{STYLE}</style><main><nav><a href="../index.html">Repository guide</a><a href="dataset_viewer.html">Databank</a><a href="model_results_viewer.html">Model answers</a></nav><h1>{title}</h1><p>{description}</p><div class="filters"><div id="filters"></div><label>Search questions, references or answers<br><input id="search" placeholder="Search"></label></div><p id="count">Loading saved records…</p><div id="cards"></div><button id="prev">Previous</button> <button id="next">Next</button><footer>Research material, not clinical decision support. No API calls or external assets. Source books and verbatim evidence excerpts are not included.</footer></main><script id="payload" type="application/octet-stream">{payload}</script><script>const MODE={json.dumps(mode)};{JS}</script></html>'''


def build():
    root = project_root()
    bank = []
    for domain in ['knowledge', 'clinical']:
        for item in read_jsonl(root / f'data/final/{domain}.jsonl'):
            bank.append({**item, 'domain': domain, 'reference': item['answer']})
    target = root / 'results'
    (target / 'dataset_viewer.html').write_text(page('The sleep-medicine databank', '1,215 knowledge questions and 120 clinical cases. The author reports completed sleep-expert review; see data/REVIEW_STATUS.json for the provenance of the released version.', bank, 'dataset'), encoding='utf-8')
    lookup = {r['item_id']: r for r in bank}
    knowledge = pd.read_csv(root / 'results/analysis/knowledge/answer_scores.csv').fillna('')
    clinical = pd.read_csv(root / 'results/analysis/clinical/judge_comparison_19_systems.csv').fillna('')
    scores = {}
    for r in knowledge.to_dict('records'):
        scores[(r['model_id'],r['precision'],r['item_id'],r['mode'])] = f"{r.get('label','')} · primary score: {r.get('primary_score','')} · {r.get('reason','')}"
    for r in clinical.to_dict('records'):
        scores[(r['model_id'],r['precision'],r['item_id'],r['mode'])] = f"Gemini: {r['gemini_label']} · GLM: {r['glm_label']}. Inclusion in paired analysis requires valid label agreement for BOTH conditions."
    records = []
    for domain in ['knowledge', 'clinical']:
        for r in answer_frame(domain).fillna('').to_dict('records'):
            key = (r['model_id'],r['precision'],r['item_id'],r['mode'])
            item = lookup[r['item_id']]
            records.append({**item, **{k:r[k] for k in ['model_id','precision','mode','answer']}, 'assessment': scores.get(key,'No score recorded')})
    if len(bank) != 1335 or len(records) != 33720:
        raise ValueError('Unexpected release population')
    (target / 'model_results_viewer.html').write_text(page('Model answers and assessments', '29,160 knowledge answers (three conditions) and 4,560 clinical answers (19 model–precision systems). Scores measure agreement with the study references; denominators and judge limitations are documented in METHODS.md.', records, 'models'), encoding='utf-8')
    return {'questions':len(bank), 'answers':len(records)}


if __name__ == '__main__':
    print(build())
