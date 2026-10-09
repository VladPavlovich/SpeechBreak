"""Dependency-free dataset audit. Run from any directory with python3.
Rule-based annotations are provisional; edit CSV labels and set review_status=human_reviewed.
"""
import argparse, csv, hashlib, json, random, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
 'coding_technology': r'\b(code|python|api|software|computer|algorithm|css|html|database|programming|magento)\b',
 'health_wellbeing': r'\b(health|medical|covid|vaccine|disease|drug|suicide|alcohol|addiction|diet|exercise|psycholog|gaslight)\w*',
 'politics_history_society': r'\b(politic|election|president|trump|biden|government|war|history|historical|racis|jewish|indigenous|immigration|society)\w*',
 'finance_business': r'\b(business|money|finance|financial|payment|cash|bank|scam|savings|marketing|company)\w*',
 'science_environment': r'\b(science|scientific|environment|climate|carbon|animal|species|physics|chemistry|biology|energy)\w*',
 'education_language': r'\b(school|teacher|student|education|degree|sentence|grammar|translate|translation|phrase|word|irony)\w*',
 'math_reasoning': r'\b(math|calculate|equation|prime|multiple|fraction|arithmetic|solve)\w*',
 'arts_entertainment': r'\b(poem|poetry|story|movie|film|music|song|art|dance|ballet|novel)\w*',
 'relationships_identity': r'\b(relationship|family|child|classmate|bully|bullies|bullying|harass|queer|lgbt|racist|antisemitic|anti-semitic)\w*',
 'crime_security': r'\b(steal|stolen|bomb|explosive|kidnap|ransom|handgun|hitman|illegal|smuggl|emergency|destroy|attack|weapon)\w*',
 'daily_life_geography': r'\b(country|population|japan|travel|city|cities|cook|recipe|home|food|sport)\w*',
}
TASKS = {
 'translation': r'\btranslate|translation\b',
 'rewriting_editing': r'\b(rewrite|rephrase|modify|edit|correct|paraphrase)\w*',
 'classification_extraction': r'\b(classify|categorize|extract|identify)\w*',
 'coding': r'\b(write|implement|create|develop)\b.*\b(code|api|function|program|script|algorithm)\b',
 'procedural_advice': r'\b(instructions|tutorial|step.by.step|guide|tips|plan|solutions|advice|how to)\b',
 'summarization': r'\b(summarize|summary|summarise)\b',
 'content_generation': r'\b(write|generate|compose|draft|create|brainstorm)\b',
 'listing': r'\b(list|name|enumerate)\b',
 'calculation_reasoning': r'\b(calculate|solve|compute|least common multiple)\b',
 'explanation_qa': r'\b(explain|what|why|which|who|find|meaning|describe)\b',
}

def norm(t): return ' '.join(t.casefold().split())
def match(t, rules): return [k for k,v in rules.items() if re.search(v,t,re.I|re.S)]
def table(counts, denominator):
 return '\n'.join(['| Label | Count | % |','|---|---:|---:|']+[f'| {k} | {v} | {100*v/denominator:.2f} |' for k,v in sorted(counts.items())])
def describe(xs):
 return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'mean':statistics.mean(xs),'max':max(xs)} if xs else None

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT); args=ap.parse_args(); root=args.root.resolve()
 out=root/'results/dataset_audit'; out.mkdir(parents=True,exist_ok=True)
 dest=root/'dataset/annotations/prompt_metadata.csv'; dest.parent.mkdir(parents=True,exist_ok=True)
 old={}
 if dest.exists():
  with dest.open(newline='') as f: old={r['prompt_id']:r for r in csv.DictReader(f)}
 split_json=json.loads((root/'results/splits.json').read_text()); memberships=defaultdict(list)
 for split,cats in split_json.items():
  for cat,names in cats.items():
   for name in names: memberships[name].append((split,cat))
 manifest={}; manifest_status={}
 for group in ['harmful','harmless']:
  p=root/f'dataset/audio/{group}_audio/manifest.json'; manifest_status[group]=p.exists()
  if p.exists():
   for r in json.loads(p.read_text()): manifest[Path(r['audio_path']).name]=r
 files=[('harmbench_illegal_harassment_misinfo.json','HarmBench'),('harmless_alpaca.json','Alpaca')]
 rows=[]; hashes={}; issues=[]; text_groups=defaultdict(list); audio_missing=[]; durations=[]
 for name,source in files:
  p=root/'dataset/processed'/name; hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
  for i,r in enumerate(json.loads(p.read_text())):
   text=r.get('instruction') or ''; source_cat=r.get('category','unknown'); cat='harmless' if source=='Alpaca' else source_cat
   pid=f'{source.lower()}:{i:04d}'; wav=f'{i:04d}_{source_cat}.wav'; sm=memberships.get(wav,[])
   if len(sm)!=1: issues.append({'prompt_id':pid,'issue':'split_membership','memberships':sm})
   if not text.strip():issues.append({'prompt_id':pid,'issue':'empty_instruction'})
   request=text.rsplit('\n---\n',1)[-1]; topics=match(request,TOPICS); tasks=match(request,TASKS)
   m=manifest.get(wav,{}); audio_path=m.get('audio_path',''); duration=m.get('duration_s','')
   if audio_path and not (root/audio_path).exists():audio_missing.append(pid)
   if isinstance(duration,(int,float)):durations.append(duration)
   row=dict(prompt_id=pid,source=source,source_index=i,instruction=text,source_category=source_cat,safety_category=cat,
    topic=topics[0] if topics else 'other_uncertain',topic_candidates=';'.join(topics),task_type=tasks[0] if tasks else 'other_uncertain',
    language='unknown',language_status='unreviewed',split=sm[0][0] if len(sm)==1 else 'unknown',source_split=r.get('split',''),
    word_count=len(text.split()),character_count=len(text),audio_filename=wav,audio_path=audio_path,duration_s=duration,voice=m.get('voice',''),
    review_status='provisional',pilot_sample='no',review_notes='Language requires review; topic/task labels are keyword suggestions.')
   if pid in old:
    for k in ['topic','task_type','language','language_status','review_status','review_notes']:row[k]=old[pid].get(k,row[k])
    if old[pid].get('instruction')!=text:row.update(review_status='provisional',language_status='unreviewed',review_notes='Source text changed; re-review labels.')
   rows.append(row);text_groups[norm(text)].append(pid)
 groups=defaultdict(list)
 for r in rows:groups[r['safety_category']].append(r)
 for cat,rs in sorted(groups.items()):
  for r in random.Random(8).sample(rs,min(10,len(rs))):r['pilot_sample']='yes'
 with dest.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for filename,selected in [('pilot_sample.csv',[r for r in rows if r['pilot_sample']=='yes']),('review_queue.csv',[r for r in rows if r['review_status']!='human_reviewed' or r['language_status']!='human_reviewed'])]:
  with (out/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(selected)
 counters={field:dict(Counter(r[field] for r in rows)) for field in ['source','safety_category','language','topic','task_type','split','review_status']}
 cross={}
 for field in ['language','split']:
  cross[field]={cat:dict(Counter(r[field] for r in rs)) for cat,rs in groups.items()}
 duplicate_groups=[ids for text,ids in text_groups.items() if text and len(ids)>1]
 summary={'total_records':len(rows),'unique_normalized_instructions':len(text_groups),'counts':counters,'category_cross_tabs':cross,
 'prompt_words':describe([r['word_count'] for r in rows]),'prompt_characters':describe([r['character_count'] for r in rows]),
 'audio_durations_seconds':describe(durations),'audio_total_seconds':sum(durations) if durations else None,'manifest_available':manifest_status,
 'voices':dict(Counter(r['voice'] for r in rows if r['voice'])),'missing_audio_files':audio_missing,'issues':issues,'duplicate_groups':duplicate_groups,'input_sha256':hashes,
 'annotation_method':'Topic/task: ordered regex suggestions. Language: unknown until reviewed. No automatic label is human-verified.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 report=['# Dataset audit','',f'This checkout contains **{len(rows):,} prompt records**. Counts describe the full dataset, not the small generation evaluation subset.','',
 '## Safety category distribution','',table(counters['safety_category'],len(rows)),'','## Frozen experiment splits','',
 '| Category | Train | Validation | Test |','|---|---:|---:|---:|']
 for cat in sorted(groups):report.append('| '+cat+' | '+' | '.join(str(cross['split'][cat].get(s,0)) for s in ['train','val','test'])+' |')
 for field in ['language','topic','task_type']:
  report+=['',f'## {field.replace("_"," ").title()} distribution','',table(counters[field],len(rows))]
 report+=['','## Category × language','', '| Category | Language | Count | % within category |','|---|---|---:|---:|']
 for cat,cs in sorted(cross['language'].items()):
  for lang,n in sorted(cs.items()):report.append(f'| {cat} | {lang} | {n} | {100*n/len(groups[cat]):.2f} |')
 report+=['','## Quality and lengths','',f'- Normalized duplicate groups: {len(duplicate_groups)}.',f'- Empty instructions / split issues: {len(issues)}.',
 f'- Word counts (whitespace-delimited): `{summary["prompt_words"]}`.',f'- Character counts: `{summary["prompt_characters"]}`.',
 f'- Available audio manifests: `{manifest_status}`.',f'- Audio duration statistics in seconds: `{summary["audio_durations_seconds"]}`.',
 f'- Missing audio files referenced by available manifests: {len(audio_missing)}.','',
 '## Method and limitations','',
 'Safety categories come from the original dataset; Alpaca is normalized to harmless. Experiment splits come from results/splits.json, not the HarmBench source split field. Prompt IDs use source and original list index; source hashes are saved in summary.json.',
 '', 'Topic/task labels are provisional ordered keyword matches on the request after the final context separator. They can miss paraphrases or choose the wrong topic. Language remains unknown until reviewed; American English TTS voices do not prove prompt language. Foreign words, translation targets, names and code require separate judgment.',
 '', 'A deterministic pilot contains ten records from each safety category (40 total); it deliberately oversamples harmful categories and is not representative of overall proportions. Review pilot_sample.csv, refine the guide, then review the full metadata CSV. Set review_status and language_status to human_reviewed only after checking the corresponding labels. Re-running preserves edited annotations.',
 '', 'Exact duplicates use case-folding and whitespace normalization, not semantic similarity. Length counts include supplied context. Missing manifests mean audio statistics are unavailable, not zero. Existing harmless/harmful source labels are inherited rather than a new independent safety assessment.',
 '', 'The harmless/harmful class imbalance and unequal harmful category sizes should be reported; this audit does not resample data or change frozen splits.']
 docs=root/'docs';docs.mkdir(exist_ok=True);(docs/'dataset_audit.md').write_text('\n'.join(report)+'\n')
 print(f'Audited {len(rows)} records; {len(issues)} input issues. Wrote {dest} and {out}.')
if __name__=='__main__':main()
