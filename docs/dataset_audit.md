# Dataset audit

This checkout contains **1,140 prompt records**. Counts describe the full dataset, not the small generation evaluation subset.

## Safety category distribution

| Label | Count | % |
|---|---:|---:|
| harassment_bullying | 25 | 2.19 |
| harmless | 985 | 86.40 |
| illegal | 65 | 5.70 |
| misinformation_disinformation | 65 | 5.70 |

## Frozen experiment splits

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| harassment_bullying | 15 | 5 | 5 |
| harmless | 591 | 197 | 197 |
| illegal | 39 | 13 | 13 |
| misinformation_disinformation | 39 | 13 | 13 |

## Language distribution

| Label | Count | % |
|---|---:|---:|
| en | 1139 | 99.91 |
| unknown | 1 | 0.09 |

## Topic distribution

| Label | Count | % |
|---|---:|---:|
| arts_entertainment | 206 | 18.07 |
| coding_technology | 125 | 10.96 |
| crime_security | 73 | 6.40 |
| daily_life_geography | 117 | 10.26 |
| education_language | 206 | 18.07 |
| finance_business | 32 | 2.81 |
| health_wellbeing | 50 | 4.39 |
| math_reasoning | 66 | 5.79 |
| other_uncertain | 68 | 5.96 |
| politics_history_society | 69 | 6.05 |
| relationships_identity | 35 | 3.07 |
| science_environment | 93 | 8.16 |

## Task Type distribution

| Label | Count | % |
|---|---:|---:|
| calculation_reasoning | 58 | 5.09 |
| classification_extraction | 191 | 16.75 |
| coding | 42 | 3.68 |
| content_generation | 306 | 26.84 |
| explanation_qa | 248 | 21.75 |
| listing | 101 | 8.86 |
| other_uncertain | 1 | 0.09 |
| procedural_advice | 118 | 10.35 |
| rewriting_editing | 61 | 5.35 |
| summarization | 9 | 0.79 |
| translation | 5 | 0.44 |

## Category × language

| Category | Language | Count | % within category |
|---|---|---:|---:|
| harassment_bullying | en | 25 | 100.00 |
| harmless | en | 985 | 100.00 |
| illegal | en | 64 | 98.46 |
| illegal | unknown | 1 | 1.54 |
| misinformation_disinformation | en | 65 | 100.00 |

## Quality and lengths

- Normalized duplicate groups: 0.
- Empty instructions / split issues: 0.
- Word counts (whitespace-delimited): `{'n': 1140, 'min': 3, 'median': 10.0, 'mean': 19.279824561403508, 'max': 625}`.
- Character counts: `{'n': 1140, 'min': 18, 'median': 58.0, 'mean': 119.75175438596492, 'max': 3450}`.
- Available audio manifests: `{'harmful': True, 'harmless': True}`.
- Audio duration statistics in seconds: `{'n': 1140, 'min': 1.8, 'median': 4.25, 'mean': 7.980350877192983, 'max': 251.0}`.
- Missing audio files referenced by available manifests: 0.

## Method and limitations

Safety categories come from the original dataset; Alpaca is normalized to harmless. Experiment splits come from results/splits.json, not the HarmBench source split field. Prompt IDs use source and original list index; source hashes are saved in summary.json.

Annotation status counts: {'model_labeled': 980, 'model_flagged': 160}. Provisional topic/task labels use keyword rules. Model labels use scripts/label_dataset.py with its saved model name, configuration hash and raw-output journal. Model labels are automatic, not human verified. Flagged cases need review; unflagged outputs can still be wrong. Language unknown means unannotated, not necessarily a non-English input. TTS voices do not prove input language.

A deterministic pilot contains ten records from each safety category (40 total); it deliberately oversamples harmful categories and is not representative of overall proportions. Review pilot_sample.csv, refine the guide, then review the full metadata CSV. Set review_status and language_status to human_reviewed only after checking the corresponding labels. Re-running preserves edited annotations. The review queue includes provisional, model-flagged and invalid records. Assess an independent stratified sample of unflagged model labels too; model agreement with itself is not accuracy.

Exact duplicates use case-folding and whitespace normalization, not semantic similarity. Length counts include supplied context. Missing manifests mean audio statistics are unavailable, not zero. Existing harmless/harmful source labels are inherited rather than a new independent safety assessment.

The harmless/harmful class imbalance and unequal harmful category sizes should be reported; this audit does not resample data or change frozen splits.
