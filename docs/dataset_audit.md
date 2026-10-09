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
| en | 40 | 3.51 |
| unknown | 1100 | 96.49 |

## Topic distribution

| Label | Count | % |
|---|---:|---:|
| arts_entertainment | 75 | 6.58 |
| coding_technology | 46 | 4.04 |
| crime_security | 23 | 2.02 |
| daily_life_geography | 34 | 2.98 |
| education_language | 128 | 11.23 |
| finance_business | 41 | 3.60 |
| health_wellbeing | 50 | 4.39 |
| math_reasoning | 24 | 2.11 |
| other_uncertain | 602 | 52.81 |
| politics_history_society | 46 | 4.04 |
| relationships_identity | 17 | 1.49 |
| science_environment | 54 | 4.74 |

## Task Type distribution

| Label | Count | % |
|---|---:|---:|
| calculation_reasoning | 18 | 1.58 |
| classification_extraction | 46 | 4.04 |
| coding | 13 | 1.14 |
| content_generation | 318 | 27.89 |
| explanation_qa | 247 | 21.67 |
| listing | 103 | 9.04 |
| other_uncertain | 249 | 21.84 |
| procedural_advice | 82 | 7.19 |
| rewriting_editing | 44 | 3.86 |
| summarization | 15 | 1.32 |
| translation | 5 | 0.44 |

## Category × language

| Category | Language | Count | % within category |
|---|---|---:|---:|
| harassment_bullying | en | 10 | 40.00 |
| harassment_bullying | unknown | 15 | 60.00 |
| harmless | en | 10 | 1.02 |
| harmless | unknown | 975 | 98.98 |
| illegal | en | 10 | 15.38 |
| illegal | unknown | 55 | 84.62 |
| misinformation_disinformation | en | 10 | 15.38 |
| misinformation_disinformation | unknown | 55 | 84.62 |

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

Topic/task labels are provisional ordered keyword matches on the request after the final context separator. They can miss paraphrases or choose the wrong topic. Language remains unknown until reviewed; American English TTS voices do not prove prompt language. Foreign words, translation targets, names and code require separate judgment.

A deterministic pilot contains ten records from each safety category (40 total); it deliberately oversamples harmful categories and is not representative of overall proportions. Review pilot_sample.csv, refine the guide, then review the full metadata CSV. Set review_status and language_status to human_reviewed only after checking the corresponding labels. Re-running preserves edited annotations.

Exact duplicates use case-folding and whitespace normalization, not semantic similarity. Length counts include supplied context. Missing manifests mean audio statistics are unavailable, not zero. Existing harmless/harmful source labels are inherited rather than a new independent safety assessment.

The harmless/harmful class imbalance and unequal harmful category sizes should be reported; this audit does not resample data or change frozen splits.
