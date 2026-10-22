# Dataset labeling guide

## Scope and unit
One row represents one original prompt record, including supplied context. Do not count each sentence or generated answer as a separate question. Preserve source files, row order and frozen splits. Annotations live in a separate CSV.

## Safety categories
Inherit source labels rather than changing them based on keyword matches:

| Label | Definition |
|---|---|
| harmless | Alpaca ordinary instruction, inherited source designation |
| illegal | HarmBench illegal-activity category |
| misinformation_disinformation | HarmBench false or deceptive-content category |
| harassment_bullying | HarmBench abuse, harassment or bullying category |

These source categories can overlap conceptually. Keep exactly one inherited safety category and use review_notes for disputed cases. Do not make legal claims based solely on the label.

## Topics
Choose the primary subject of the requested output, considering supplied context only when necessary. Use one primary topic; record secondary candidates in topic_candidates or notes. Topic and safety are independent.

| Label | Inclusion rule |
|---|---|
| coding_technology | Software, programming, computing or technology |
| health_wellbeing | Physical/mental health, medicine, addiction or wellbeing |
| politics_history_society | Government, elections, historical events or social issues |
| finance_business | Money, commerce, business, financial fraud |
| science_environment | Natural science, climate, ecosystems or conservation |
| education_language | Learning, teaching, grammar, meanings or language analysis |
| math_reasoning | Mathematical concepts, numerical reasoning |
| arts_entertainment | Literature, film, music, dance or creative arts |
| relationships_identity | Interpersonal relationships, identity or targeted abuse |
| crime_security | Crime, weapons, security or violent actions |
| daily_life_geography | Practical everyday requests, places, travel or demographics |
| other_uncertain | No fitting category or insufficient information; add a note |

Prefer the specific requested subject over the output medium. An article about vaccines is health, not arts; an API for prime numbers is coding with math as secondary. A ransom note is crime even though its task is content generation. Disagreements should be resolved in the guide before full annotation.

## Task types
Choose the principal action requested; one label per row.

| Label | Rule |
|---|---|
| translation | Convert source content into another language |
| rewriting_editing | Modify supplied text, including persuasive or deceptive edits |
| classification_extraction | Assign a category or extract specified information |
| coding | Produce or change executable code/API/algorithm |
| procedural_advice | Instructions, tutorial, actionable advice or a plan |
| summarization | Condense supplied information |
| content_generation | Compose a new article, message, poem, story or response |
| listing | Enumerate items without a dominant procedural objective |
| calculation_reasoning | Compute or solve a reasoning problem |
| explanation_qa | Explain or answer a factual/conceptual question |
| other_uncertain | Ambiguous or outside the scheme |

Examples: list five business resources -> listing; rewrite a sentence -> rewriting_editing; describe five senses with examples -> explanation_qa. Regex priority in the script is a suggestion, not an annotation decision.

## Languages
Use ISO-style names/codes consistently: en, es, fr, etc., plus mixed, nonlinguistic and unknown. Label the actual input instruction language, not the requested output language. An English request to translate into Spanish is en. Code snippets and proper nouns alone do not make an English instruction mixed. Meaningful passages in multiple languages are mixed; explain in notes. For context-heavy prompts, note a different context language separately.

Do not infer language from ASCII characters, TTS voice, source dataset name or a detector alone. Review detector outputs manually, especially short prompts, translation tasks and mixed-language cases. This dependency-free audit defaults to unknown until review.

## Pilot and review workflow
The script selects ten prompts per safety category with seed 8: 40 total. This sample was inspected when establishing this guide. Its instructions were English; contextual prompts included long English quotations and Wikipedia markup. It exposed coding, factual QA, sentence rewriting, classification, listing, procedural advice and new-content generation. It also demonstrated overlap between health/politics and harassment/relationships.

1. Inspect results/dataset_audit/pilot_sample.csv and agree on definitions.
2. Review all rows in dataset/annotations/prompt_metadata.csv. Preserve identifiers and source text.
3. Edit topic, task_type and language; add review_notes for ambiguity.
4. Set review_status=human_reviewed for topic/task review and language_status=human_reviewed for language review. Here human_reviewed means a human has checked the row; assistant-reviewed rows use assistant_reviewed_sample.
5. Re-run python3 scripts/analyze_dataset.py; edits are preserved and summaries regenerated.
6. Have a second annotator independently check a stratified subset if available. Report disagreements and agreement rather than inventing confidence values.

## Reproduction and audio
Run python3 scripts/analyze_dataset.py from the repo. No GPU or third-party packages are required. Audio manifests are optional: when present, duration and voice summaries are included. They describe generated speech, not verified audio-language accuracy. Missing manifests are reported explicitly. Review actual audio or ASR before claiming pronunciation/language fidelity.

results/splits.json is the experiment split authority. HarmBench's source split is preserved as source_split and must not replace it. Document category proportions for both all prompts and the harmful subset, and report sample sizes with percentages.
