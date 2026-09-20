# Knowledge-generation prompts

## Current portable workflow

Notebook 01 displays the complete rendered prompt for each of `mcq`, `free_text_qa`, and `criteria_qa` before the generation step. The implementation is `src/sleepinn_study/databank.py::generation_prompt`. All three use a shared template with a different requested `item_type`; the template states the answer format for each type. The displayed source packet is a labelled placeholder. The live request replaces it with extracted PDF pages.

This short prompt is a portable reconstruction created for this clean package. It was not the full prompt used to generate the archived ICSD questions.

## Archived ICSD generation

`icsd3_historical_generation_instructions.md` is a byte-identical copy of the instruction snapshot from run `icsd3_tr_v2_003`. `icsd3_historical_prompt_assembly.txt` contains the prompt-building function extracted from that run's final notebook snapshot. See `PROVENANCE.json` for source and released-file hashes.

The historical workflow used one shared instruction block for all three knowledge types, followed by a current request containing the source unit, chapter, topic, quota, required subtopics, and prior questions to avoid. It then appended the source pages. It did not use three separate fixed prompts. The instructions define MCQ keys A–D, free-text string answers, and criteria-list answers, along with evidence and coverage requirements. These instructions precede the human review reported by the author; the release-level review statement is in `data/REVIEW_STATUS.json`.

The 215 AASM items were retained from the earlier dataset. These ICSD files do not establish the exact prompts originally used for those AASM items. The portable workflow can use an AASM PDF for a new run, but that is not a reconstruction of its historical requests.

Individual source packets, changing duplicate-avoidance lists, and raw request histories remain in the original EVAL archive. These two files document the instruction text and assembly logic, not every historical request body.
