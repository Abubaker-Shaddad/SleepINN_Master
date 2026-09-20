# Publication checks

Validated on 20 September 2026:

- All 12 notebooks executed top to bottom with their default offline settings.
- All 14 unit tests passed, including complete-review export, original-record preservation, tampered-export rejection, blinding, retrieval fusion and API output boundaries.
- The reconstructed no-RAG prompt matched its saved SHA-256 for all 1,335 questions and cases.
- All 33,720 model-answer texts matched their recorded SHA-256 hashes.
- Clinical inclusion reproduced 19 systems, 4,053 response agreements and 1,831 complete consensus pairs.
- The five aggregate clinical contrasts reproduced the saved effects and shared-case permutation p values.
- Both HTML viewers loaded in a browser; dataset, model, precision, condition and text filters were checked.
- A text scan found no credential patterns, private user paths or email addresses in the publication files. Private source excerpts, raw provider responses and operational files were excluded during assembly.

No model inference, paid API assessment or GPU compatibility test was performed for this publication. The notebooks are distributed without execution outputs; rerun them locally to inspect the analysis.

Input and result hashes are in `data/FILE_SHA256.json`. Generated files and private experiments belong under ignored `outputs/`.
