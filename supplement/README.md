# Supplementary materials

Scripts that produced the RAG-TRACE supplement, and the compiled supplement PDF.

The PDF is `supplementary-materials.pdf`. Waveforms, saved result files, and API keys are not in this folder. The notes and scores already in this repository stay under `results/ptbxl_baseline/`.

These scripts expect `results/` and `knowledge/` beside the script file. Do not rerun `scripts/figures/make_figures.py` or `scripts/figures/make_review_figures.py`. Those files also redraw figures that belong in the main paper.

| Supplement section | Script |
| --- | --- |
| Methods S1, retrieval calibration | `confirm_retrieval.py`, `clean_retrieval.py` |
| Methods S2, affirmation list | `claim_scoring.py` |
| Methods S3, statement links | `revision_evals.py` |
| Methods S4 and Results S3, EEG, sleep, noise, waveform network | `eeg_check.py`, `sleep_check.py`, `download_ds004504.py`, `nstdb_noise.py`, `noise_robustness.py`, `train_criteria_net.py`, `mimic_transfer.py`, `mimic_coverage.py` |
| Methods S5, numeric cards | `clean_retrieval.py`, `constrained_notes.py` |
| Methods S6, identical schema and constrained identifiers | `fair_notes.py`, `matched_constrained.py`, `score_matched_constrained.py`, `matched_intervals.py` |
| Methods S7, incomplete open-model notes | `score_qwen.py`, `score_deepseek_full.py`, `cluster/qwen_notes.py` |
| Methods S8 and Results S8, second algorithm | `independent_measurements.py`, `retention_eval.py`, `retention_paired.py` |
| Results S1, measurement agreement and noise | `independent_measurements.py`, `noise_robustness.py`, `nstdb_noise.py` |
| Results S2, earlier samples | `retrieval_harm.py`, `compare_policies.py`, `rescore_corrected.py`, `confirm_retrieval.py` |
| Figures in the supplement | `figures/make_figures.py`, `figures/make_review_figures.py` |
| Prompts | `heldout_notes.py`, `fair_notes.py`, `matched_constrained.py`, `eeg_check.py`, `sleep_check.py` |

Shared scoring used by those scripts is included: `diagnosis_rule.py`, `exploratory_revision.py`, `glm_heldout.py`, and `score_glm_independent.py`.
