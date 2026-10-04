# Auditable research on advertisements, content, and predicted neural signals

This portfolio documents three linked academic studies. I built reproducible pipelines to ask whether models can track measured responses to advertisements, whether model-predicted neural features add value beyond the ad's own content, and how well audiovisual and spoken-language representations predict released clip-level emotion labels. The work includes positive, negative, and incomplete results.

**The outcomes here are measured fMRI correspondence, survey ratings, and dataset emotion labels. No study in this repository measures clicks, conversions, or commercial advertising lift.** Predicted brain activity is a model output, not a recording from a viewer of these ads.

## Verified progress

| Study | Question and sample | Verified result | Boundary |
| --- | --- | --- | --- |
| [1. TVC35](study1/STUDY1_CASE_STUDY.md) | Do predicted cortical time courses correspond to measured fMRI? Partial cohort: 16 participants, 35 ads. | After removing a leave-one-ad-out generic response from both sides, group-level advertisement-specific correlation was **0.250** (ad-bootstrap 95% CI **0.189–0.290**). | Partial cohort; preference and recall links were not established. |
| [2. Pitt](pitt/PITT_CASE_STUDY.md) | Do predicted cortical features improve survey-rating prediction beyond content on 1,865 ads? | For *Funny*, matched content + brain MLP Spearman was **0.7135**, versus **0.7043** for content MLP (family-adjusted 99.5% CI for the difference **0.0016–0.0168**). | It did **not** reliably exceed the stronger content-only Ridge model (**0.7078**); no reliable gain was established for the other studied targets. |
| [3. AdCumen](study2/STUDY2_CASE_STUDY.md) | Can video, audio, and spoken-language representations predict eight-class emotion-change labels? | **10,000** feature sets passed independent validation. On **1,180** development-validation clips, combined content reached **0.455** balanced accuracy versus **0.391** for the train-selected strongest single modality. | Transcript-screened cohort; final test sealed. No Study 2 content-plus-brain result yet. |

The Pitt and AdCumen results address different datasets and outcomes. The TVC35 result tests correspondence to measured fMRI; it does not validate emotion or purchasing claims. Read each case study for the split, controls, uncertainty, and negative results.

![Audited Study 2 development-validation results for seven content views](study2/figures/validation_balanced_accuracy.png)

This figure is **content-only** development validation. The [Pitt neural comparison](pitt/PITT_CASE_STUDY.md) and [TVC35 fMRI check](study1/STUDY1_CASE_STUDY.md) answer different questions.

## What this repository demonstrates

- **Research engineering:** resumable extraction, per-item receipts, hashes, exact cohort checks, independent audits, and repair of only observed failures.
- **Evaluation discipline:** parent-ad-disjoint folds, train-only feature processing, matched content-versus-content-plus-brain comparisons, clustered uncertainty, and a sealed final test.
- **A runnable public example:** the [Study 2 synthetic demo](study2/examples/synthetic_demo.py) exercises selection and integrity checks without redistributing an advertisement or a participant record.
- **Honest limits:** a small neural gain in one Pitt comparison, a stronger content-only baseline, null behavioral associations, and unfinished comparator/neural work are reported alongside positive findings.

## Quick start

This public demo uses synthetic data only. It is not a reproduction of the private full-cohort runs.

```bash
python -m pip install -r study2/requirements.txt
python study2/examples/synthetic_demo.py
python -m unittest discover -s study2/tests -v
```

The [Pitt analysis code](pitt/PITT_CASE_STUDY.md) is provided for methods inspection and requires separately obtained, licensed inputs to run. It contains no source advertisements, per-ad predictions, or model weights.

## Next research steps

The [research roadmap](RESEARCH_ROADMAP.md) separates finished work from proposed tests: a carefully scoped published-model comparator for AdCumen, a conditional Study 2 neural-feature test, locked final-test evaluation, replication of the partial fMRI result, and eventually a separately permissioned prospective study if click outcomes become available.

## Data, code, and interpretation

Only reviewed original code, synthetic fixtures, and aggregate research summaries belong here. The [data and rights note](DATA_AND_RIGHTS.md) explains what is excluded and why. This repository does not grant rights to third-party datasets, advertisements, pretrained weights, or participant data. It is an academic research portfolio, not a validated ad-optimization product.
