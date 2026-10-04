# Study 2 research roadmap

This is a prospective plan, **not** a list of completed findings. Each stage has an evidence gate; null results and unmet gates will be reported rather than replaced by a positive-sounding claim. Status is current to **4 October 2026**.

| Stage | Status | Exit gate and public deliverable |
| --- | --- | --- |
| Cohort and extraction | **Complete** | 10,000 transcript-screened, label-blind selected clips; exact video/audio/language feature and receipt reconciliation; documented split and failure ledger. |
| Seven content views | **Complete on development validation** | Parent-ad-grouped training selection and one audited 1,180-clip validation evaluation. The fixed primary `VAL − L` difference is +0.064 balanced-accuracy points, 95% parent-ad bootstrap interval +0.035 to +0.091. |
| Error and calibration reporting | **Planned descriptive analysis** | Publish aggregate per-class recall, confusion and calibration summaries without changing the fitted models or choosing a new primary comparison after viewing validation. Any recalibration must use a separate training-only or designated development calibration split. |
| Published TSAM reference | **Preparation complete; real-media inference pending** | Safe checkpoint loading and synthetic preprocessing checks passed locally. A small training-only media run needs permissioned transfer of the restricted checkpoint and exact preprocessing verification. Any score on the current validation clips must be labelled a *descriptive reference*: the released TSAM checkpoint was selected on the official validation set containing these parent ads. The released-source 12-segment setting also differs from the paper's reported 16-frame condition. |
| Conditional predicted-brain features | **Technical three-clip pilot only** | First freeze a five-second context/padding protocol and test sensitivity on training-only clips. Preserve the full predicted cortical arrays and their provenance. Demonstrate correspondence with measured fMRI in a separate study before biological interpretation. |
| Matched neural increment | **Not started; conditional** | If the preceding gates pass, compare `VAL + B_VAL` with matched `VAL` on identical clip IDs, grouping, model family, and selection budget. Include feature-dimension and perturbation controls. Report a null increment if observed. No current result supports a neural benefit claim. |
| Final locked evaluation | **Not started** | Fix eligible cohort, duplicate policy, model choice, metrics and uncertainty method in writing; then open and score the sealed final test once. Distinguish its result from the present development-validation estimate. |

## Translation beyond this academic study

A future commercial test would require documented rights for every dataset, model and media asset, plus consented impression/click outcomes and a prospective evaluation in the actual serving context. Neither this roadmap nor the present emotion-label analysis demonstrates ad clickability, conversion lift, causal creative improvement, precise trigger times, or measured responses in individual viewers.
