# Study 2 case study: from short ads to audited content models

**Status at 4 October 2026:** content-only development validation complete; final test sealed. The aim is to estimate generalization to previously unseen parent advertisements within a transcript-screened subset of AdCumen, using the dataset's released eight-class emotion-change label for five-second clips. It is an academic prediction study, not a test of sales or clicks.

## Research and engineering design

1. **Choose examples without their labels.** A fixed text-plausibility rule and deterministic ranking selected 10,000 development clips: 8,820 training clips from 3,792 parents and 1,180 validation clips from 467 different parents. Known strong visual similarities across the released development splits were quarantined before selection. The subset is therefore explicitly called *transcript-screened*; the speech was not human-audio-verified.
2. **Keep every clip recoverable.** GPU work was assigned deterministically across eight shards. Each clip produced one compressed feature file and one hash-bearing receipt. A deliberate interruption test checked resumption before the full run. An all-zero padded sample at the exact five-second boundary was caught in a bounded benchmark; the frozen rule retains the half-open interval `[0, 5)` on a 2 Hz grid.
3. **Repair only observed failures.** The first production pass left 8,694 validated pairs. A versioned recovery produced another 1,303; three narrow transcript-alignment/parsing cases were diagnosed and repaired separately. Valid earlier artifacts and logs were preserved. An independent validator then reconciled **8,694 + 1,303 + 3 = 10,000** exact feature/receipt pairs and checked cohort membership, hashes, source-word retention, provenance, shapes, finite values, and timing.
4. **Model with parent separation.** Label-free video, audio, and language vectors were assembled first. Seven views (`V`, `A`, `L`, `VA`, `VL`, `AL`, `VAL`) used the same clips and labels. Training-only, three-fold parent-ad-grouped cross-validation selected regularization for a multiclass linear model. Scaling was fitted inside each training fold. The strongest single-modality comparator was chosen *before* validation labels were opened.
5. **Evaluate once and audit independently.** The frozen validation evaluation saved probabilities for all seven views and 1,180 clips. A second program reconciled all **8,260** prediction rows, recomputed metrics, verified file hashes, and checked the paired bootstrap interval. Neither stage accessed final-test labels.

## Development-validation result

Balanced accuracy is the mean recall across the eight fixed classes. Every row below uses the same 1,180 clips from 467 parents. The primary comparison, chosen before validation, is `VAL` against the strongest training-selected single modality, `L`.

| Content view | Balanced accuracy | Macro-F1 | Macro-AUROC | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Video (`V`) | 0.350 | 0.339 | 0.740 | 1.973 |
| Audio (`A`) | 0.274 | 0.260 | 0.693 | 2.663 |
| Language (`L`) | 0.391 | 0.388 | 0.779 | 2.126 |
| Video + audio (`VA`) | 0.353 | 0.341 | 0.753 | 2.185 |
| Video + language (`VL`) | 0.445 | 0.438 | 0.810 | 2.120 |
| Audio + language (`AL`) | 0.430 | 0.421 | 0.805 | 1.987 |
| Video + audio + language (`VAL`) | **0.455** | **0.445** | **0.823** | **1.834** |

The exact primary balanced-accuracy difference was `0.455055 − 0.391238 = 0.063817`. A 2,000-replicate **paired, parent-ad-clustered** bootstrap gave a percentile 95% interval of `[0.035063, 0.091328]`. The combined model's ten-bin expected calibration error was **0.241**, so its probabilities should not be presented as well calibrated. These are development-validation results; the released final test remains sealed.

The other views and metrics describe the audited evaluation; they were not separately predeclared as primary contrasts, and this table makes no multiple-comparison significance claim for them.

![Aggregate development-validation results](figures/validation_balanced_accuracy.png)

The figure can be regenerated from its aggregate values with `python3 study2/figures/render_validation_figure.py` from the repository root. Per-clip outcomes, identifiers, and model weights are deliberately absent.

## What this result does and does not establish

The result supports a narrow predictive conclusion: within this transcript-screened cohort and development split, combining the three content representations improved fixed-class balanced accuracy over the train-selected language-only representation on matched clips. It is **not** a final-test estimate. The cohort excludes some silent, non-English, or text-implausible clips by design; it is not representative of all advertisements. Strong visual quarantine does not guarantee that every cross-split duplicate or campaign variant is known.

The three-clip TRIBE predicted-cortical pilot is a separate technical feasibility check. It is **not** included in the table and gives no Study 2 emotion-classification result. A content-versus-content-plus-predicted-brain comparison has not been run. Predicted responses are model outputs, not measured viewer fMRI; measured-fMRI correspondence would require a separate validation study. The dataset does not contain outcomes sufficient to claim click-through rate, conversions, precise trigger seconds, or causal effects.

## Reproduction boundary

The original production pipeline depends on separately controlled AdCumen media, automatic transcripts, and pretrained model assets. The accompanying [synthetic demo](examples/synthetic_demo.py) demonstrates the cohort and artifact-checking ideas without redistributing those inputs. Its toy counts and arrays are not the study result. The public score figure contains only the independently audited aggregate numbers shown above.
