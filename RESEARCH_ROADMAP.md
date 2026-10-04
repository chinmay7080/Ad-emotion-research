# Research roadmap

This plan is a sequence of evidence gates, not a promise that every proposed model will produce a positive result. The three studies have different outcomes and cannot be pooled into one accuracy number.

## Completed and independently checked

1. **TVC35 physiological check (partial cohort).** Matched model-predicted cortical time courses to the available measured fMRI; tested advertisement-specific residual correspondence against temporal, wrong-ad, spatial, reversal, and edge controls. Reported null preference and recall associations. Four of 20 Study 1b participants and most Study 1a scans remain unavailable.
2. **Pitt matched model comparisons.** Evaluated content-only and content-plus-predicted-brain models on the same 1,865 advertisements and five frozen ad-disjoint outer folds. Reported paired ad-level intervals, a ten-comparison family adjustment, and the stronger Ridge comparator, including targets without reliable neural improvement.
3. **AdCumen content pipeline.** Froze a label-blind, transcript-screened 10,000-clip cohort; independently validated video, audio, and language feature artifacts; fit seven content views with parent-ad-grouped train-only selection; evaluated the quarantined 1,180-clip development-validation set once. Final-test labels remain sealed.

## Next evidence gates

| Gate | Work | Pass condition |
| --- | --- | --- |
| Published comparator | Run the released TSAM model on the AdCumen cohort using safe checkpoint loading, verified preprocessing, eight-class mapping, and a training-only smoke test. | Exact checkpoint/input provenance, eight probabilities per clip, complete artifact audit, and a clearly labelled selected-cohort comparison. The released model may have used different training data. |
| Short-clip neural protocol | Freeze TRIBE's five-second audiovisual-language context, padding, and masks without using outcome labels to choose them. Benchmark a label-blind training sample. | Stable complete outputs, preserved raw cortical arrays, source/input hashes, measured throughput and a sensitivity report. The current three-clip A/V pilot is technical feasibility only. |
| Matched Study 2 increment | On identical clips and folds, compare content-only V+A+L with the same model family plus derived TRIBE-predicted neural features. Include dimension, regularization, and perturbation controls. | A predeclared paired result with parent-ad uncertainty; report a null result if the increment is not reliable. Measured-fMRI correspondence is required before biological interpretation. |
| Final-test lock | Freeze split/duplicate policy, feature and model choices, metrics, seeds, uncertainty, and custodian before accessing released final-test labels. | One auditable evaluation with all predictions retained and no test-driven redesign. |
| Independent replication | Obtain the missing TVC35 participants and a separate site or stimulus cohort under their respective permissions. | Replicated measured-fMRI correspondence with a defined reliability ceiling and site-aware inference. |
| Real-world outcome study | If rights, consent, and platform access are established, collect impression/click outcomes and run a prospective design. | Directly measured outcomes and appropriately controlled tests; current datasets cannot substitute for this stage. |

## Decision rule

The AdCumen content-only result stands independently. If the published-model comparator or neural gates fail, the study should report that limitation and complete the content analysis. A predicted-neural feature is never described as an actual viewer's brain response, and a five-second emotion label is never described as click performance or an exact causal trigger.
