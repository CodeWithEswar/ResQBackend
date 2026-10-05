# ResQ model validation report

This report is generated from the active model manifest. It describes a local research validation run, not certified disaster identification.

## Active artifacts

- YOLO checkpoint: `resq-face-289b57fb99f0.pt`
- Detector SHA256: `289b57fb99f00aa83b0157aac75e0af36e46ee4f3464b25761f3e05cdfe7735e`
- SFace SHA256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- Runtime manifest: `models/pipeline.pkl` (reviewable copy: `models/pipeline.json`)
- Face cosine threshold: 0.363000
- Ear identity ranking: disabled; photographs remain supporting evidence.
- Automatic identity confirmation: disabled; independent review required.

## Detection training

Trained for 3 epochs on 512 officially annotated WIDER FACE photographs and validated on 128 photographs. The laptop used CPU PyTorch. The initialization was a published face-trained checkpoint; this was transfer learning, not training from random weights.

| Detection metric | Published initialization | Active fine-tune |
| --- | ---: | ---: |
| Precision | 90.48% | 90.01% |
| Recall | 67.25% | 67.33% |
| mAP@0.5 | 75.49% | 75.89% |
| mAP@0.5:0.95 | 44.20% | 44.11% |

The first fine-tune failed regression gates and was not deployed. The conservative run passed the precision, recall and mAP gates. The published initialization may already have been evaluated on WIDER validation; this is a regression comparison and cannot establish generalization to disaster victims.

## Recognition validation

The threshold was calibrated using 500 official LFW development pairs and evaluated on 1000 development-test pairs. Calibration and test subjects are disjoint. LFW central-portrait metadata supplied the explicit face selection used by the API.

| Measure | Calibration | Subject-disjoint test |
| --- | ---: | ---: |
| Usable same-person pairs | 250 | 500 |
| Usable different-person pairs | 250 | 499 |
| False accepts | 0 | 0 |
| True-accept rate | 98.40% | 97.20% |
| False-accept rate | 0.00% | 0.00% |
| False-accept rate 95% Wilson upper bound | 1.51% | 0.76% |
| Rejected input pairs | 0 | 1 |

Recognition rates are conditional on usable, correctly selected faces. Pair verification does not measure identification against a large registry, and zero observed false accepts does not mean zero real-world risk.

## Deployment limits

The live Supabase registry was reachable but had no cases or face references at verification time. Enroll authorized reference photographs before searching. Public benchmark people were never inserted into the missing-person registry.

The runtime reports `disaster_validated=false`. Representative, authorized disaster data and separate field evaluation are still required, especially for injury, severe occlusion, tiny faces and postmortem images. The WIDER training subset contains rescue/car-accident events, but only a handful; those images do not validate disaster-specific accuracy.

Datasets and model licenses are documented in `README.md`. The deployable archive is `work/resq-model.zip`; its data excludes case photographs and training datasets.
