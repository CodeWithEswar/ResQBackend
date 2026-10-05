"""Generate a reviewable report from the active, verified model release."""
from pathlib import Path
from ml.artifacts import read_bundle

ROOT=Path(__file__).resolve().parents[1]


def main():
    bundle=read_bundle(ROOT/'models/pipeline.pkl')
    training=bundle['detector'].get('training')
    recognition=bundle['calibration'].get('evaluation')
    if not training or not recognition:
        raise RuntimeError('Both an active local training report and promoted calibration are required')
    lines=['# ResQ model validation report','',
           'This report is generated from the active model manifest. It describes a local research validation run, not certified disaster identification.','',
           '## Active artifacts','',
           f"- YOLO checkpoint: `{bundle['detector']['file']}`",
           f"- Detector SHA256: `{bundle['detector']['sha256']}`",
           f"- SFace SHA256: `{bundle['recognizer']['sha256']}`",
           f"- Runtime manifest: `models/pipeline.pkl` (reviewable copy: `models/pipeline.json`)",
           f"- Face cosine threshold: {bundle['thresholds']['face']:.6f}",
           '- Ear identity ranking: disabled; photographs remain supporting evidence.',
           '- Automatic identity confirmation: disabled; independent review required.','',
           '## Detection training','',
           f"Trained for {training['epochs_requested']} epochs on {training['train_images']} officially annotated WIDER FACE photographs and validated on {training['validation_images']} photographs. The laptop used CPU PyTorch. The initialization was a published face-trained checkpoint; this was transfer learning, not training from random weights.",'',
           '| Detection metric | Published initialization | Active fine-tune |',
           '| --- | ---: | ---: |']
    for key,label in [('precision','Precision'),('recall','Recall'),('map50','mAP@0.5'),('map50_95','mAP@0.5:0.95')]:
        lines.append(f"| {label} | {100*training['baseline'][key]:.2f}% | {100*training['candidate'][key]:.2f}% |")
    lines.extend(['','The first fine-tune failed regression gates and was not deployed. The conservative run passed the precision, recall and mAP gates. The published initialization may already have been evaluated on WIDER validation; this is a regression comparison and cannot establish generalization to disaster victims.','',
                  '## Recognition validation','',
                  f"The threshold was calibrated using {recognition['train_pairs_attempted']} official LFW development pairs and evaluated on {recognition['test_pairs_attempted']} development-test pairs. Calibration and test subjects are disjoint. LFW central-portrait metadata supplied the explicit face selection used by the API.",'',
                  '| Measure | Calibration | Subject-disjoint test |',
                  '| --- | ---: | ---: |'])
    for key,label in [('genuine_pairs','Usable same-person pairs'),('impostor_pairs','Usable different-person pairs'),('false_accepts','False accepts')]:
        lines.append(f"| {label} | {recognition['calibration'][key]} | {recognition['test'][key]} |")
    for key,label in [('true_accept_rate','True-accept rate'),('false_accept_rate','False-accept rate'),('far_wilson_95_upper','False-accept rate 95% Wilson upper bound')]:
        lines.append(f"| {label} | {100*recognition['calibration'][key]:.2f}% | {100*recognition['test'][key]:.2f}% |")
    lines.extend([f"| Rejected input pairs | {recognition['train_rejected']} | {recognition['test_rejected']} |",'',
                  'Recognition rates are conditional on usable, correctly selected faces. Pair verification does not measure identification against a large registry, and zero observed false accepts does not mean zero real-world risk.','',
                  '## Deployment limits','',
                  'The live Supabase registry was reachable but had no cases or face references at verification time. Enroll authorized reference photographs before searching. Public benchmark people were never inserted into the missing-person registry.','',
                  'The runtime reports `disaster_validated=false`. Representative, authorized disaster data and separate field evaluation are still required, especially for injury, severe occlusion, tiny faces and postmortem images. The WIDER training subset contains rescue/car-accident events, but only a handful; those images do not validate disaster-specific accuracy.','',
                  'Datasets and model licenses are documented in `README.md`. The deployable archive is `work/resq-model.zip`; its data excludes case photographs and training datasets.',''])
    target=ROOT/'MODEL_REPORT.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
