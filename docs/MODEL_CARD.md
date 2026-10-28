# Model card — Digits CNN Lab

Version: 1.0.0 · Experiment seed: 42 · Selected model: `cnn`

## Intended use

Recognize handwritten digits from 8×8 grayscale images while demonstrating the complete deep-learning experiment cycle: deterministic initialization, minibatch optimization, validation checkpointing, a classical baseline and visual error analysis.

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

The scikit-learn digits subset contains 1,797 real images. A seeded stratified 60%/20%/20% image split gives 1,078 training, 359 validation and 360 test examples. Values are normalized by the fixed known sensor maximum, 16. The CNN uses two convolutions, max pooling, a hidden layer and dropout, totaling 38,282 parameters. AdamW trains for 40 epochs on CPU; the checkpoint with lowest validation cross-entropy is restored. A scaled logistic baseline uses the same training examples. Validation log loss selects the stronger family; test results are reported for both. The prediction CLI explicitly demonstrates the CNN checkpoint.

| Model | accuracy | macro f1 | log loss |
| --- | ---: | ---: | ---: |
| `scaled_logistic` | 0.9639 | 0.9640 | 0.1263 |
| `cnn` | 0.9861 | 0.9858 | 0.0598 |


## Interpretation

The restored CNN reaches 98.61% test accuracy versus 96.39% for logistic regression on this split. Five of 360 test images are wrong. Inspect the error gallery and learning curve before interpreting a headline accuracy. Softmax confidence is not calibrated, and a random image split does not measure performance on entirely unseen writers.

## Limitations

Writer IDs are unavailable in the bundled subset, so writers can appear across partitions. The small images and curated preprocessing differ greatly from camera photos and scanned documents. There is no image upload or OCR pipeline. CPU determinism is requested, but different libraries, hardware or platforms can still produce small numeric differences. An all-zero input is schema-valid and can still get a confident prediction; out-of-distribution rejection is not implemented.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.
