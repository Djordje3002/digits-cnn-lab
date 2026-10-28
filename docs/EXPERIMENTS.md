# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m digits_cnn train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m digits_cnn predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

The scikit-learn digits subset contains 1,797 real images. A seeded stratified 60%/20%/20% image split gives 1,078 training, 359 validation and 360 test examples. Values are normalized by the fixed known sensor maximum, 16. The CNN uses two convolutions, max pooling, a hidden layer and dropout, totaling 38,282 parameters. AdamW trains for 40 epochs on CPU; the checkpoint with lowest validation cross-entropy is restored. A scaled logistic baseline uses the same training examples. Validation log loss selects the stronger family; test results are reported for both. The prediction CLI explicitly demonstrates the CNN checkpoint.

## Reading the result

The restored CNN reaches 98.61% test accuracy versus 96.39% for logistic regression on this split. Five of 360 test images are wrong. Inspect the error gallery and learning curve before interpreting a headline accuracy. Softmax confidence is not calibrated, and a random image split does not measure performance on entirely unseen writers.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.
