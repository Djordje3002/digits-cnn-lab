# Data card

The 1,797-image digits dataset bundled with scikit-learn is a copy of the original UCI optical-digits test subset. Each 8×8 image has integer intensities 0–16 and a label 0–9. This project re-splits that subset for a compact educational experiment; it does not reproduce the original UCI train/test protocol or use MNIST.

## Source and citation

[Original dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)

Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49 . Subset distributed by scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

## License

UCI lists the dataset under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source when reusing data or derived figures. The repository's MIT license applies to its code, not to the dataset.

## Integrity and storage

```text
Bundled scikit-learn digits subset; the scikit-learn version is pinned in pyproject.toml. No separate download occurs.
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

The scikit-learn digits subset contains 1,797 real images. A seeded stratified 60%/20%/20% image split gives 1,078 training, 359 validation and 360 test examples. Values are normalized by the fixed known sensor maximum, 16. The CNN uses two convolutions, max pooling, a hidden layer and dropout, totaling 38,282 parameters. AdamW trains for 40 epochs on CPU; the checkpoint with lowest validation cross-entropy is restored. A scaled logistic baseline uses the same training examples. Validation log loss selects the stronger family; test results are reported for both. The prediction CLI explicitly demonstrates the CNN checkpoint.

## Representation and privacy

Writer IDs are unavailable in the bundled subset, so writers can appear across partitions. The small images and curated preprocessing differ greatly from camera photos and scanned documents. There is no image upload or OCR pipeline. CPU determinism is requested, but different libraries, hardware or platforms can still produce small numeric differences. An all-zero input is schema-valid and can still get a confident prediction; out-of-distribution rejection is not implemented.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.
