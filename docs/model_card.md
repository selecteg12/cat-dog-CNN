# Model card

## Intended use

Educational binary classification of cat and dog photographs. The model always
returns a two-class decision, including for unrelated images. Sigmoid scores are
not calibrated probabilities. No pretrained model or external weights are used.

## Data

The author supplied a local `train.zip` with 25,000 JPEG images, balanced between
cats and dogs. The original download source and redistribution license were not
recorded; the repository does not redistribute the archive or images. See
[data_audit.json](data_audit.json) for the checksum and original checked counts.

File prefixes define labels: `cat.` is 0 and `dog.` is 1. Corrupt files are excluded.
Duplicate dimensions and EXIF-corrected RGB pixels are hashed before resizing.
One image per identical group is retained; conflicting-label groups are excluded.
The remaining images are split 70/15/15 per class with seed 42. Exact RGB hashes
are checked for overlap between subsets. Near-duplicates are not grouped.

## Preprocessing and architecture

Pillow corrects EXIF orientation, converts to RGB and resizes to 128×128 using
bilinear interpolation. The network receives float32 pixels in [0, 255] and
normalizes with an internal Rescaling(1/255) layer. Resizing stretches aspect ratios.

Four Conv2D/ReLU/MaxPooling blocks have 32, 64, 128 and 128 channels and 3×3 kernels
with same padding. The final pooled map is 8×8×128. The classifier is Flatten →
Dense(64, ReLU) → Dense(1, sigmoid), with Dropout 0.25 before Flatten and 0.5 before
the output. There are 765,249 trainable parameters.

Training-only augmentation consists of horizontal flips, rotation and zoom.
Inference uses `training=False`, disabling augmentation and Dropout.
The output is named `dog_probability`: p > 0.5 means dog, otherwise cat.

## Training and evaluation protocol

Adam, learning rate 0.001, binary cross-entropy, batch size 32, and at most 20
epochs. Early stopping has patience 4; the checkpoint is selected by minimum
validation loss. Test images are not passed to fit or used to select the threshold.
Full-run caching checks the dataset checksum, seed, architecture and settings.
The verification run explicitly enables fresh training rather than using the cache.

## Verification results — 24 October 2026

Fresh training in Google Colab on an NVIDIA T4, TensorFlow 2.21.0 and Python
3.13.16 completed 20 epochs. Epoch 19 had the lowest validation loss (0.2507).
The selected checkpoint achieved 90.18% test accuracy (3,380 / 3,748 correct)
and test loss 0.2388. Cat precision/recall/F1: 90.44% / 89.86% / 90.15%.
Dog precision/recall/F1: 89.93% / 90.50% / 90.21%. Each class has 1,874 test images.
The confusion matrix in cat/dog order is `[[1684, 190], [178, 1696]]`.
See [verification_run.json](verification_run.json) for unrounded metrics and
training history. No threshold or hyperparameter was chosen using these test
results. Weights are kept on the author's Drive, not distributed with the source.

## Limitations

- Only cats and dogs are supported; unknown inputs still receive one of those labels.
- Scores have not been calibrated, and confident errors are possible.
- Input resizing loses detail and distorts proportions.
- Exact-copy removal does not eliminate recompressed copies, series or photos of the same animal.
- The network may use background correlations rather than animal-specific features.
- Performance on one random test split does not demonstrate cross-source generalization.

Improvements should be selected using training/validation data. Since this test
set has been examined, an external held-out dataset is needed to assess further
improvements independently.
