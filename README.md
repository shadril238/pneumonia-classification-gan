
# Pneumonia Classification with GAN Augmentation

Chest X-ray pneumonia classification using pretrained vision models and DCGAN-generated synthetic pneumonia images to reduce class imbalance.

## Highlights
- Preprocessing: resize to **224×224**, convert to **3-channel grayscale**, ImageNet normalization
- Train **DCGAN** on **train/PNEUMONIA** only, then generate synthetic images (**~5000**)
- Compare models with and without augmentation
- Metrics: Accuracy, Recall, F1, AUROC + confusion matrices + ROC curves

## Architecture
- **GAN (DCGAN)**: trains on pneumonia images only, produces synthetic pneumonia images saved to a folder
- **Classifier**: pretrained backbone from `timm` with `num_classes=1` (binary logit) + `BCEWithLogitsLoss`
- **Mixed training**: real train set + synthetic pneumonia images (validation/test remain real only)

## Models
- EfficientNet-B0 (baseline)
- EfficientNet-B0 (standard augmentation)
- Swin Tiny (GAN-augmented)
- EfficientNet-B0 (GAN-augmented)

