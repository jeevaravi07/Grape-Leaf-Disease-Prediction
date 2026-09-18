# Grape Leaf Disease Detector

A transfer-learning image classifier (MobileNetV2) that identifies common grape
leaf diseases from a photo, plus a Streamlit app and CLI scripts for running
predictions on single images or whole folders.

## Classes

The model classifies a grape leaf image into one of four categories
(from [`class_names.txt`](class_names.txt)):

- `Grape___Black_rot`
- `Grape___Esca_(Black_Measles)`
- `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
- `Grape___healthy`

## Project structure

```
grapes/
├── app.py                  # Streamlit web app — upload one or many images, get predictions
├── predict.py               # CLI: predict every image in a folder
├── predictfolder.py         # Duplicate of predict.py
├── class_names.txt          # Ordered class labels used by the model
├── grape_disease_model.keras  # Trained MobileNetV2 model (Keras format)
├── grape_dataset/
│   ├── train/                # Training images, one subfolder per class
│   └── test/                 # Held-out test images, one subfolder per class
├── Untitled0.ipynb           # Colab notebook: data prep, training, evaluation
├── 12k images.zip                          # Raw dataset archive
└── drive-download-20260829T105943Z-1-001.zip  # Raw dataset archive (Drive export)
```

> **Note:** `predict.py` and `predictfolder.py` are currently identical files.

## Dataset

Images live under `grape_dataset/`, split into `train/` and `test/`, each with
one subfolder per class (Keras `image_dataset_from_directory` layout).

| Class | Train | Test |
|---|---:|---:|
| Black rot | 1,888 | 472 |
| Esca (Black Measles) | 1,920 | 480 |
| Leaf blight (Isariopsis Leaf Spot) | 1,722 | 430 |
| Healthy | 1,692 | 423 |

## Model

Defined and trained in `Untitled0.ipynb` (originally run on Google Colab):

1. **Backbone:** `MobileNetV2` pretrained on ImageNet, `include_top=False`.
2. **Head:** data augmentation (random flip/rotation/zoom) → global average
   pooling → dropout (0.3) → dense softmax over the 4 classes.
3. **Phase 1:** train the classification head only, backbone frozen
   (`Adam`, lr `1e-3`, default 15 epochs).
4. **Phase 2:** fine-tune the last 30 layers of the backbone
   (`Adam`, lr `1e-5`, default 5 epochs).
5. Images are resized to `224x224` and preprocessed with
   `mobilenet_v2.preprocess_input`.
6. Output artifacts: `grape_disease_model.keras` and `class_names.txt`.

The notebook also writes out standalone `train.py` and `evaluate.py` scripts
(via `%%writefile`) that mirror this pipeline, e.g.:

```bash
python train.py --data_dir grape_dataset/train --epochs 15 --fine_tune_epochs 5
python evaluate.py --test_dir grape_dataset/test --model grape_disease_model.keras
```

## Setup

```bash
pip install streamlit tensorflow pillow numpy
```

## Usage

### Web app

```bash
streamlit run app.py
```

Upload one or many grape leaf photos (use Ctrl+A in the file dialog to select
every image in a folder at once). The app shows a per-image prediction with
confidence, plus a summary count per class.

### CLI (predict a folder of images)

```bash
python predict.py --folder path/to/images
```

Optional flags:

```bash
python predict.py --folder path/to/images \
  --model grape_disease_model.keras \
  --class_names class_names.txt
```

Prints a per-image prediction (`filename: class (confidence%)`) followed by a
summary count per class.
