# GrapeGuard AI

**A deep-learning web app that diagnoses grape leaf diseases from a single photo.**

Upload a grape leaf image to a Flask-based dashboard powered by a MobileNetV2 neural network. The app identifies the disease (or confirms the leaf is healthy), shows a confidence score, and returns a tailored treatment plan including pesticide recommendations, application timing, and duration.

---

## Supported Classes

The model classifies each image into one of four categories ([`class_names.txt`](class_names.txt)):

| Class | Description |
|---|---|
| `Grape___Black_rot` | Fungal infection causing dark, circular lesions on leaves and fruit |
| `Grape___Esca_(Black_Measles)` | Chronic wood disease causing stripe-like discoloration |
| `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Necrotic leaf spots caused by *Pseudocercospora vitis* |
| `Grape___healthy` | No disease detected |

---

## Dataset

Images are organised under `grape_dataset/` in the standard Keras `image_dataset_from_directory` layout (`train/` and `test/` subfolders, one folder per class):

| Class | Train | Test |
|---|---:|---:|
| Black rot | 1,888 | 472 |
| Esca (Black Measles) | 1,920 | 480 |
| Leaf blight (Isariopsis Leaf Spot) | 1,722 | 430 |
| Healthy | 1,692 | 423 |
| **Total** | **7,222** | **1,805** |

---

## Model

Trained in [`Untitled0.ipynb`](Untitled0.ipynb) on Google Colab (Tesla T4 GPU):

| | Detail |
|---|---|
| **Backbone** | `MobileNetV2` (ImageNet weights), `include_top=False` |
| **Head** | Random flip/rotation/zoom -> GlobalAveragePooling2D -> Dropout(0.3) -> Dense(4, softmax) |
| **Phase 1** | Classification head only (backbone frozen); Adam, lr=`1e-3`, 15 epochs |
| **Phase 2** | Fine-tune last 30 backbone layers; Adam, lr=`1e-5`, 5 epochs |
| **Input** | 224 x 224 x 3, `mobilenet_v2.preprocess_input` |
| **Parameters** | 2,263,108 (8.6 MB) |
| **Test accuracy** | **99.39 %** (test loss 0.0205) |

Output artifacts: `grape_disease_model.keras` and `class_names.txt`.

---

## Project Structure

```
grapes/
├── app.py                      # Flask app: routes, inference, disease database
├── class_names.txt             # Ordered class labels (loaded by model)
├── grape_disease_model.keras   # Trained Keras classifier
├── requirements.txt            # Python dependencies
├── Untitled0.ipynb             # Colab notebook — data prep, training, evaluation
├── grape_dataset/
│   ├── train/                  # 7,222 training images (4 subfolders)
│   └── test/                   # 1,805 held-out test images
├── templates/
│   ├── base.html               # Shared layout: navbar, flash messages, footer
│   ├── index.html              # Landing page with "How It Works" overview
│   ├── predict.html            # Upload form + results display
│   ├── about.html              # Project mission and technology
│   ├── login.html              # Mock login (session-based)
│   └── register.html           # Mock registration
└── static/
    └── uploads/                # User-uploaded images (git-ignored after initial commit)
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- ~250 MB free disk (model + dataset)

### Install

```bash
git clone https://github.com/<your-username>/grapes.git
cd grapes
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The server starts at **http://127.0.0.1:5000**.

---

## Usage

### Web Dashboard

1. Open **http://127.0.0.1:5000** and log in (any email/password combination is accepted in demo mode).
2. Navigate to **Predict** and upload a photo of a grape leaf (JPG/PNG/JPEG).
3. The app runs the MobileNetV2 model and displays:
   - **Predicted class** and **confidence %**
   - **Care tips** relevant to the diagnosis
   - **Treatment plan** (pesticide, timing, duration) when a disease is detected

### Available Routes

| Route | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | No | Landing page |
| `/login` | GET/POST | No | Session-based login |
| `/register` | GET/POST | No | Mock account creation |
| `/logout` | GET | No | Clears session |
| `/predict` | GET/POST | Yes | Upload image and view diagnosis |
| `/about` | GET | Yes | Project overview and mission |

---

## Prediction Flow

1. The user uploads an image through the `/predict` form.
2. `predict_image()` loads the image at **224x224**, runs it through the Keras model, and returns the class name + confidence.
3. The predicted class is looked up in the `DISEASE_INFO` dictionary, which contains per-disease care tips, pesticide recommendations, application timing, and reapplication duration.
4. Results are rendered on the diagnosis card with the uploaded image, disease status (healthy / diseased), and actionable advice.

---

## Limitations

- **Demo authentication only**: login and registration do not validate credentials or persist users — any submission logs you in via a session cookie.
- **Single-image upload**: the web app processes one leaf image per request.
- **Hardcoded secret key**: the Flask `secret_key` in `app.py` is set to a static value; replace it before deploying to production.
- **CPU inference only**: no GPU acceleration is configured on the Flask serving side; a single prediction typically takes 0.5-2 seconds depending on hardware.

---

## Acknowledgements

- Dataset: [Grape Leaf Disease Detection](https://github.com/lamphamit/grape-leaf-disease-detection) and [Grape Leaf Disease Detector](https://github.com/thesab/grape-leaf-disease-detector)
- [TensorFlow / Keras](https://tensorflow.org) and [MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al.)
- [Flask](https://flask.palletsprojects.com) / [Bootstrap 5](https://getbootstrap.com) / [Google Fonts Poppins](https://fonts.google.com/specimen/Poppins)
