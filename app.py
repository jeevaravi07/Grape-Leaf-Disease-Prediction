
import os
import sqlite3
import uuid
import numpy as np
import tensorflow as tf

from functools import wraps
from flask import (
    Flask, request, render_template, redirect,
    url_for, flash, session
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")

DATABASE = os.path.join(BASE_DIR, "grapeguard.db")

MODEL_PATH = os.path.join(BASE_DIR, "grape_disease_model.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")

os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.secret_key = os.environ.get(
    "GRAPEGUARD_SECRET_KEY",
    "change-this-to-a-long-random-secret-key"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ==========================================================
# 2. DATABASE CONFIGURATION
# ==========================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("[INFO] Database initialized successfully!")


initialize_database()

# ==========================================================
# 3. LOAD TRAINED MODEL
# ==========================================================

model = None
class_names = []

try:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    print("\n[INFO] Model loaded successfully!")
    print("[INFO] Available classes:", class_names)

except Exception as e:

    print("\n[WARNING] Model loading failed!")
    print("[ERROR]", str(e))

    model = None
    class_names = []


# ==========================================================
# 4. DISEASE INFORMATION DATABASE
# ==========================================================

DISEASE_INFO = {

    "Grape___Black_rot": {
        "status": "Diseased",

        "tips": (
            "Remove and destroy infected plant material. "
            "Maintain good air circulation through proper "
            "canopy management. Avoid unnecessary overhead watering."
        ),

        "pesticides": (
            "Fungicide options may include locally registered "
            "products containing Mancozeb, Myclobutanil, or copper."
        ),

        "timing": (
            "Disease management should begin early in the growing "
            "season, based on local disease pressure and expert advice."
        ),

        "duration": (
            "Follow the product label and local agricultural "
            "recommendations for application intervals."
        )
    },

    "Grape___Esca_(Black_Measles)": {
        "status": "Diseased",

        "tips": (
            "Remove severely affected wood where appropriate. "
            "Disinfect pruning tools between plants and avoid "
            "pruning during wet conditions."
        ),

        "pesticides": (
            "There is no universally effective curative fungicide "
            "for established Esca infections. Consult a plant pathologist."
        ),

        "timing": (
            "Inspect vines regularly and manage pruning wounds "
            "using locally recommended practices."
        ),

        "duration": (
            "Long-term vineyard monitoring and sanitation "
            "are important."
        )
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "Diseased",

        "tips": (
            "Remove fallen infected leaves, improve vineyard "
            "drainage, and maintain sufficient airflow around plants."
        ),

        "pesticides": (
            "Use only locally registered fungicides recommended "
            "by an agricultural specialist for the confirmed disease."
        ),

        "timing": (
            "Inspect plants regularly, especially during "
            "conditions favorable to fungal development."
        ),

        "duration": (
            "Follow the selected product label and local "
            "agricultural recommendations."
        )
    },

    "Grape___healthy": {
        "status": "Healthy",

        "tips": (
            "Maintain regular watering and balanced fertilization. "
            "Continue routine pruning, canopy management, and "
            "periodic inspection of grape leaves."
        ),

        "pesticides": "No disease-specific pesticide is indicated.",

        "timing": "Not applicable.",

        "duration": "Not applicable."
    }
}


# ==========================================================
# 5. IMAGE VALIDATION
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================================
# 6. IMAGE PREDICTION FUNCTION
# ==========================================================

def predict_image(img_path):

    if model is None:
        raise RuntimeError(
            "The trained grape disease model is not available."
        )

    if not class_names:
        raise RuntimeError(
            "Class names are not available."
        )

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predictions = np.asarray(predictions)

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError("Unexpected model output shape.")

    if predictions.shape[1] != len(class_names):
        raise ValueError(
            "Model output count does not match class_names.txt. "
            "Check the class order and number of classes."
        )

    predicted_class_idx = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        np.max(predictions[0])
    ) * 100

    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence


# ==========================================================
# 7. LOGIN REQUIRED DECORATOR
# ==========================================================

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if not session.get("logged_in"):

            flash(
                "Please log in to access this page.",
                "danger"
            )

            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated_function


# ==========================================================
# 8. HOME PAGE
# ==========================================================

@app.route("/")
def index():

    return render_template("index.html")


# ==========================================================
# 9. USER REGISTRATION
# ==========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()

        email = request.form.get("email", "").strip().lower()

        password = request.form.get("password", "")

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # ------------------------------------------
        # VALIDATION
        # ------------------------------------------

        if not name or not email or not password:

            flash(
                "Please fill in all required fields.",
                "danger"
            )

            return redirect(url_for("register"))

        if len(name) < 2:

            flash(
                "Please enter a valid full name.",
                "danger"
            )

            return redirect(url_for("register"))

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "danger"
            )

            return redirect(url_for("register"))

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return redirect(url_for("register"))

        # ------------------------------------------
        # HASH PASSWORD
        # ------------------------------------------

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()

        try:

            existing_user = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            if existing_user:

                flash(
                    "This email is already registered. Please login.",
                    "warning"
                )

                return redirect(url_for("register"))

            conn.execute("""
                INSERT INTO users (name, email, password)
                VALUES (?, ?, ?)
            """, (
                name,
                email,
                hashed_password
            ))

            conn.commit()

            flash(
                "Registration successful! Please login with your account.",
                "success"
            )

            return redirect(url_for("login"))

        except sqlite3.Error as e:

            print("[DATABASE ERROR]", e)

            flash(
                "Registration failed due to a database error.",
                "danger"
            )

            return redirect(url_for("register"))

        finally:

            conn.close()

    return render_template("register.html")


# ==========================================================
# 10. USER LOGIN
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        if not email or not password:

            flash(
                "Please enter your email and password.",
                "danger"
            )

            return redirect(url_for("login"))

        conn = get_db_connection()

        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session.clear()

            session["logged_in"] = True
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            flash(
                f"Welcome back, {user['name']}!",
                "success"
            )

            return redirect(url_for("predict"))

        else:

            flash(
                "Invalid email address or password.",
                "danger"
            )

            return redirect(url_for("login"))

    return render_template("login.html")


# ==========================================================
# 11. LOGOUT
# ==========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out successfully.",
        "info"
    )

    return redirect(url_for("login"))


# ==========================================================
# 12. ABOUT PAGE
# ==========================================================

@app.route("/about")
@login_required
def about():

    return render_template("about.html")


# ==========================================================
# 13. GRAPE DISEASE PREDICTION
# ==========================================================

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        # ------------------------------------------
        # CHECK FILE
        # ------------------------------------------

        if "file" not in request.files:

            flash(
                "No image file was selected.",
                "danger"
            )

            return redirect(url_for("predict"))

        file = request.files["file"]

        if file.filename == "":

            flash(
                "Please select an image.",
                "warning"
            )

            return redirect(url_for("predict"))

        if not allowed_file(file.filename):

            flash(
                "Only JPG, JPEG, and PNG images are allowed.",
                "danger"
            )

            return redirect(url_for("predict"))

        if model is None:

            flash(
                "The prediction model is unavailable. "
                "Please check the model files.",
                "danger"
            )

            return redirect(url_for("predict"))

        # ------------------------------------------
        # SAVE IMAGE WITH UNIQUE FILENAME
        # ------------------------------------------

        original_filename = secure_filename(file.filename)

        extension = original_filename.rsplit(".", 1)[1].lower()

        filename = f"{uuid.uuid4().hex}.{extension}"

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        try:

            file.save(filepath)

            # --------------------------------------
            # PREDICT IMAGE
            # --------------------------------------

            pred_class, confidence = predict_image(filepath)

            # --------------------------------------
            # GET DISEASE INFORMATION
            # --------------------------------------

            info = DISEASE_INFO.get(
                pred_class,
                {
                    "status": "Unknown",

                    "tips": (
                        "The predicted class does not have "
                        "specific management information available. "
                        "Consult an agricultural expert."
                    ),

                    "pesticides": "Consult an agricultural expert.",

                    "timing": "Based on expert recommendation.",

                    "duration": "Based on expert recommendation."
                }
            )

            # --------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------

            return render_template(
                "predict.html",

                filename=filename,

                prediction=pred_class,

                confidence=round(confidence, 2),

                info=info
            )

        except Exception as e:

            print("[PREDICTION ERROR]", str(e))

            flash(
                "Unable to process the image. "
                "Please upload a valid grape leaf image.",
                "danger"
            )

            return redirect(url_for("predict"))

    return render_template("predict.html")


# ==========================================================
# 14. IMAGE UPLOAD SIZE ERROR
# ==========================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "Image size is too large. Maximum allowed size is 10 MB.",
        "danger"
    )

    return redirect(url_for("predict"))


# ==========================================================
# 15. APPLICATION START
# ==========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("       GRAPEGUARD AI APPLICATION")
    print("========================================")

    print("[INFO] Base directory:", BASE_DIR)

    print("[INFO] Templates:", TEMPLATE_DIR)

    print("[INFO] Upload folder:", UPLOAD_FOLDER)

    print("[INFO] Database:", DATABASE)

    print("[INFO] Model loaded:", model is not None)

    print("[INFO] Classes:", class_names)

    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
