import os
import numpy as np
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, flash, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename

# --- CRITICAL FIX: Force Flask to use absolute paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "super_secret_key"

# Ensure directories exist
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model and Class Names
MODEL_PATH = os.path.join(BASE_DIR, 'grape_disease_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.txt')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Model loaded successfully! Classes: {class_names}")
except Exception as e:
    print(f"WARNING: Error loading model or class names: {e}")
    model = None
    class_names = []

# Disease Database
DISEASE_INFO = {
    "Grape___Black_rot": {
        "status": "Diseased",
        "tips": "Prune and destroy infected plant parts. Ensure good air circulation through canopy management. Avoid overhead watering.",
        "pesticides": "Mancozeb, Myclobutanil, or Copper-based fungicides.",
        "timing": "Apply early in the spring, right before blooms open.",
        "duration": "Reapply every 7-14 days until grapes reach pea-size."
    },
    "Grape___Esca_(Black_Measles)": {
        "status": "Diseased",
        "tips": "Remove and burn infected wood. Clean pruning tools between cuts. Avoid pruning during wet weather.",
        "pesticides": "Fosetyl-aluminum or Thiophanate-methyl (mostly preventative).",
        "timing": "Apply immediately after winter pruning.",
        "duration": "Single application over fresh pruning wounds."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "Diseased",
        "tips": "Rake up and destroy fallen leaves to reduce overwintering fungi. Improve vineyard drainage.",
        "pesticides": "Azoxystrobin or broad-spectrum fungicides.",
        "timing": "Apply at the first sign of leaf spots in mid-to-late summer.",
        "duration": "Reapply every 10-14 days if wet weather persists."
    },
    "Grape___healthy": {
        "status": "Healthy",
        "tips": "Maintain regular watering and fertilization schedules. Continue routine canopy management.",
        "pesticides": "None required.",
        "timing": "N/A",
        "duration": "N/A"
    }
}

def predict_image(img_path):
    if model is None:
        return "Model not loaded", 0.0
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    if predicted_class_idx < len(class_names):
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = "Unknown"
        
    return predicted_class, confidence


# --- LOGIN REQUIRED DECORATOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Sets a session variable to remember the user is logged in
        session['logged_in'] = True
        flash('Login successful! Welcome to your dashboard.', 'success')
        return redirect(url_for('predict'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    # Clears the session variable to log the user out
    session.pop('logged_in', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/about')
@login_required  # Protects this route
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required  # Protects this route
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            pred_class, confidence = predict_image(filepath)
            
            info = DISEASE_INFO.get(pred_class, {
                "status": "Unknown",
                "tips": "Consult a local agricultural expert.",
                "pesticides": "N/A", "timing": "N/A", "duration": "N/A"
            })
            
            return render_template('predict.html', 
                                   filename=filename, 
                                   prediction=pred_class, 
                                   confidence=confidence,
                                   info=info)
            
    return render_template('predict.html')


if __name__ == '__main__':
    print(f"\n[INFO] Flask is looking for templates in: {TEMPLATE_DIR}")
    print(f"[INFO] Make sure your HTML files are exactly inside that folder!\n")
    app.run(host='127.0.0.1', port=5000, debug=True)