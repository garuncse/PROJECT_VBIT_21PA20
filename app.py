from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file is a CSV
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

# Handle file upload and display analysis results
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform analysis on the dataset
        try:
            # Read the CSV file using Pandas
            data = pd.read_csv(filepath)
            
            # Perform some basic analysis
            result = analyze_data(data)
            
            # Return the result to the frontend
            return render_template("index.html", results=result)
        
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "error", "message": "Invalid file format. Only CSV is allowed."})

# A simple function to analyze the dataset (e.g., displaying the shape, basic statistics)
def analyze_data(data):
    summary = {
        "columns": list(data.columns),
        "shape": data.shape,
        "head": data.head().to_html(),  # Display the first few rows
        "describe": data.describe().to_html(),  # Basic statistics
    }
    return summary

@app.route("/train", methods=["POST"])
def train_model():
    try:
        # Call the training pipeline script
        process = subprocess.run(
            ["python3", "train_pipeline.py"], capture_output=True, text=True, check=True
        )
        return jsonify({
            "status": "success",
            "message": "Model trained successfully!",
            "details": process.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Error during training!",
            "details": e.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/test", methods=["POST"])
def test_model():
    try:
        # Call the testing pipeline script
        process = subprocess.run(
            ["python3", "test_pipeline.py"], capture_output=True, text=True, check=True
        )
        return jsonify({
            "status": "success",
            "message": "Testing completed!",
            "details": process.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "message": "Error during testing!",
            "details": e.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
