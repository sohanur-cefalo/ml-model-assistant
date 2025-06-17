import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from common_pipeline import pipeline 
from flask_bootstrap import Bootstrap

from flask import jsonify

app = Flask(__name__)
Bootstrap(app)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def detect_pipeline_type(df, label):
    if label not in df.columns:
        return None
    unique_vals = df[label].nunique()
    if pd.api.types.is_numeric_dtype(df[label]) and unique_vals > 10:
        return "regression"
    else:
        return "classification"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath)
            columns = df.columns.tolist()

            return render_template("select.html",
                                   file=file.filename,
                                   columns=columns,
                                   pipeline_type=None,
                                   label_selected=None)
        else:
            flash("❌ Please upload a CSV file.")
            return redirect(url_for('index'))
    return render_template("index.html")

@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    file = request.form['file']
    label = request.form['label']
    pipeline_type = request.form['type']
    threshold = float(request.form.get('threshold', 0.8))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    df = pd.read_csv(filepath)

    if label not in df.columns:
        flash(f"❌ Label column '{label}' not found.")
        return redirect(url_for('index'))

    results, processed_df, max_accuracy = pipeline(filepath, label=label, type=pipeline_type, threshold=threshold)

    base_name = os.path.splitext(os.path.basename(file))[0]  # drops directory and .csv
    cleaned_file_name = f"processed_{base_name}.csv"

    # processed_df.to_csv(cleaned_file_name, index=False)
    top10 = processed_df.head(10).to_html(classes='table table-bordered table-striped', index=False)

    if max_accuracy < threshold:
        flash(f"❕ Model accuracy {max_accuracy*100:.2f}% is below {threshold*100}%. Download cleaned data for future improvement.")
    else:
        flash(f"🎉 Model accuracy {max_accuracy*100:.2f}% meets or exceeds {threshold*100}%!")

    return render_template("result.html", cleaned_file=cleaned_file_name, top10=top10)



@app.route("/detect_type", methods=["POST"])
def detect_type():
    file = request.form['file']
    label = request.form['label']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    df = pd.read_csv(filepath)

    p_type = detect_pipeline_type(df, label)
    if not p_type:
        return jsonify({"error": f"Label '{label}' not found in dataset."}), 400

    return jsonify({"pipeline_type": p_type})

@app.route("/download_file/<filename>")
def download_file(filename):
    """Serve the cleaned CSV file for download."""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash("❕ File not found.")
        return redirect(url_for('index'))
        
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
