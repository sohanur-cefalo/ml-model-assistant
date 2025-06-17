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

    output = pipeline(filepath, label=label, type=pipeline_type, threshold=threshold)

    if isinstance(output, tuple):
        results, processed_df = output
        cleaned_file = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + file)
        processed_df.to_csv(cleaned_file, index=False)
        flash(f"🚀 Accuracy < {threshold*100}%. Saved cleaned data.")
        return send_file(cleaned_file, as_attachment=True)
    else:
        results = output
        max_score = 0.0  # to track max Accuracy or R2

        for k, res in results.items():
            if k == 'type':
                continue
            if results['type'] == 'classification':
                score = res['accuracy']
            else:
                score = res['R2']

            if score > max_score:
                max_score = score

        print([res for k, res in results.items() if k != 'type'])

        flash(f"🎉 Score {max_score*100:.2f}% meets or exceeds {threshold*100}%.")
        return redirect(url_for('index'))


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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
