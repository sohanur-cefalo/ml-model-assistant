import os
import pandas as pd
from common_pipeline import pipeline
from common_process import visualize_data

# Define file path and label first
# file = 'data/train_simple.csv'
# label = 'productive_day'
# threshold = 0.8  # 50% R^2 for regression
# type = 'classification'


file = 'data/train.csv'
label = 'target_label'
threshold = 0.8  # 50% R^2 for regression
type = 'classification'

# Validate label first
df = pd.read_csv(file)
if label not in df.columns:
    raise ValueError(f"Label column '{label}' not found in CSV. Available columns are: {list(df.columns)}")

# Perform EDA first (optional, if you have visualize_data implemented)
# eda_df = visualize_data(file, label)

print("Running pipeline...")

# Unpack pipeline output directly
results, processed_df, max_accuracy = pipeline(file, label=label, type=type, threshold=threshold)

print("Pipeline finished.")
print(f"Model maximum {type} score: {max_accuracy*100:.2f}%")

original_file_name = os.path.basename(file)
file_base, file_ext = os.path.splitext(original_file_name)
cleaned_file = os.path.join('output', f"{file_base}-eda{file_ext}")
processed_df.to_csv(cleaned_file, index=False)

print(processed_df.head())

if max_accuracy < threshold:
    print(f"🚀 Accuracy < {threshold*100}%. Saved cleaned data for further analysis.")
else:
    print(f"🎉 Accuracy meets or exceeds {threshold*100}%.")
