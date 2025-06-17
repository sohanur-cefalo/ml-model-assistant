import os
import pandas as pd
from common_pipeline import pipeline
from common_process import visualize_data

# Define file path and label first
file = 'data/Housing.csv'
label = 'price'
threshold = 0.5  # 80% accuracy
type = 'regression'

# Validate label first
df = pd.read_csv(file)
if label not in df.columns:
    raise ValueError(f"Label column '{label}' not found in CSV. Available columns are: {list(df.columns)}")

# Perform EDA first
# eda_df = visualize_data(file, label)  # visualize_data should return cleaned/preprocessed df

print("here main file...")
output = pipeline(df, label=label, type=type, threshold=threshold)

print(output)

if isinstance(output, tuple):
    # If we got a tuple, it's (results, processed_df)
    results, processed_df = output
    
    original_file_name = os.path.basename(file)
    file_base, file_ext = os.path.splitext(original_file_name)
    cleaned_file = os.path.join('output', f"{file_base}-eda{file_ext}")
    processed_df.to_csv(cleaned_file, index=False)
    print(f"🚀 Accuracy < {threshold*100}%. Saved cleaned data for further analysis.")
else:
    # Otherwise it's just results
    results = output
    print(f"🎉 Accuracy is above the threshold.")