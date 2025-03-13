import os
import sys
import numpy as np
import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# print cwd
print(os.getcwd())

# Define directories
processed_dir = "data/processed/"

# List to hold data
data = []

# List files directly in the processed directory (no subdirectories)
files = [f for f in os.listdir(processed_dir) if f.endswith(".npz")]

for file in files:
    processed_file_path = os.path.join(processed_dir, file)

    raw_data = np.load(processed_file_path)

    labels = raw_data["labels"]
    labels_str = ",".join(str(l) for l in labels)

    print(labels_str)

    # Append to data list
    data.append({"file_path": processed_file_path, "labels": labels_str})

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv = "data/labels.csv"
df.to_csv(output_csv, index=False)
print(f"Saved label mappings to {output_csv}")
