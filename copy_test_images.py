import os
import shutil
import pandas as pd

def process_split(df, split_dir):
    """Helper function to process a single split"""
    for _, row in df.iterrows():
        src_path = row['path']
        filename = os.path.basename(src_path)
        
        # Determine destination directory based on source
        if 'normal' in src_path:
            dest_dir = os.path.join(split_dir, 'normal')
        elif 'pneumonia' in src_path:
            dest_dir = os.path.join(split_dir, 'pneumonia')
        else:
            continue  # skip if path doesn't match expected structure
            
        dest_path = os.path.join(dest_dir, filename)
        
        # Copy the file - try original path first, then try without 'images/' if needed
        try:
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        except FileNotFoundError:
            # Try alternative path by removing 'images/' if present
            if 'images/' in src_path:
                alt_src_path = src_path.replace('images/', '')
                try:
                    shutil.copy2(alt_src_path, dest_path)
                    print(f"Copied {alt_src_path} to {dest_path} (using alternative path)")
                except FileNotFoundError as e:
                    print(f"Error: Could not find file at either {src_path} or {alt_src_path}")
                    continue
            else:
                print(f"Error: Could not find file at {src_path}")
                continue

# Define paths
data_dir = 'data'
test_split_dir = 'test_split'
val_split_dir = 'val_split'
train_split_dir = 'train_split'
test_csv_path = 'test_split.csv'
val_csv_path = 'val_split.csv'
train_csv_path = 'train_split.csv'

# Create split directories if they don't exist
for split_dir in [test_split_dir, val_split_dir, train_split_dir]:
    os.makedirs(os.path.join(split_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'pneumonia'), exist_ok=True)

# Process test split
print("Processing test split...")
df_test = pd.read_csv(test_csv_path)
process_split(df_test, test_split_dir)

# Process validation split if CSV exists
if os.path.exists(val_csv_path):
    print("\nProcessing validation split...")
    df_val = pd.read_csv(val_csv_path)
    process_split(df_val, val_split_dir)
else:
    print("\nValidation split CSV not found, skipping")

# Process training split if CSV exists
if os.path.exists(train_csv_path):
    print("\nProcessing training split...")
    df_train = pd.read_csv(train_csv_path)
    process_split(df_train, train_split_dir)
else:
    print("\nTraining split CSV not found, skipping")

print("Finished copying all split images")
