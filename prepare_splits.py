import os
import shutil
import pandas as pd

def create_split_directories():
    """Create directory structure for train/val/test splits"""
    for split in ['train', 'val', 'test']:
        for label in ['normal', 'pneumonia']:
            os.makedirs(f'{split}_split/{label}', exist_ok=True)

def organize_images():
    """Organize images into split directories based on CSV files"""
    # Verify and read split CSVs
    split_files = ['train_split.csv', 'val_split.csv', 'test_split.csv']
    missing = [f for f in split_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing required split files: {missing}\n"
            "Run test_splits.py first to generate these files."
        )
    
    train_df = pd.read_csv('train_split.csv')
    val_df = pd.read_csv('val_split.csv')
    test_df = pd.read_csv('test_split.csv')

    # Process each split
    for split_name, df in [('train', train_df), 
                          ('val', val_df), 
                          ('test', test_df)]:
        print(f"Processing {split_name} split...")
        
        for _, row in df.iterrows():
            src_path = row['path']
            # Extract label from path more reliably
            label = os.path.basename(os.path.dirname(src_path))
            if label not in ['normal', 'pneumonia']:
                raise ValueError(f"Invalid label in path: {src_path}")
            filename = os.path.basename(src_path)
            dest_path = f"{split_name}_split/{label}/{filename}"
            
            # Copy image to new location
            try:
                shutil.copy2(src_path, dest_path)
            except FileNotFoundError:
                print(f"Warning: Source file not found - {src_path}")
                continue

if __name__ == "__main__":
    print("Creating directory structure...")
    create_split_directories()
    
    print("Organizing images into splits...")
    organize_images()
    
    print("Done! Split directories are ready.")
