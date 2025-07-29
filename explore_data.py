import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pydicom
import cv2

# Set up paths
base_dir = 'data'
normal_dir = os.path.join(base_dir, 'normal')
pneumonia_dir = os.path.join(base_dir, 'pneumonia')  # Corrected path
pneumonia_metadata = os.path.join(base_dir, 'pneumonia')

# Get file counts
normal_count = len([f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.dcm'))])
pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.dcm'))])

print(f"Normal images: {normal_count}")
print(f"Pneumonia images: {pneumonia_count}")

# Create dataframe with image paths and labels
data = []
for label, dir_path in [('normal', normal_dir), ('pneumonia', pneumonia_dir)]:
    # Skip metadata files in pneumonia directory
    if label == 'pneumonia' and not dir_path.endswith('images'):
        continue
    for img_file in os.listdir(dir_path):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.dcm')):
            data.append({
                'path': os.path.join(dir_path, img_file),
                'label': label,
                'file_type': img_file.split('.')[-1]
            })

df = pd.DataFrame(data)

# Basic statistics
print("\nData Summary:")
print(df['label'].value_counts())
print("\nFile Types:")
print(df['file_type'].value_counts())

# Sample image analysis
def analyze_sample_images(df, n_samples=3):
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
    fig.suptitle('Sample Images from Each Class')
    
    for i, (label, group) in enumerate(df.groupby('label')):
        samples = group.sample(n_samples)
        for j, (_, row) in enumerate(samples.iterrows()):
            try:
                if row['file_type'] == 'dcm':
                    img = pydicom.dcmread(row['path']).pixel_array
                else:
                    img = cv2.imread(row['path'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[i,j].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[i,j].set_title(f"{label}\n{row['path'].split('/')[-1]}")
                axes[i,j].axis('off')
                
                # Print image stats
                print(f"\nImage: {row['path']}")
                print(f"Shape: {img.shape}")
                print(f"Min pixel value: {img.min()}")
                print(f"Max pixel value: {img.max()}")
                print(f"Mean pixel value: {img.mean():.2f}")
                
            except Exception as e:
                print(f"Error loading {row['path']}: {str(e)}")
    
    plt.tight_layout()
    plt.show()

# Run analysis
analyze_sample_images(df)

# Save the dataframe for future use
df.to_csv('data_summary.csv', index=False)
print("\nData summary saved to data_summary.csv")
