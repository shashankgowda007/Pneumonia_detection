import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import hashlib
import pydicom
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

class DataQualityChecker:
    def __init__(self, data_dir='data'):
        self.base_dir = data_dir
        self.normal_dir = os.path.join(data_dir, 'normal')
        self.pneumonia_dir = os.path.join(data_dir, 'pneumonia')
        self.report = {
            'basic_stats': {},
            'quality_issues': [],
            'duplicates': [],
            'corrupt_files': []
        }
        
    def load_dataframe(self):
        """Load or create dataframe with image metadata"""
        if os.path.exists('data_summary.csv'):
            return pd.read_csv('data_summary.csv')
        return self._create_dataframe()
    
    def _create_dataframe(self):
        """Create dataframe with image paths and metadata"""
        data = []
        for label, dir_path in [('normal', self.normal_dir), 
                               ('pneumonia', self.pneumonia_dir)]:
            for img_file in os.listdir(dir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    data.append({
                        'path': os.path.join(dir_path, img_file),
                        'label': label,
                        'file_type': img_file.split('.')[-1].lower()
                    })
        return pd.DataFrame(data)
    
    def validate_images(self, df):
        """Check for corrupt/invalid images"""
        print("\nValidating image files...")
        corrupt_files = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                if row['file_type'] == 'dcm':
                    pydicom.dcmread(row['path'])
                else:
                    img = cv2.imread(row['path'])
                    if img is None:
                        raise ValueError("Failed to load image")
            except Exception as e:
                corrupt_files.append({
                    'path': row['path'],
                    'error': str(e)
                })
        
        self.report['corrupt_files'] = corrupt_files
        return corrupt_files
    
    def check_dimensions(self, df):
        """Analyze image dimensions consistency"""
        print("\nChecking image dimensions...")
        dim_stats = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                if row['file_type'] == 'dcm':
                    import pydicom
                    img = pydicom.dcmread(row['path']).pixel_array
                else:
                    img = cv2.imread(row['path'])
                
                dim_stats.append({
                    'path': row['path'],
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'channels': img.shape[2] if len(img.shape) == 3 else 1
                })
            except:
                continue
        
        dim_df = pd.DataFrame(dim_stats)
        self.report['dimension_stats'] = {
            'height': dim_df['height'].describe().to_dict(),
            'width': dim_df['width'].describe().to_dict(),
            'channels': dim_df['channels'].value_counts().to_dict()
        }
        return dim_df
    
    def find_duplicates(self, df, threshold=0.95):
        """Find duplicate or near-duplicate images"""
        print("\nChecking for duplicate images...")
        hashes = {}
        duplicates = []
        
        # First pass: exact duplicates via hash
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if row['file_type'] == 'dcm':
                continue  # Skip DICOM for now
                
            try:
                with open(row['path'], 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if file_hash in hashes:
                    duplicates.append({
                        'original': hashes[file_hash],
                        'duplicate': row['path'],
                        'similarity': 1.0,
                        'method': 'hash'
                    })
                else:
                    hashes[file_hash] = row['path']
            except:
                continue
        
        # Second pass: near-duplicates via SSIM
        if len(df) < 1000:  # SSIM is computationally expensive
            paths = df[~df['path'].isin([d['duplicate'] for d in duplicates])]['path'].tolist()
            for i in tqdm(range(len(paths))):
                for j in range(i+1, len(paths)):
                    try:
                        img1 = cv2.imread(paths[i])
                        img2 = cv2.imread(paths[j])
                        
                        if img1.shape != img2.shape:
                            continue
                            
                        # Resize if too large for SSIM
                        if img1.shape[0] > 512 or img1.shape[1] > 512:
                            img1 = cv2.resize(img1, (256, 256))
                            img2 = cv2.resize(img2, (256, 256))
                            
                        similarity = ssim(img1, img2, 
                                        multichannel=True,
                                        channel_axis=2)
                        
                        if similarity > threshold:
                            duplicates.append({
                                'original': paths[i],
                                'duplicate': paths[j],
                                'similarity': similarity,
                                'method': 'ssim'
                            })
                    except:
                        continue
        
        self.report['duplicates'] = duplicates
        return duplicates
    
    def generate_report(self, output_dir='reports'):
        """Generate comprehensive quality report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save report as JSON
        import json
        with open(os.path.join(output_dir, 'data_quality_report.json'), 'w') as f:
            json.dump(self.report, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(output_dir)
        
        print(f"\nQuality report generated in {output_dir} directory")
    
    def _generate_visualizations(self, output_dir):
        """Create visualizations for quality report"""
        # Dimension distribution plots
        if 'dimension_stats' in self.report:
            dim_df = pd.DataFrame({
                'height': [self.report['dimension_stats']['height']['mean']],
                'width': [self.report['dimension_stats']['width']['mean']]
            })
            
            plt.figure(figsize=(10, 5))
            sns.barplot(data=dim_df)
            plt.title('Average Image Dimensions')
            plt.savefig(os.path.join(output_dir, 'dimensions.png'))
            plt.close()
        
        # Class distribution
        if 'basic_stats' in self.report and 'class_distribution' in self.report['basic_stats']:
            class_df = pd.DataFrame.from_dict(
                self.report['basic_stats']['class_distribution'],
                orient='index'
            ).reset_index()
            class_df.columns = ['class', 'count']
            
            plt.figure(figsize=(8, 5))
            sns.barplot(data=class_df, x='class', y='count')
            plt.title('Class Distribution')
            plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
            plt.close()

if __name__ == "__main__":
    checker = DataQualityChecker()
    df = checker.load_dataframe()
    
    # Basic stats
    checker.report['basic_stats']['class_distribution'] = df['label'].value_counts().to_dict()
    checker.report['basic_stats']['file_types'] = df['file_type'].value_counts().to_dict()
    
    # Run quality checks
    checker.validate_images(df)
    dim_df = checker.check_dimensions(df)
    duplicates = checker.find_duplicates(df)
    
    # Generate final report
    checker.generate_report()
