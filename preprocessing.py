import os
import numpy as np
import cv2
import pydicom
import pandas as pd
from PIL import Image
from skimage import exposure
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split

class MedicalImagePreprocessor:
    def __init__(self, 
                 output_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True,
                 equalize_hist: bool = False,
                 augment: bool = False):
        """
        Initialize medical image preprocessor.
        
        Args:
            output_size: Target size for resizing images (height, width)
            normalize: Whether to normalize pixel values to [0,1]
            equalize_hist: Whether to apply histogram equalization
            augment: Whether to enable data augmentation
        """
        self.output_size = output_size
        self.normalize = normalize
        self.equalize_hist = equalize_hist
        self.augment = augment
        
    def load_image(self, file_path: str) -> np.ndarray:
        """Load image from file path, handling both DICOM and standard formats"""
        if file_path.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(file_path)
            img = dicom.pixel_array
            # Handle potential rescale slope/intercept
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                img = img * dicom.RescaleSlope + dicom.RescaleIntercept
        else:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to a single image"""
        # Convert to float32 if needed
        if img.dtype != np.float32:
            img = img.astype(np.float32)
            
        # Resize image
        img = cv2.resize(img, (self.output_size[1], self.output_size[0]))
        
        # Normalize pixel values
        if self.normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-7)
            
        # Histogram equalization (for grayscale only)
        if self.equalize_hist and len(img.shape) == 2:
            img = exposure.equalize_hist(img)
            
        return img
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str,
                         file_types: Tuple[str] = ('.png', '.jpg', '.jpeg', '.dcm')):
        """
        Process all images in a directory and save to output directory.
        
        Args:
            input_dir: Path to directory with input images
            output_dir: Path to save processed images
            file_types: Tuple of file extensions to process
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(file_types)
        ]
        
        # Process each image
        for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
            try:
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, img_file)
                
                # Load and preprocess
                img = self.load_image(input_path)
                processed_img = self.preprocess_image(img)
                
                # Save processed image (convert to uint8 for standard formats)
                if img_file.lower().endswith('.dcm'):
                    np.save(output_path.replace('.dcm', '.npy'), processed_img)
                else:
                    processed_img = (processed_img * 255).astype(np.uint8)
                    Image.fromarray(processed_img).save(output_path)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

    def process_dataframe(self, 
                         df: pd.DataFrame,
                         output_base_dir: str = 'processed_data'):
        """
        Process images from a dataframe with 'path' and 'label' columns.
        
        Args:
            df: DataFrame containing image paths and labels
            output_base_dir: Base directory for processed data
        """
        # Group by label and process each group
        for label, group in df.groupby('label'):
            input_dir = os.path.dirname(group['path'].iloc[0])
            output_dir = os.path.join(output_base_dir, label)
            self.process_directory(input_dir, output_dir)

    def create_patient_splits(self,
                            input_dirs: List[str],
                            test_size: float = 0.2,
                            val_size: float = 0.1,
                            random_state: int = 42) -> Dict[str, Dict[str, List[str]]]:
        """
        Create patient-aware stratified splits to prevent data leakage.
        
        Args:
            input_dirs: List of directories containing images
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing paths and labels for each split
        """
        patient_data = {}
        
        for input_dir in input_dirs:
            print(f"\n--- Processing directory: {input_dir} ---")
            
            # Normalize path for consistent comparison
            norm_path = os.path.normpath(input_dir).lower()
            
            # Determine label based on directory path (exact match)
            base_dir = os.path.basename(norm_path)
            if base_dir == 'pneumonia':
                label = 'pneumonia'
            elif base_dir == 'normal':
                label = 'normal'
            else:
                raise ValueError(f"Cannot determine label for directory: {input_dir}")
            
            print(f"Directory label: {label}")
            
            if not os.path.exists(input_dir):
                print(f"ERROR: Directory does not exist: {input_dir}")
                continue
                
            # Collect all image files from this directory
            image_files = []
            for root, _, files in os.walk(input_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                        full_path = os.path.join(root, f)
                        image_files.append(full_path)
            
            print(f"Found {len(image_files)} images in {input_dir}")
            
            # Process each image file
            for img_path in image_files:
                img_file = os.path.basename(img_path)
                
                # Extract patient ID from filename (e.g., '00000001_000.png')
                patient_id = img_file.split('_')[0]
                if not patient_id.isdigit():
                    raise ValueError(f"Invalid patient ID format in filename: {img_file}")
                
                # Initialize or update patient data
                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'paths': [],
                        'label': label
                    }
                else:
                    # If patient exists but current directory is pneumonia, update label
                    if label == 'pneumonia':
                        patient_data[patient_id]['label'] = label
                    elif patient_data[patient_id]['label'] != label:
                        print(f"WARNING: Patient {patient_id} has conflicting labels: "
                              f"{patient_data[patient_id]['label']} vs {label}")
                
                # Add image path to patient data
                patient_data[patient_id]['paths'].append(img_path)
        
        # Debug: Print patient data distribution
        print("\nPatient Data Distribution:")
        label_counts = {'normal': 0, 'pneumonia': 0}
        patient_counts = {'normal': 0, 'pneumonia': 0}
        
        for patient_id, data in patient_data.items():
            label = data['label']
            label_counts[label] += len(data['paths'])
            patient_counts[label] += 1
            
        print(f"Normal: {label_counts['normal']} images from {patient_counts['normal']} patients")
        print(f"Pneumonia: {label_counts['pneumonia']} images from {patient_counts['pneumonia']} patients")
        
        # Verify we have both classes
        if patient_counts['normal'] == 0:
            raise ValueError("No normal patients found! Check your directory structure.")
        if patient_counts['pneumonia'] == 0:
            raise ValueError("No pneumonia patients found! Check your directory structure.")
        
        # Convert to lists for stratified splitting
        patients = list(patient_data.keys())
        labels = [patient_data[p]['label'] for p in patients]
        
        print(f"\nTotal patients: {len(patients)}")
        print(f"Labels distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # First split into train+val and test sets
        train_val_patients, test_patients = train_test_split(
            patients, test_size=test_size, stratify=labels, random_state=random_state)
        
        # Then split train_val into train and val
        train_labels = [patient_data[p]['label'] for p in train_val_patients]
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_size/(1-test_size), 
            stratify=train_labels, random_state=random_state)
        
        # Collect all paths for each split
        splits = {
            'train': {'paths': [], 'labels': []},
            'val': {'paths': [], 'labels': []},
            'test': {'paths': [], 'labels': []}
        }
        
        # Populate splits
        for patient in train_patients:
            splits['train']['paths'].extend(patient_data[patient]['paths'])
            splits['train']['labels'].extend([patient_data[patient]['label']] * len(patient_data[patient]['paths']))
            
        for patient in val_patients:
            splits['val']['paths'].extend(patient_data[patient]['paths'])
            splits['val']['labels'].extend([patient_data[patient]['label']] * len(patient_data[patient]['paths']))
            
        for patient in test_patients:
            splits['test']['paths'].extend(patient_data[patient]['paths'])
            splits['test']['labels'].extend([patient_data[patient]['label']] * len(patient_data[patient]['paths']))
        
        return splits

if __name__ == "__main__":
    # Example usage
    preprocessor = MedicalImagePreprocessor(
        output_size=(256, 256),
        normalize=True,
        equalize_hist=True
    )
    
    # Process all images (adjust paths as needed)
    preprocessor.process_directory('data/normal', 'processed_data/normal')
    preprocessor.process_directory('data/pneumonia', 'processed_data/pneumonia')
