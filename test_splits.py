import os
import numpy as np
from preprocessing import MedicalImagePreprocessor

def test_patient_splits():
    """Test the patient-aware stratified splitting functionality"""
    preprocessor = MedicalImagePreprocessor()
    
    # Define input directories
    input_dirs = [
        'data/normal',
        'data/pneumonia'
    ]
    
    # Verify directories exist before processing
    for dir_path in input_dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory does not exist: {dir_path}")
            return
        else:
            print(f"✓ Directory exists: {dir_path}")
    
    try:
        # Create splits
        splits = preprocessor.create_patient_splits(input_dirs)
        
        # Print statistics
        print("\n" + "="*50)
        print("SPLIT STATISTICS")
        print("="*50)
        
        total_images = 0
        total_patients = set()
        
        for split_name, split_data in splits.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Total images: {len(split_data['paths'])}")
            
            # Count labels
            normal_count = split_data['labels'].count('normal')
            pneumonia_count = split_data['labels'].count('pneumonia')
            
            print(f"  Normal images: {normal_count}")
            print(f"  Pneumonia images: {pneumonia_count}")
            
            # Get unique patients
            patient_ids = {os.path.basename(p).split('_')[0] for p in split_data['paths']}
            print(f"  Unique patients: {len(patient_ids)}")
            
            # Calculate percentages
            if len(split_data['paths']) > 0:
                normal_pct = (normal_count / len(split_data['paths'])) * 100
                pneumonia_pct = (pneumonia_count / len(split_data['paths'])) * 100
                print(f"  Normal percentage: {normal_pct:.1f}%")
                print(f"  Pneumonia percentage: {pneumonia_pct:.1f}%")
            
            total_images += len(split_data['paths'])
            total_patients.update(patient_ids)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total images across all splits: {total_images}")
        print(f"  Total unique patients: {len(total_patients)}")
        
        # Verify no patient overlap between splits
        train_patients = {os.path.basename(p).split('_')[0] for p in splits['train']['paths']}
        val_patients = {os.path.basename(p).split('_')[0] for p in splits['val']['paths']}
        test_patients = {os.path.basename(p).split('_')[0] for p in splits['test']['paths']}
        
        print(f"\nPATIENT OVERLAP CHECK:")
        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients
        
        print(f"  Train-Val overlap: {len(train_val_overlap)} patients")
        print(f"  Train-Test overlap: {len(train_test_overlap)} patients")
        print(f"  Val-Test overlap: {len(val_test_overlap)} patients")
        
        if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
            print("  ✓ No patient overlap detected - splits are valid!")
        else:
            print("  ⚠ WARNING: Patient overlap detected!")
            if train_val_overlap:
                print(f"    Train-Val overlapping patients: {list(train_val_overlap)[:10]}...")
            if train_test_overlap:
                print(f"    Train-Test overlapping patients: {list(train_test_overlap)[:10]}...")
            if val_test_overlap:
                print(f"    Val-Test overlapping patients: {list(val_test_overlap)[:10]}...")
        
        # Sample some paths to verify labeling
        print(f"\nSAMPLE PATHS VERIFICATION:")
        for split_name, split_data in splits.items():
            if len(split_data['paths']) > 0:
                print(f"\n{split_name.upper()} samples:")
                for i, (path, label) in enumerate(zip(split_data['paths'][:3], split_data['labels'][:3])):
                    print(f"  {i+1}. {os.path.basename(path)} -> {label}")
        
    except Exception as e:
        print(f"ERROR during split creation: {str(e)}")
        import traceback
        traceback.print_exc()

def verify_directory_structure():
    """Verify the directory structure and show some sample files"""
    input_dirs = [
        'data/normal',
        'data/pneumonia'
    ]
    
    print("DIRECTORY STRUCTURE VERIFICATION")
    print("="*40)
    
    for dir_path in input_dirs:
        print(f"\nChecking: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"  ❌ Directory does not exist")
            continue
            
        # Count files
        file_count = 0
        sample_files = []
        
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    file_count += 1
                    if len(sample_files) < 5:
                        sample_files.append(f)
        
        print(f"  ✓ Found {file_count} image files")
        print(f"  Sample files: {sample_files}")
        
        # Check patient ID patterns
        if sample_files:
            patient_ids = [f.split('_')[0] for f in sample_files]
            print(f"  Sample patient IDs: {patient_ids}")

if __name__ == "__main__":
    print("Starting directory verification...")
    verify_directory_structure()
    
    print("\n" + "="*60)
    print("Starting patient splits test...")
    test_patient_splits()
