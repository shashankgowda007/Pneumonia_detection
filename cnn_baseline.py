import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMG_SIZE = (224, 224)  # Changed to match checkpoint dimensions
BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 2  # normal and pneumonia

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'pneumonia']  # Force these class names
        self.samples = []
        
        # Verify directory structure
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found")
            
        # Scan for both classes
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Missing class directory {class_dir}")
                continue
                
            img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(img_files)} {class_name} images in {class_dir}")
            
            for img_name in img_files:
                self.samples.append((os.path.join(class_dir, img_name), class_idx))
        
        print(f"Total samples in {root_dir}: {len(self.samples)}")
        
        # Oversample pneumonia cases if needed
        if len(self.samples) > 0:
            pneumonia_indices = [i for i, (_, label) in enumerate(self.samples) if label == 1]
            normal_indices = [i for i, (_, label) in enumerate(self.samples) if label == 0]
            
            if len(pneumonia_indices) < len(normal_indices):
                oversample_factor = len(normal_indices) // len(pneumonia_indices)
                extra_samples = []
                for i in pneumonia_indices:
                    extra_samples.extend([self.samples[i]] * (oversample_factor - 1))
                self.samples.extend(extra_samples)
                print(f"Oversampled pneumonia cases. New total samples: {len(self.samples)}")
        
        if not self.samples:
            raise ValueError(f"No valid images found in {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            # Debug: Print first image info
            if idx == 0:
                print(f"First sample - Path: {img_path}")
                print(f"Label: {label}")
                print(f"Image size: {img.shape if isinstance(img, torch.Tensor) else img.size}")
                print(f"Image type: {type(img)}")
                print(f"Image min/max: {img.min()}/{img.max() if isinstance(img, torch.Tensor) else max(img.getdata())}")
                
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a dummy image if loading fails
            dummy_img = torch.zeros(3, 256, 256) if self.transform else Image.new('RGB', (256, 256))
            return dummy_img, label

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        
        # Debug print to verify feature map size
        self.debug = True
    
    def forward(self, x):
        x = self.features(x)
        if self.debug:
            print(f"Feature map shape: {x.shape}")
            self.debug = False
        x = self.classifier(x)
        return x

def create_data_loaders():
    """Create data loaders with augmentation for training set"""
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(0, shear=20, scale=(0.9, 1.1)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PneumoniaDataset('train_split', train_transform)
    val_dataset = PneumoniaDataset('val_split', test_transform)
    test_dataset = PneumoniaDataset('test_split', test_transform)
    
    # Debug: Check class distribution in first batch
    def check_batch_distribution(loader):
        batch = next(iter(loader))
        print(f"Batch class counts: {torch.bincount(batch[1])}")
        print(f"Unique labels: {torch.unique(batch[1])}")
    
    print("Training set:")
    check_batch_distribution(DataLoader(train_dataset, batch_size=BATCH_SIZE*4, shuffle=True))
    print("Validation set:")
    check_batch_distribution(DataLoader(val_dataset, batch_size=BATCH_SIZE*4))
    print("Test set:")
    check_batch_distribution(DataLoader(test_dataset, batch_size=BATCH_SIZE*4))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def train_model():
    """Train and evaluate the CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PneumoniaCNN().to(device)
    
    # Calculate class weights for imbalanced data
    train_loader, _, _ = create_data_loaders()
    class_counts = torch.bincount(torch.tensor([label for _, label in train_loader.dataset.samples]))
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = optim.Adam(model.parameters())
    
    train_loader, val_loader, test_loader = create_data_loaders()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}: '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if epoch > 5 and val_loss > max(val_losses[-5:-1]):
            print("Early stopping")
            break
    
    # Test evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = correct / total
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    # Verify split directory structure exists
    required_dirs = [
        'train_split/normal',
        'train_split/pneumonia',
        'val_split/normal', 
        'val_split/pneumonia',
        'test_split/normal',
        'test_split/pneumonia'
    ]
    missing = [d for d in required_dirs if not os.path.exists(d)]
    if missing:
        raise FileNotFoundError(
            f"Missing required directories: {missing}\n"
            "Run test_splits.py and prepare_splits.py first."
        )
    
    # Train and evaluate model
    model = train_model()
    
    # Save model
    torch.save(model.state_dict(), 'pneumonia_cnn.pth')
