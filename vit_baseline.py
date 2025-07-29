import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm  # For Vision Transformer models
import os
import matplotlib.pyplot as plt
from cnn_baseline import PneumoniaDataset  # Reuse dataset class

# Constants (same as CNN)
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 2  # normal and pneumonia

class PneumoniaViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ViT model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Custom classifier head
        self.head = nn.Sequential(
            nn.Linear(768, 512),  # ViT base has 768-dim embeddings
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        
        # Freeze ViT layers initially
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # For GradCAM
        self.activations = None
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # Get features from ViT
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        
        for blk in self.vit.blocks:
            x = blk(x)
        
        x = self.vit.norm(x)
        
        # Register hook for last block activations
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        self.activations = x
        
        features = x[:, 0]  # CLS token
        return self.head(features)
        
    def get_activations(self, x, target_layer='last_block'):
        """Get activations from target layer for GradCAM"""
        if target_layer == 'last_block':
            return self.activations
        return None

def create_data_loaders():
    """Create data loaders with enhanced transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT expects 224x224 input
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(0, shear=20, scale=(0.9, 1.1)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PneumoniaDataset('train_split', train_transform)
    val_dataset = PneumoniaDataset('val_split', test_transform)
    test_dataset = PneumoniaDataset('test_split', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader

def train_model():
    """Train and evaluate the ViT model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PneumoniaViT().to(device)
    
    # Calculate class weights for imbalanced data
    train_loader, _, _ = create_data_loaders()
    class_counts = torch.bincount(torch.tensor([label for _, label in train_loader.dataset.samples]))
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = optim.Adam([
        {'params': model.vit.parameters(), 'lr': 1e-5},  # Lower LR for pretrained
        {'params': model.head.parameters(), 'lr': 1e-3}
    ])
    
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
        
        # After 3 epochs, unfreeze some ViT layers
        if epoch == 3:
            for param in model.vit.blocks[-4:].parameters():  # Unfreeze last 4 blocks
                param.requires_grad = True
    
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
    
    plt.savefig('vit_training_history.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    # Verify split directory structure exists (same as CNN)
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
    torch.save(model.state_dict(), 'pneumonia_vit.pth')
