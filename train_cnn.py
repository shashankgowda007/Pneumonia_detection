import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from cnn_baseline import PneumoniaCNN, PneumoniaDataset
from torch.utils.data import DataLoader

def train_model():
    # Initialize model
    model = PneumoniaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Consistent 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PneumoniaDataset('train_split', transform)
    val_dataset = PneumoniaDataset('val_split', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Training loop (reduced to 5 epochs as requested)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save model
    torch.save(model.state_dict(), 'pneumonia_cnn.pth')
    print("CNN model saved to pneumonia_cnn.pth")

if __name__ == "__main__":
    train_model()
