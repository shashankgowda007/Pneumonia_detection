import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import optuna
from functools import partial
import json
from cnn_baseline import PneumoniaDataset

# Constants
IMG_SIZE = (224, 224)
EPOCHS = 10
NUM_CLASSES = 2

def create_model(trial):
    # Model architecture parameters
    model_name = trial.suggest_categorical('model_name', 
        ['vit_base_patch16_224', 'vit_small_patch16_224'])
    unfreeze_layers = trial.suggest_int('unfreeze_layers', 0, 12)
    
    # Create model
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    # Head architecture
    hidden_size = trial.suggest_int('hidden_size', 256, 1024)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    head = nn.Sequential(
        nn.Linear(model.num_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, NUM_CLASSES)
    )
    
    # Freeze/unfreeze layers
    for param in model.parameters():
        param.requires_grad = False
    for block in model.blocks[-unfreeze_layers:]:
        for param in block.parameters():
            param.requires_grad = True
            
    return model, head

def objective(trial, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, head = create_model(trial)
    model = model.to(device)
    head = head.to(device)
    
    # Optimizer parameters
    lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 1e-4, log=True)
    lr_head = trial.suggest_float('lr_head', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': lr_backbone},
        {'params': head.parameters(), 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            features = model(inputs)
            outputs = head(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = model(inputs)
                outputs = head(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        
        # Report intermediate results
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_acc

def create_data_loaders():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = PneumoniaDataset('train_split', transform)
    val_dataset = PneumoniaDataset('val_split', transform)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def main():
    train_loader, val_loader = create_data_loaders()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        partial(objective, train_loader=train_loader, val_loader=val_loader),
        n_trials=50,
        timeout=3600
    )
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Accuracy: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    with open('best_vit_params.json', 'w') as f:
        json.dump(trial.params, f, indent=4)

if __name__ == "__main__":
    main()
