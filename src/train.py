import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.data_loader import load_staqc, preprocess_data
from src.model import RAGModel

def train():
    # Load data
    staqc_data = load_staqc("data/staqc/python_question_code_pairs.pkl")
    staqc_df = preprocess_data(staqc_data, dataset_type="staqc")
    
    # Convert to Dataset/Dataloader
    class QADataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.questions = df["question"].tolist()
            self.codes = df["code"].tolist()
        
        def __len__(self):
            return len(self.questions)
        
        def __getitem__(self, idx):
            return self.questions[idx], self.codes[idx]
    
    train_dataset = QADataset(staqc_df)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    model = RAGModel()
    optimizer = AdamW(model.model.parameters(), lr=5e-5)
    
    # Training loop
    for epoch in range(3):
        for batch in train_loader:
            questions, codes = batch
            inputs = model.tokenizer(questions, return_tensors="pt", padding=True)
            labels = model.tokenizer(codes, return_tensors="pt", padding=True).input_ids
            outputs = model.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
