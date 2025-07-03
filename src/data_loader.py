import pickle
from datasets import load_dataset
import pandas as pd

def load_staqc(file_path):
    """Load StaQC data from pickle files."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data, columns=["question_id", "code_snippet", "question_title"])
    return df

def load_conala(split="train"):
    """Load CoNaLa dataset via Hugging Face."""
    dataset = load_dataset("neulab/conala", split=split)
    return dataset

def preprocess_data(df, dataset_type="staqc"):
    """Normalize code/questions (e.g., variable name anonymization)."""
    if dataset_type == "staqc":
        df["question"] = df["question_title"]
        df["code"] = df["code_snippet"].apply(lambda x: x.strip())
    elif dataset_type == "conala":
        df["question"] = df["rewritten_intent"]
        df["code"] = df["snippet"]
    return df[["question", "code"]]

if __name__ == "__main__":
    print("Testing data_loader functions...")
    
    # Test StaQC loading (requires sample.pkl in data/ directory)
    try:
        staqc_df = load_staqc("data/sample.pkl")
        print("StaQC loading successful!")
        print(staqc_df.head())
    except Exception as e:
        print(f"StaQC loading failed: {str(e)}")
    
    # Test CoNaLa loading
    try:
        conala_data = load_conala()
        print("CoNaLa loading successful!")
        # Convert to pandas DataFrame for display
        conala_df = pd.DataFrame(conala_data)
        print(conala_df.head())
    except Exception as e:
        print(f"CoNaLa loading failed: {str(e)}")
