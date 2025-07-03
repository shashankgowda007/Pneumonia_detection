# DocAssist - Document Processing Assistant

DocAssist is a Python-based document processing system that provides data loading, model training, evaluation, and a web interface for interacting with processed documents.

## Features

- **Data Processing**: Load and preprocess documents from various sources
- **Model Training**: Train custom document processing models
- **Evaluation**: Evaluate model performance with standard metrics
- **Web Interface**: Interactive UI for document processing
- **API Server**: REST API for programmatic access

## Project Structure

```
DocAssist/
├── configs/          # Configuration files
├── data/             # Raw and processed data
├── models/           # Saved model files
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Source code
│   ├── __init__.py
│   ├── api_server.py # REST API server
│   ├── data_loader.py # Data loading and preprocessing
│   ├── evaluate.py   # Model evaluation
│   ├── model.py      # Model definitions
│   ├── train.py      # Training scripts
│   ├── utils.py      # Utility functions
│   └── web_interface.py # Web UI
├── tests/            # Unit and integration tests
├── main.py           # Main entry point
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DocAssist.git
cd DocAssist
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface
```bash
python main.py --mode web
```

### Running the API Server
```bash
python main.py --mode api
```

### Training a Model
```bash
python src/train.py --config configs/training_config.yaml
```

### Evaluating a Model
```bash
python src/evaluate.py --model_path models/best_model.pt
```

## Configuration

Edit files in `configs/` to customize:
- Data loading parameters
- Model architecture
- Training hyperparameters
- API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

[MIT License](LICENSE)
