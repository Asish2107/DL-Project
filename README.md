# Health Analysis System with Deep Learning

[![GitHub Stars](https://img.shields.io/github/stars/Asish2107/DL-Project?style=social)](https://github.com/Asish2107/DL-Project/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive health analysis system integrating skin condition prediction, health parameter evaluation, and medical document Q&A using Retrieval-Augmented Generation (RAG).

## Features

### 🩺 Skin Condition Analysis
- MobileNetV2-based image classification
- 85% accuracy on common skin conditions
- Real-time confidence scoring

### 📊 Health Report Generation
- 10+ parameter analysis (BMI, Blood Pressure, Cholesterol, etc.)
- Rule-based health risk assessment
- Personalized recommendations

### 📚 Medical Document Q&A
- RAG system with Mistral-7B LLM
- FAISS vector database for document retrieval
- Source citation for answers

## Installation

1. Clone repository:
```bash
git clone https://github.com/Asish2107/DL-Project.git
cd DL-Project
```

2. Install dependencies:

```bash
Copy
pip install -r requirements.txt
```

3. Set up Ollama (required for Q&A system):

```bash
Copy
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

## Usage

 Start the Flask server:

```bash
Copy
python app.py
Access web interface at http://localhost:5001

Use features through web UI:

Skin Analysis: Upload dermatology images

Health Report: Input health parameters

Document Q&A: Ask questions about medical texts
```

## Project Structure
```bash
Copy
DL-Project/
├── data/                  # Source documents for RAG system
├── notebook/              # EDA performed on skin condition prediction data set
├── src/
│   ├── components/        # Health report rules engine & Skin condition prediction training model.
│   └── exception.py       # Exception Handling
│   └── logger.py          # Log Handling
├── templates/             # Flask HTML templates
├── app.py                 # Main application entry point
├── predict_pipeline.py    # Skin condition prediction pipeline
├── rag_system.py          # RAG system with Mistral-7B LLM
└── requirements.txt       # Dependency list
```

## Technologies Used
```bash
1. Core ML:

TensorFlow 2.12.0

MobileNetV2 (ImageNet weights)

NumPy/Pandas (Data processing)

2. NLP & Embeddings:

Ollama (Mistral-7B)

LangChain

Sentence-Transformers (all-MiniLM-L6-v2)

3. Web Framework:

Flask 2.3.2

Werkzeug (Secure file handling)

4. Utilities:

FAISS (Vector storage)

BeautifulSoup (HTML processing)
```

## License
Distributed under MIT License. See LICENSE for details.

## Acknowledgements

Ollama for local LLM infrastructure

LangChain for RAG implementation

Sentence Transformers for embeddings