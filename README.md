# Health Analysis System with Deep Learning

[![GitHub Stars](https://img.shields.io/github/stars/Asish2107/DL-Project?style=social)](https://github.com/Asish2107/DL-Project/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive health analysis system integrating skin condition prediction based on the user provided images, health parameter evaluation, and medical document Q&A using Retrieval-Augmented Generation (RAG).

## Features

### 🩺 Skin Condition Analysis
- MobileNetV2-based image classification
- 85% accuracy on skin conditions of type Eczema, Atopic Dermatitis, Melanocytic Nevi, Psoriasis,Seborrheic Keratoses, Tinea Ringworm Candidiasis and Warts Molluscum.
- Real-time confidence scoring

### 📊 Health Report Generation
- 10+ parameter analysis (BMI, Blood Pressure, Cholesterol, etc.)
- Rule-based health risk assessment
- Personalized recommendations

### 📚 Medical Document Q&A
- RAG system with Mistral-7B LLM
- FAISS vector database for document retrieval
- Source citation(Input PDF Documents) for answers

## Installation

1. Clone repository:
```bash
git clone https://github.com/Asish2107/DL-Project.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up Ollama (required for Q&A system):

```bash
ollama serve # Run this in a separate terminal for the continous running of ollama model.
ollama pull mistral # Run this command in new terminal.
```

## Usage

 Start the Flask server:

```bash
python app.py
Access web interface at http://localhost:5001

Use features through web UI:

Skin Analysis: Upload dermatology images(Note - For now this model is trained only on skin diseases of type Eczema, Atopic Dermatitis, Melanocytic Nevi, Psoriasis,Seborrheic Keratoses, Tinea Ringworm Candidiasis and Warts Molluscum)

Health Report: Input health parameters

Document Q&A: Ask any medical related questions.
```

## Project Structure
```bash
DL-Project/
├── data/                  # Source documents for RAG system
├── notebook/              # EDA performed on different skin conditions data set
├── src/
│   ├── components/        # Health report rules engine & Skin condition model training.
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

NumPy/Pandas/ScikitLearn (Data processing)

2. NLP & Embeddings:

Ollama (Mistral-7B)

LangChain

Sentence-Transformers (all-MiniLM-L6-v2)

3. Web Framework:

Flask 2.3.2

Werkzeug (Secure file handling)

4. Utilities:

FAISS (Vector storage)

HTML
```

## License
Distributed under MIT License. See LICENSE for details.

## Acknowledgements

Ollama for local LLM infrastructure

LangChain for RAG implementation

Sentence Transformers for embeddings

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    %% ========== System Overview ==========
    A[User] --> B[Web Interface]
    B --> C{Select Service}
    
    %% ========== Skin Analysis Path ==========
    C --> |Skin Analysis| D[Upload Image]
    D --> E["Validate File (PNG/JPG/JPEG)"]
    E --> F["Preprocess Image\n(Resize 224x224, Normalize)"]
    F --> G["MobileNetV2 Prediction\n(85% Accuracy)"]
    G --> H["Display Diagnosis\n(Eczema, Psoriasis, Warts, etc.)"]
    
    %% ========== Health Report Path ==========
    C --> |Health Report| I["Input Parameters\n(BMI, BP, Cholesterol)"]
    I --> J[Validate Data Ranges]
    J --> K["Rule Engine Analysis\n(10+ Health Metrics)"]
    K --> L[Generate Summary]
    
    %% ========== Document Q&A Path ==========
    C --> |Document Q&A| M[Upload PDF/Ask Question]
    M --> N["Text Chunking\n(LangChain Splitter)"]
    N --> O["Embed Documents\n(Sentence Transformers)"]
    O --> P[FAISS Vector Store]
    P --> Q["Query Mistral-7B\n(via Ollama)"]
    Q --> R["Display Answer\n(With Source PDFs)"]
    
    %% ========== Data Flow ==========
    H --> S[Results Page]
    L --> S
    R --> S
    
    %% ========== Styling ==========
    classDef ui fill:#e3f2fd,stroke:#2196f3;
    classDef ml fill:#c8e6c9,stroke:#4caf50;
    classDef data fill:#fff3e0,stroke:#ff9800;
    
    class A,B,C,S ui;
    class G,K,Q ml;
    class E,F,J,N,O data;
```