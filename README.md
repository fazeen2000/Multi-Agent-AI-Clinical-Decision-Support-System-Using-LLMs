# Multi-Agent-AI-Clinical-Decision-Support-System-Using-LLMs

## Project Overview
This project implements a **Multi-Agent AI Clinical Decision Support System** using Large Language Models (LLMs) and the PubMedQA (PQAa) dataset. The system is designed to assist clinicians and researchers by providing:

- Automated clinical diagnosis suggestions
- Personalized treatment plans
- Interactive evaluation through a Streamlit interface
- Reproducible experimentation using multiple embeddings, vector databases, and LLMs

The system uses **embeddings models** (SapBERT, SciBERT, BioBERT, All-Mini, All-MPNet, Multi-Mini-qa), **vector databases** (FAISS, Qdrant), and **API-based LLMs** (GPT-4o, Claude Sonent 4) to provide context-aware clinical recommendations.

This repository implements a multi-agent AI clinical decision support system using LLMs. The system allows automated diagnosis and treatment plan generation based on patient cases and histories. It supports multiple embeddings, vector databases, and Top-K retrieval configurations, ensuring full reproducibility for research and peer review.

## Dataset Information

**Dataset Name:** PubMedQA [(PQAa)](https://drive.google.com/open?id=15v1x6aQDlZymaHGP7cZJZZYFfeJt2NdS)  

**Source:** Publicly available biomedical question-answering dataset from PubMed abstracts. 

@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}  

**Format:** JSON, with the following structure for each sample: 
{
  "question": "Your biomedical question here",
  "context": {
    "contexts": ["Abstract sentence 1", "Abstract sentence 2", "..."],
    "meshes": ["MeSH terms if available"],
    "labels": ["Optional labels"]
  },
  "long_answer": "Full text or detailed answer",
  "final_decision": "yes/no/maybe"
}  

**Size:** Over 211k+ QA pairs (for training and evaluation).  

This dataset is widely used for training AI models to understand and answer clinical and biomedical questions. It provides a real-world evaluation for clinical decision support systems and supports reproducibility in biomedical NLP research.

## Code Files & Structure

### 1. `ClaudeSAPBertFinetuned_testcontext93.py`
Main Kaggle notebook performing:  
- Library installation and imports  
- PubMedQA preprocessing  
- Embeddings generation using pretrained embedding models as discussed above.  
- FAISS and Qdrant index search and retrieval  
- Streamlit multi-agent workflow (history → diagnosis → treatment)  
- Evaluation using cosine similarity against PubMedQA test cases
- Note: all experimental combinations are performed with this code.

### 2. `finetuned-faiss-sapbert.ipynb`
- Fine-tuning pretrained embedding models on PubMedQA dataset  
- Indexing embeddings into FAISS vector database  

### 3. `finetuned-qdrant-sapbert.ipynb`
- Fine-tuning pretrained embedding models on PubMedQA dataset  
- Indexing embeddings into Qdrant vector database

### 4. Note: Finetuned files with both vector databases are attached because search function of both databases were different.


## Workflow & Methodology

### Dataset Preprocessing
- Load PubMedQA dataset  
- Extract nested fields: question, context, long answer  
- Combine question + abstract sentences for embedding generation  

### Embedding Generation
- Pretrained models: SapBERT, SciBERT, BioBERT, All-MiniLM, All-MPNet, Multi-qa-MiniLM  
- Generate vector embeddings for each sample  
- Save embeddings for indexing  

### Vector Database Indexing
- FAISS or Qdrant used for fast similarity search  
- Create indexes for all embeddings  
- Support retrieval with Top-K values (10–50)  

### Streamlit Multi-Agent Workflow
- **History Node:** Processes prior patient history  
- **Diagnosis Node:** Uses LLM and retrieved PubMed contexts to generate 3 diagnoses  
- **Treatment Node:** Uses LLM, retrieved contexts, and generated diagnoses to generate step-by-step treatment plan  
- **UI:** User inputs patient case, optional history, clicks "Run Diagnosis" to receive results  

### Evaluation
- Compare generated outputs with PubMedQA test dataset  
- Compute cosine similarity of embeddings between generated outputs and ground-truth PubMed passages  
- Evaluate case, diagnosis, and treatment separately  

### Fine-Tuning
- Fine-tune embedding models using PubMedQA dataset  
- Repeat process with embedding generation with finetuned models → indexing → retrieval → evaluation workflow for improved performance  

---

## Experimental Design

| Elements | Values |
|----------|--------|
| API-based LLMs | GPT-4o, Claude Sonent 4 |
| Embedding Models | 3 General-purpose (All-MiniLM, All-MPNet, Multi-qa-MiniLM) <br> 3 Domain-specific (SapBERT, SciBERT, BioBERT) |
| Vector Databases | FAISS, Qdrant |
| Top-K Retrieval Values | 10, 15, 20, 25, 30, 35, 40, 45, 50 |

**Procedure:**  
For each combination of LLM, embedding model, vector database, and Top-K value:  
1. Preprocess dataset  
2. Generate embeddings  
3. Index embeddings in the vector database  
4. Run Streamlit workflow  
5. Evaluate outputs against PubMedQA test set  

This systematic approach allows a **comparative analysis** to determine the optimal setup for clinical QA accuracy.


## Usage Instructions
### 1. Clone the repository

git clone https://github.com/yourusername/Multi-Agent-AI-Clinical-Decision-Support-System-Using-LLMs.git
cd Multi-Agent-AI-Clinical-Decision-Support-System-Using-LLMs

### 2. Install required libraries

pip install -r requirements.txt

### 3. Run dataset preprocessing, embeddings generation, and indexing

1. Preprocess datasets
2. Generate embeddings
3. Index embeddings using FAISS or Qdrant

### 4. Run the Streamlit application

streamlit run ClaudeSAPBertFinetuned_testcontext93.py

### 5. Input patient case and history (if available)
### 6. Deploy via Ngrok for external access.
### 7. Enter patient case in UI anf Click "Run Diagnosis" to generate a diagnosis and treatment plan.

## Requirements
- Python 3.9+
- Libraries:
streamlit, sentence-transformers, faiss, numpy, scikit-learn, tqdm, langchain_anthropic, langgraph, re
- Pretrained and fientuned embeddings: SapBERT, SciBERT, BioBERT, allMini, AllMPNet, Multi-mini-qa
- Vector database: FAISS or Qdrant
- Dataset: PubMedQA (PQAa)

## Reproducibility
- All code, datasets, embeddings, and index files are versioned and included.
- Experimental setup ensures every combination of LLM, embedding, vector database, and Top-K retrieval is reproducible.
- Fully reproducible for peer review and validation.
