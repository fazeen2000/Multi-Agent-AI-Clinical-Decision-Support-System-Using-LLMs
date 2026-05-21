# Multi-Agent-AI-Clinical-Decision-Support-System-Using-LLMs

## Project Overview
This project implements a **Multi-Agent AI Clinical Decision Support System** using Large Language Models (LLMs) and the PubMedQA (PQAa) dataset. The system is designed to assist clinicians and researchers by providing:

- Automated clinical diagnosis suggestions
- Personalized treatment plans
- Interactive evaluation through a Streamlit interface
- Reproducible experimentation using multiple embeddings, vector databases, and LLMs

The system uses **embeddings models** (SapBERT, SciBERT, BioBERT, All-Mini, All-MPNet, Multi-Mini-qa), **vector databases** (FAISS, Qdrant), and **API-based LLMs** (GPT-4o, Claude Sonent 4) to provide context-aware clinical recommendations.

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




