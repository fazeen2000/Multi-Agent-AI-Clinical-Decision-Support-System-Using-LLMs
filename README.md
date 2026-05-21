# Multi-Agent-AI-Clinical-Decision-Support-System-Using-LLMs

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
