
# Legal Named Entity Recognition (NER)

## 1. Introduction
This project focuses on **Legal Named Entity Recognition (NER)**, automating the extraction of key legal entities from unstructured text. Identifying entities such as laws, violations, and involved parties is crucial for legal document analysis, compliance monitoring, and case law understanding. 

This repository contains a complete pipeline for loading data, preprocessing, training state-of-the-art NLP models (Transformers and BiLSTM-CRF), and evaluating their performance.

## 2. Dataset
The project uses a dataset consisting of legal texts with annotated named entities.

*   **Source**: `data.csv`
*   **Size**: 1,327 samples
*   **Total Entities**: ~19,278
*   **Average Sentence Length**: ~68 tokens

### Entity Types
The system identifies the following entity types:
*   **LAW**: References to specific laws, acts, or regulations.
*   **VIOLATION**: Actions or states representing a breach of law or agreement.
*   **VIOLATED BY**: The entity committing the violation.
*   **VIOLATED ON**: The entity or object against which the violation was committed.

### Data Split
The dataset is stratified and split as follows:
*   **Train**: 80% (1061 samples)
*   **Validation**: 10% (133 samples)
*   **Test**: 10% (133 samples)

## 3. Models
We implemented and compared four different architectures, ranging from domain-specific Transformers to classical deep learning approaches:

1.  **DeBERTa-v3-large** (`microsoft/deberta-v3-large`): A large transformer model known for superior performance on NLU tasks.
2.  **DeBERTa-v3-base** (`microsoft/deberta-v3-base`): A base version of DeBERTa, offering a balance between performance and speed.
3.  **Legal-BERT** (`nlpaueb/legal-bert-base-uncased`): A BERT model pre-trained specifically on legal domain corpora (contracts, statutes, cases).
4.  **BiLSTM-CRF**: A recurrent neural network architecture using **GloVe 300d** word embeddings, enhanced with a Conditional Random Field (CRF) layer for sequence modeling.

## 4. Results and Conclusion

The models were evaluated on the test set (10% split). The **F1-score** was used as the primary metric.

| Model | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **DeBERTa-v3-large** | **0.7198** | **0.7489** | **0.7341** |
| DeBERTa-v3-base | 0.6680 | 0.7309 | 0.6981 |
| Legal-BERT | 0.6255 | 0.7265 | 0.6722 |
| BiLSTM-CRF | ~0.67 | ~0.65 | ~0.66 |

### Conclusion
*   **DeBERTa-v3-large** achieved the best overall performance, demonstrating the value of larger capacity and advanced pre-training objectives.
*   **Legal-BERT**, despite being domain-specific, trailed behind the generalized DeBERTa models, suggesting that the underlying architecture improvements in DeBERTa (like disentangled attention) outweigh domain adaptation for this specific dataset size.
*   **BiLSTM-CRF** provided a decent baseline but failed to capture the complex contextual nuances as well as the Transformer models.

## 5. Demo Sentences

Below are examples of how the model analyzes legal text:

> "The **platform** [VIOLATED BY] has been found guilty of **breaching player trust** [VIOLATION]."

> "This constitutes a violation of **Article 5 of the GDPR** [LAW] regarding user data privacy."

> "The **defendant** [VIOLATED BY] failed to comply with the **contractual obligations** [VIOLATION] owed to the **plaintiff** [VIOLATED ON]."

---

## How to Run
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the pipeline**:
    ```bash
    python main.py
    ```
    *Modify `src/config.py` to toggle specific models or change hyperparameters.*
