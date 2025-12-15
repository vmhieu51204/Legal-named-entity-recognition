
import torch
import gc
from torch.utils.data import DataLoader
from src.config import (
    TEST_MODE, TEST_BATCH_SIZE, FULL_BATCH_SIZE, DATA_PATH,
    TRAIN_DEBERTA_BASE, TRAIN_DEBERTA_LARGE, 
    TRAIN_LEGAL_BERT, TRAIN_BILSTM_CRF, BILSTM_CONFIG
)
from src.preprocessing import (
    load_data, get_label_mappings, prepare_splits, 
    create_hf_dataset_dict, load_glove_embeddings, 
    build_vocab, create_embedding_matrix, NERDataset, collate_fn
)
from src.models import BiLSTM_CRF
from src.training import train_transformer_model, train_bilstm_model
from src.evaluation import evaluate_bilstm

def main():
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    df = load_data(DATA_PATH, test_mode=TEST_MODE)
    
    # Prepare Mappings
    label2id, id2label, label_list = get_label_mappings(df)
    
    # Split
    train_df, val_df, test_df = prepare_splits(df, label2id)
    
    # Create HF Dataset
    hf_dataset = create_hf_dataset_dict(train_df, val_df, test_df)
    
    results = {}
    batch_size = TEST_BATCH_SIZE if TEST_MODE else FULL_BATCH_SIZE
    
    # 1. Train DeBERTa-v3-base
    if TRAIN_DEBERTA_BASE:
        print("Training DeBERTa-v3-base...")
        trainer, tokenized = train_transformer_model(
            "microsoft/deberta-v3-base", hf_dataset, label2id, id2label,
            batch_size=batch_size
        )
        res = trainer.evaluate(tokenized['test'])
        results['DeBERTa-v3-base'] = res
        print(f"DeBERTa-v3-base F1: {res['eval_f1']:.4f}")
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Train DeBERTa-v3-large
    if TRAIN_DEBERTA_LARGE:
        print("Training DeBERTa-v3-large...")
        large_batch = max(4, batch_size // 2)
        trainer, tokenized = train_transformer_model(
            "microsoft/deberta-v3-large", hf_dataset, label2id, id2label,
            batch_size=large_batch, gradient_accumulation_steps=2
        )
        res = trainer.evaluate(tokenized['test'])
        results['DeBERTa-v3-large'] = res
        print(f"DeBERTa-v3-large F1: {res['eval_f1']:.4f}")
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Train Legal-BERT
    if TRAIN_LEGAL_BERT:
        print("Training Legal-BERT...")
        trainer, tokenized = train_transformer_model(
            "nlpaueb/legal-bert-base-uncased", hf_dataset, label2id, id2label,
            batch_size=batch_size
        )
        res = trainer.evaluate(tokenized['test'])
        results['Legal-BERT'] = res
        print(f"Legal-BERT F1: {res['eval_f1']:.4f}")
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Train BiLSTM-CRF
    if TRAIN_BILSTM_CRF:
        print("Training BiLSTM-CRF...")
        config = BILSTM_CONFIG['test'] if TEST_MODE else BILSTM_CONFIG['full']
        
        glove_embeddings = load_glove_embeddings()
        word2idx = build_vocab(hf_dataset)
        embedding_matrix = create_embedding_matrix(word2idx, glove_embeddings)
        
        train_ds = NERDataset(train_df['tokens'].tolist(), train_df['label_ids'].tolist(), word2idx, label2id)
        val_ds = NERDataset(val_df['tokens'].tolist(), val_df['label_ids'].tolist(), word2idx, label2id)
        test_ds = NERDataset(test_df['tokens'].tolist(), test_df['label_ids'].tolist(), word2idx, label2id)
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
        
        model = BiLSTM_CRF(
            vocab_size=len(word2idx),
            embedding_dim=300,
            hidden_dim=config['hidden_dim'],
            num_labels=len(label2id),
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            embedding_matrix=embedding_matrix
        ).to(device)
        
        model = train_bilstm_model(model, train_loader, val_loader, config, device, id2label)
        res = evaluate_bilstm(model, test_loader, id2label, device)
        results['BiLSTM-CRF'] = res
        print(f"BiLSTM-CRF F1: {res['eval_f1']:.4f}")

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"{name}: F1={metrics.get('eval_f1', 0):.4f}")

if __name__ == "__main__":
    main()
