
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(eval_pred, id2label):
    """Compute entity-level metrics using seqeval."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_label = []
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(id2label[p])
                true_label.append(id2label[l])
        true_predictions.append(true_pred)
        true_labels.append(true_label)
    
    return {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions)
    }

def evaluate_bilstm(model, dataloader, id2label, device):
    """Evaluate BiLSTM model."""
    import torch
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            seq_lens = batch['seq_lens']
            
            predictions = model(input_ids, seq_lens)
            
            for pred, label, length in zip(predictions, labels, seq_lens):
                pred_tags = [id2label[p] for p in pred[:length]]
                true_tags = [id2label[l.item()] for l in label[:length]]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)
    
    return {
        'eval_precision': precision_score(all_labels, all_preds),
        'eval_recall': recall_score(all_labels, all_preds),
        'eval_f1': f1_score(all_labels, all_preds)
    }
