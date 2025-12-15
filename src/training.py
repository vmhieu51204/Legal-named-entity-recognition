
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .config import TEST_MODE, FULL_MAX_EPOCHS, TEST_MAX_EPOCHS
from .preprocessing import tokenize_and_align_labels
from .evaluation import compute_metrics, evaluate_bilstm

def get_training_args(model_name, batch_size=16, gradient_accumulation_steps=1, output_dir="./results"):
    num_epochs = TEST_MAX_EPOCHS if TEST_MODE else FULL_MAX_EPOCHS
    lr = 5e-5 if TEST_MODE else 2e-5
    
    return TrainingArguments(
        output_dir=f"{output_dir}/{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_dir=f"{output_dir}/{model_name}/logs",
        logging_steps=10 if TEST_MODE else 50,
        report_to="none",
        push_to_hub=False,
    )

def train_transformer_model(model_checkpoint, dataset, label2id, id2label, 
                          batch_size=16, gradient_accumulation_steps=1, max_length=256, output_dir="./results"):
    print(f"\n{'='*60}")
    print(f"Training: {model_checkpoint}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    tokenize_fn = lambda examples: tokenize_and_align_labels(examples, tokenizer, max_length)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model_name = model_checkpoint.split('/')[-1]
    training_args = get_training_args(model_name, batch_size, gradient_accumulation_steps, output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting training...")
    trainer.train()
    
    return trainer, tokenized_dataset

def train_bilstm_model(model, train_loader, val_loader, config, device, id2label):
    print(f"\n{'='*60}")
    print("Training: BiLSTM-CRF")
    print(f"{'='*60}")
    
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    scheduler = None
    if not TEST_MODE:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', 
            patience=config['lr_patience'], 
            factor=config['lr_factor']
        )
    
    best_f1 = -1
    best_model_state = model.state_dict().copy()
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            seq_lens = batch['seq_lens']
            
            optimizer.zero_grad()
            loss = model(input_ids, seq_lens, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_metrics = evaluate_bilstm(model, val_loader, id2label, device)
        val_f1 = val_metrics['eval_f1']
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}")
        
        if scheduler:
            scheduler.step(val_f1)
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  ✅ New best model saved (F1={best_f1:.4f})")
            
    model.load_state_dict(best_model_state)
    return model
