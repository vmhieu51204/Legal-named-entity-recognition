
import pandas as pd
import numpy as np
import ast
import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .config import GLOVE_PATH, GLOVE_URL

def load_data(file_path, test_mode=False, test_sample_size=300):
    """Load and preprocess data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    df = pd.read_csv(file_path)

    # Parse string arrays to Python lists
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['ner_tags'] = df['ner_tags'].apply(ast.literal_eval)

    # Apply test mode sampling
    if test_mode and len(df) > test_sample_size:
        print(f"🧪 TEST MODE: Sampling {test_sample_size} from {len(df)} samples")
        df = df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)

    print(f"Dataset size: {len(df)} samples")
    return df

def get_label_mappings(df):
    """Extract unique labels and create mappings."""
    all_labels = set()
    entity_count = 0
    for tags in df['ner_tags']:
        all_labels.update(tags)
        entity_count += sum(1 for t in tags if t != 'O')

    # Sort labels: O first, then B-tags, then I-tags
    label_list = sorted(list(all_labels), key=lambda x: (x != 'O', x.startswith('I-'), x))
    
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    
    print(f"Entity types found ({len(label_list)} labels):")
    for label in label_list:
        print(f"  - {label}")
        
    return label2id, id2label, label_list

def prepare_splits(df, label2id):
    """Split data into train/val/test with stratification."""
    # Convert NER tags to label IDs
    df['label_ids'] = df['ner_tags'].apply(lambda tags: [label2id[tag] for tag in tags])

    # Create stratification key
    def get_entity_signature(tags):
        entities = set(tag.split('-')[1] if '-' in tag else 'O' for tag in tags)
        return '_'.join(sorted(entities))

    df['entity_signature'] = df['ner_tags'].apply(get_entity_signature)
    
    # Handle rare classes
    signature_counts = df['entity_signature'].value_counts()
    rare_signatures = signature_counts[signature_counts < 2].index.tolist()
    if rare_signatures:
        most_common_signature = signature_counts.index[0]
        df.loc[df['entity_signature'].isin(rare_signatures), 'entity_signature'] = most_common_signature

    # Split: 80% train, 10% validation, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['entity_signature'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['entity_signature'])

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_hf_dataset_dict(train_df, val_df, test_df):
    """Convert DataFrames to HuggingFace DatasetDict."""
    def create_hf_dataset(dataframe):
        return HFDataset.from_dict({
            'id': dataframe['id'].tolist(),
            'tokens': dataframe['tokens'].tolist(),
            'ner_tags': dataframe['label_ids'].tolist()
        })

    return DatasetDict({
        'train': create_hf_dataset(train_df),
        'validation': create_hf_dataset(val_df),
        'test': create_hf_dataset(test_df)
    })

def tokenize_and_align_labels(examples, tokenizer, max_length=256):
    """Tokenize and align labels with -100 masking."""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding='max_length'
    )
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def load_glove_embeddings(glove_path=GLOVE_PATH, embedding_dim=300):
    """Load GloVe embeddings."""
    if not os.path.exists(glove_path):
        print("📥 Downloading GloVe embeddings...")
        zip_path = "glove.6B.zip"
        if not os.path.exists(zip_path):
            try:
                urllib.request.urlretrieve(GLOVE_URL, zip_path)
            except Exception as e:
                print(f"Failed to download GloVe: {e}")
                return {}
        
        print("📦 Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(glove_path)
    
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings_index[word] = vector
    except Exception as e:
        print(f"Error loading GloVe file: {e}")
        return {}
        
    print(f"Loaded {len(embeddings_index)} word vectors")
    return embeddings_index

def build_vocab(datasets, min_freq=1):
    """Build vocabulary from datasets."""
    word_freq = {}
    for split in ['train', 'validation', 'test']:
        for tokens in datasets[split]['tokens']:
            for token in tokens:
                word = token.lower()
                word_freq[word] = word_freq.get(word, 0) + 1
    
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1
    
    print(f"Vocabulary size: {len(word2idx)}")
    return word2idx

def create_embedding_matrix(word2idx, glove_embeddings, embedding_dim=300):
    """Create embedding matrix."""
    vocab_size = len(word2idx)
    embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    found = 0
    for word, idx in word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found += 1
            
    print(f"Found {found}/{vocab_size} words in GloVe ({found/vocab_size*100:.1f}%)")
    return embedding_matrix

class NERDataset(Dataset):
    def __init__(self, tokens_list, labels_list, word2idx, label2id, max_len=256):
        self.tokens_list = tokens_list
        self.labels_list = labels_list
        self.word2idx = word2idx
        self.label2id = label2id
        self.max_len = max_len
    
    def __len__(self):
        return len(self.tokens_list)
    
    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]
        
        token_ids = [self.word2idx.get(t.lower(), self.word2idx['<UNK>']) for t in tokens]
        label_ids = labels
        
        token_ids = token_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]
        seq_len = len(token_ids)
        
        padding_len = self.max_len - seq_len
        token_ids = token_ids + [0] * padding_len
        label_ids = label_ids + [0] * padding_len
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'seq_len': seq_len
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    seq_lens = torch.tensor([item['seq_len'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels, 'seq_lens': seq_lens}
