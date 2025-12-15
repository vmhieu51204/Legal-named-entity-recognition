
# ============================================================
# CONFIGURATION
# ============================================================
import os

# Disable wandb to prevent stalling
os.environ["WANDB_DISABLED"] = "true"

TEST_MODE = False  # Set to False for full training

# Test mode settings
TEST_SAMPLE_SIZE = 300
TEST_MAX_EPOCHS = 5
TEST_BATCH_SIZE = 8

# Full training settings
FULL_MAX_EPOCHS = 10
FULL_BATCH_SIZE = 16

# Models to train
TRAIN_DEBERTA_BASE = True
TRAIN_DEBERTA_LARGE = True
TRAIN_LEGAL_BERT = True
TRAIN_BILSTM_CRF = True

# BiLSTM-CRF + GloVe Configuration
BILSTM_CONFIG = {
    'test': {
        'epochs': 5,
        'batch_size': 16,
        'lr': 0.002,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'grad_clip': 5.0
    },
    'full': {
        'epochs': 15,
        'batch_size': 64,
        'lr': 0.001,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'grad_clip': 5.0,
        'lr_patience': 2,
        'lr_factor': 0.5
    }
}

DATA_PATH = "data.csv"
OUTPUT_DIR = "./results"
GLOVE_PATH = "glove.6B.300d.txt"
GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
