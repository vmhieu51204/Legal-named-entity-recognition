
import torch
import torch.nn as nn
import numpy as np

try:
    from torchcrf import CRF
except ImportError:
    # Minimal mock CRF for structure if not installed, but user should install it
    # Ideally should raise error or install
    raise ImportError("Please install pytorch-crf: pip install pytorch-crf")

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, 
                 num_layers=2, dropout=0.5, embedding_matrix=None):
        super(BiLSTM_CRF, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, seq_lens, labels=None):
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        actual_max_len = emissions.shape[1]
        
        mask = torch.zeros((input_ids.shape[0], actual_max_len), dtype=torch.bool, device=input_ids.device)
        for i, length in enumerate(seq_lens):
            mask[i, :length] = True
        
        if labels is not None:
            labels = labels[:, :actual_max_len]
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions
