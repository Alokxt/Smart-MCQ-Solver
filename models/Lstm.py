import torch 
import torch.nn as nn 


class LSTMScorer(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(lstm_out_dim + hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.scorer = nn.Sequential(
            nn.Linear(lstm_out_dim + hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
 
    def forward(self, q_emb, opt_embs):
        q_repr = self.q_proj(q_emb)
        lstm_out, _ = self.lstm(opt_embs)
        q_repr_expanded = q_repr.unsqueeze(1).expand(-1, 5, -1)
        combined = torch.cat([lstm_out, q_repr_expanded], dim=-1)
        attn_weights = torch.sigmoid(self.attn(combined))
        attended = combined * attn_weights
        scores = self.scorer(attended).squeeze(-1)
        return scores

def get_lstm():
    return LSTMScorer()
