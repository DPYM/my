"""
Custom DeepFM implementation for DualGuard.
Designed for Opacus compatibility and RotEmb integration.

Key design decisions:
  - Single embedding table, uniform k, to support a single rotation R.
  - Explicit attribute for the first DNN Linear layer so RotEmb can compensate it.
  - No library dependency beyond torch.
"""

import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(
        self,
        sparse_vocab_sizes,       # list[int]: vocab size for each sparse field
        n_dense,                  # int: number of dense (continuous) features
        embed_dim=16,
        dnn_hidden_units=(400, 400, 400),
        dropout=0.2,
    ):
        super().__init__()

        self.n_sparse = len(sparse_vocab_sizes)
        self.n_dense = n_dense
        self.embed_dim = embed_dim

        # Offsets for shared embedding table
        self.offsets = torch.tensor([0] + list(self._cumsum(sparse_vocab_sizes))[:-1])

        # ---- Embeddings ----
        total_vocab = sum(sparse_vocab_sizes)
        self.embedding = nn.Embedding(total_vocab, embed_dim, padding_idx=None)

        # ---- Linear (first-order) weights ----
        self.linear_sparse = nn.Embedding(total_vocab, 1, padding_idx=None)
        self.linear_dense = nn.Linear(n_dense, 1, bias=False) if n_dense > 0 else None
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ---- DNN ----
        dnn_in = self.n_sparse * embed_dim + n_dense
        layers = []
        prev = dnn_in
        for h in dnn_hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.dnn = nn.Sequential(*layers)

        # Expose first DNN Linear for RotEmb compensation
        self.first_dnn_linear = self._find_first_linear(self.dnn)

        self._init_weights()

    # ---------------------------------------------------------------
    @staticmethod
    def _cumsum(lst):
        s = 0
        for x in lst:
            s += x
            yield s

    @staticmethod
    def _find_first_linear(seq):
        for m in seq:
            if isinstance(m, nn.Linear):
                return m
        raise ValueError("DNN Sequential must contain at least one nn.Linear")

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name or "linear_sparse" in name:
                nn.init.normal_(p, mean=0, std=0.01)
            elif "weight" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------
    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse : (batch, n_sparse)  – long, feature indices  [0, vocab_size_f)
        X_dense  : (batch, n_dense)   – float, or None
        Returns  : (batch,)            – sigmoid probabilities
        """
        batch = X_sparse.shape[0]

        # Offset indices into shared tables
        off = self.offsets.to(X_sparse.device)
        idx = X_sparse + off  # (batch, n_sparse)

        # Embedding lookup  (batch, n_sparse, k)
        emb = self.embedding(idx)

        # ---- Linear term ----
        logit = self.global_bias
        logit = logit + self.linear_sparse(idx).sum(dim=(1, 2))
        if self.linear_dense is not None and X_dense is not None:
            logit = logit + self.linear_dense(X_dense).squeeze(-1)

        # ---- FM term  (efficient paired interaction) ----
        sum_emb = emb.sum(dim=1)          # (batch, k)
        sum_sq = (emb ** 2).sum(dim=1)    # (batch, k)
        fm = 0.5 * ((sum_emb ** 2).sum(dim=1) - sum_sq.sum(dim=1))
        logit = logit + fm

        # ---- DNN term ----
        dnn_in = emb.reshape(batch, self.n_sparse * self.embed_dim)
        if self.n_dense > 0:
            if X_dense is None:
                X_dense = torch.zeros(batch, self.n_dense, device=emb.device, dtype=emb.dtype)
            dnn_in = torch.cat([dnn_in, X_dense], dim=1)
        dnn_out = self.dnn(dnn_in).squeeze(-1)
        logit = logit + dnn_out

        return torch.sigmoid(logit)
