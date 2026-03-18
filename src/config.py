from dataclasses import dataclass, field


@dataclass
class DataConfig:
    num_samples: int = 50_000        # Total samples (train + test)
    test_split: float = 0.2          # 80/20 split
    max_seq_len: int = 128
    # tiktoken cl100k_base vocab size
    vocab_size: int = 100_277


@dataclass
class TransformerConfig:
    vocab_size: int = 100_277        # cl100k_base
    max_seq_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024                 # 4 × d_model
    dropout: float = 0.1


@dataclass
class SAEConfig:
    d_input: int = 256               # Must match TransformerConfig.d_model
    expansion_factor: int = 8        # d_hidden = 256 * 8 = 2048
    k: int = 32                      # Top-K sparsity
    layer_idx: int = 3               # Which transformer layer to hook

    @property
    def d_hidden(self) -> int:
        return self.d_input * self.expansion_factor


@dataclass
class TransformerTrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_every: int = 100
    eval_every: int = 500


@dataclass
class SAETrainingConfig:
    batch_size: int = 256
    learning_rate: float = 2e-4
    num_epochs: int = 5
    max_activations: int = 500_000   # Cap activation vectors collected
    checkpoint_dir: str = "checkpoints"
    log_every: int = 100
