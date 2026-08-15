from dataclasses import dataclass


LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


@dataclass
class ModelConfig:

    vocab_size: int

    max_len: int = 128

    d_model: int = 256

    n_heads: int = 8

    n_layers: int = 6

    d_ff: int = 1024

    dropout: float = 0.1

    num_classes: int = 4