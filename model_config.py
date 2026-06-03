import json
from dataclasses import asdict, dataclass, field


@dataclass
class TinyDecoderConfig:
    n_layers: int = 8
    hidden_dim: int = 384
    n_heads: int = 6
    ffn_dim: int = 1536
    vocab_size: int = 16000
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    task_token_ids: list[int] = field(default_factory=lambda: [4, 5, 6])

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load_json(path: str) -> "TinyDecoderConfig":
        with open(path, "r", encoding="utf-8") as f:
            cfg = TinyDecoderConfig(**json.load(f))
        params = estimate_param_count(cfg)
        print(f"TinyDecoder config loaded: {cfg.n_layers}L x {cfg.hidden_dim}H | {params:,} parameters")
        return cfg


def estimate_param_count(config: TinyDecoderConfig) -> int:
    h = config.hidden_dim
    f = config.ffn_dim
    per_layer = 0
    per_layer += 4 * h * h + 4 * h
    per_layer += h * f + f
    per_layer += f * h + h
    per_layer += 4 * h
    total = config.vocab_size * h
    total += config.n_layers * per_layer
    total += 2 * h
    return total


if __name__ == "__main__":
    TinyDecoderConfig().save_json("model_config.json")
    print("=== DONE: model_config.py ===")
