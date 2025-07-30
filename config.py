from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig

@dataclass
class LLMConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    mlp_ratio: float
    vocab_size: int
    max_position_embeddings: int
    weight_dtype: str = "bf16"
    kv_dtype: str = "fp16"
    tie_embeddings: bool = True
    mlp_type: str = "swiglu"
    intermediate_size: Optional[int] = None



def from_hf_config(hf_config:PretrainedConfig) -> LLMConfig:
    d_model = hf_config.hidden_size
    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", n_heads)
    intermediate_size = getattr(hf_config, "intermediate_size", None)

    if intermediate_size is not None:
        mlp_ratio = intermediate_size / d_model
    else:
        mlp_ratio = 4.0

    return LLMConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        mlp_ratio=mlp_ratio,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        weight_dtype=str(hf_config.torch_dtype).replace("torch.", ""),
        kv_dtype="fp16",
        tie_embeddings=getattr(hf_config, "tie_word_embeddings", True),
        mlp_type=getattr(hf_config, "hidden_act", "swiglu"),
        intermediate_size=intermediate_size
    )