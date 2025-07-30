from config import LLMConfig

class LLMestimator:
    def __init__(self, config: LLMConfig):
        self.d_model    = int(config.d_model)
        self.n_layers   = int(config.n_layers)
        self.n_heads    = int(config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.mlp_ratio  = float(config.mlp_ratio)

        self.vocab_size              = int(config.vocab_size)
        self.max_position_embeddings = int(config.max_position_embeddings)

        self.weight_dtype = config.weight_dtype.lower()
        self.kv_dtype     = config.kv_dtype.lower()

        self.tie_embeddings = bool(config.tie_embeddings)
        self.mlp_type       = config.mlp_type.lower()


        # validation
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not (1 <= self.n_kv_heads <= self.n_heads):
            raise ValueError("n_kv_heads must be in [1, n_heads].")
        for x in (self.d_model, self.n_layers, self.n_heads, self.n_kv_heads,
                  self.vocab_size, self.max_position_embeddings):
            if x <= 0:
                raise ValueError("Model/embedding integer params must be > 0.")
        
        # dereived value
        self.head_dim = self.d_model // self.n_heads
        
        # mlp(FFN) intermediate size
        if config.intermediate_size is not None:
            self.intermediate_size = int(config.intermediate_size)
        else:
            eff_ratio = self.mlp_ratio * (2/3 if self.mlp_type == "swiglu" else 1.0)
            raw = eff_ratio * self.d_model
            self.intermediate_size = int((raw+255) // 256*256)

        # data_size
        self.weight_bytes = self._dtype_bytes(self.weight_dtype)
        self.kv_bytes     = self._dtype_bytes(self.kv_dtype)

    @staticmethod
    def _dtype_bytes(dtype: str) -> float:
        table = {
            "fp32": 4, "float32": 4,
            "fp16": 2, "half": 2,
            "bf16": 2, "bfloat16": 2,
            "int8": 1, "uint8": 1,
            "int4": 0.5
        }
        if dtype not in table:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return table[dtype]

    def summary(self):
        print("=== LLM Estimator Configuration ===")
        print(f"d_model:                {self.d_model}")
        print(f"n_layers:               {self.n_layers}")
        print(f"n_heads:                {self.n_heads}")
        print(f"n_kv_heads:             {self.n_kv_heads}")
        print(f"head_dim:               {self.head_dim}")
        print(f"mlp_ratio:              {self.mlp_ratio}")
        print(f"intermediate_size:      {self.intermediate_size}")
        print(f"vocab_size:             {self.vocab_size}")
        print(f"max_position_embeddings:{self.max_position_embeddings}")
        print(f"weight_dtype:           {self.weight_dtype} ({self.weight_bytes} bytes)")
        print(f"kv_dtype:               {self.kv_dtype} ({self.kv_bytes} bytes)")
        print(f"tie_embeddings:         {self.tie_embeddings}")
        print(f"mlp_type:               {self.mlp_type}")
        print("===================================")

        