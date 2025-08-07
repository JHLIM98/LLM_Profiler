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

    #---utils---
    def _bytes(self, params: int) -> int:
        return int(params) * self.weight_bytes

    def _fmt(self, n_bytes: float) -> str:
        gb = n_bytes / (1024**3)
        if gb >= 1:
            return f"{gb:.2f} GB"
        mb = n_bytes / (1024**2)
        return f"{mb:.2f} MB"

    #---estimation methods---
    def estimate_embeddings_bytes(self, include_pos_embed:bool = False) -> float:
        embed_params = self.vocab_size * self.d_model
        pos_params = self.max_position_embeddings * self.d_model if include_pos_embed else 0
        return self._bytes(embed_params + pos_params)
    
    def estimate_lm_head_bytes(self) -> float:
        if self.tie_embeddings:
            lm_head_params = 0
        else:
            lm_head_params = self.vocab_size * self.d_model
        return self._bytes(lm_head_params)

    def estimate_attention_one_layer_bytes(self, attention_bias: bool = False) -> float:
        d, h, h_kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        
        q_out = d
        k_out = h_kv * hd
        v_out = h_kv * hd
        o_out = d

        q_params = d * q_out
        k_params = d * k_out
        v_params = d * v_out
        o_params = d * o_out

        bias_params = q_out + k_out + v_out + o_out if attention_bias else 0

        per_layer_params = q_params + k_params + v_params + o_params + bias_params

        return self._bytes(per_layer_params)
    
    def estimate_mlp_one_layer_bytes(self, mlp_bias: bool = False) -> float:
        d, i = self.d_model, self.intermediate_size

        gate_params = d * i        
        up_params   = d * i
        down_params = i * d

        bias_total = 0
        if mlp_bias:
            bias_total = i + i + d
        
        per_layer_params = gate_params + up_params + down_params + bias_total

        return self._bytes(per_layer_params)
    

    
    def estimate_norm_once_bytes(self) -> float:
        params = self.d_model
        return self._bytes(params)
    
    
    def estimate_weights_bytes(self, 
                                include_pos_embed: bool = False,
                                attention_bias: bool = False,
                                mlp_bias: bool = False
    ) -> float:
        
        emb_b = self.estimate_embeddings_bytes(include_pos_embed)
        lm_b = self.estimate_lm_head_bytes()

        attn_b = self.n_layers * self.estimate_attention_one_layer_bytes(attention_bias)
        mlp_b  = self.n_layers * self.estimate_mlp_one_layer_bytes(mlp_bias)
        norm_b = (2 * self.n_layers + 1) * self.estimate_norm_once_bytes()

        return emb_b + lm_b + attn_b + mlp_b + norm_b
    
    def estimate_kv_cache_bytes(self,
                                # todo
    )
        return 0

    def memory_report():
        #todo
        return