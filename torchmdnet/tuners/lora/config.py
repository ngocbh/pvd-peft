class LoraConfig:
    def __init__(self, args):
        self.init_lora_weights = True
        self.inference_mode = False
        self.bias = 'lora_only'
        self.r = 4
        self.lora_alpha = 1.0
        self.lora_dropout = 0.9
        self.init_lora_weights = True
        self.target_modules = [
            "k_proj",
            "v_proj",
            "o_proj",
            "dk_proj",
            "dv_proj",
            "vec_proj",
            "vec1_proj",
            "vec2_proj",
            "update_net.0",
            "update_net.2",
        ]
