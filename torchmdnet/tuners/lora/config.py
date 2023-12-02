class LoraConfig:
    def __init__(self, args):
        self.init_lora_weights = True
        self.inference_mode = False
        self.target_modules = [
            "k_proj",
            "v_proj",
            "o_proj",
            "dk_proj",
            "dv_proj",
            "vec_proj"
        ]
