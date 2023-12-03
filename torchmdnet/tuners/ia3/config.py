class IA3Config:
    def __init__(self, args):
        # Adapter Config
        self.init_ia3_weights = True
        self.is_feedforward = False
        self.inference_mode = False
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

        # self.trainable_param_names = ".*layer_norm.*|.*adapter.*"
