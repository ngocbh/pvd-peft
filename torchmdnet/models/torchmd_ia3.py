from torchmdnet.tuners.ia3.model import IA3Model


class TorchMD_Net_IA3(IA3Model):
    def __init__(self, model, peft_config, adapter_name='ia3'):
        super().__init__(model, peft_config, adapter_name)
        self.representation_model = self.model.representation_model
        self.output_model = self.model.output_model
        self.prior_model = self.model.prior_model
        self.reduce_op = self.model.reduce_op
        self.derivative = self.model.derivative
        self.output_model_noise = self.model.output_model_noise
        self.position_noise_scale = self.model.position_noise_scale
        self.register_buffer("mean", self.model.mean)
        self.register_buffer("std", self.model.std)
        if hasattr(self.model, 'pos_normalizer'):
            self.pos_normalizer = self.model.pos_normalizer

    def reset_parameters(self):
        self.model.reset_parameters()
