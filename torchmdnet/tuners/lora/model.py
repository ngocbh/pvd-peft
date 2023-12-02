from typing import Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch import nn

from torchmdnet.tuners.utils import (
    get_submodules,
    check_target_module_exists,
)

from torchmdnet.tuners.lora.config import LoraConfig


class Linear_Lora(nn.Module):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,
        init_ia3_weights: bool = True,  # whether to initialize IA3 weights
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        self.adapter_name = adapter_name

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError('base_layer is not a nn.Linear')

        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        # self.update_layer(adapter_name, init_ia3_weights)

    @property
    def weight(self) -> torch.Tensor:
        base_layer = self.base_layer
        if hasattr(base_layer, "qweight"):
            # QuantLinear
            weight = base_layer.qweight
        else:
            # Other layers
            weight = base_layer.weight
        return weight

    @property
    def bias(self) -> torch.Tensor:
        return self.base_layer.bias

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.base_layer.weight.device)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        dtype = previous_dtype = x.dtype

        ia3_scaling = 1
        ia3_scaling *= self.ia3_l[self.adapter_name].flatten()

        if self.is_feedforward:
            x = x.to(dtype)
            # TODO: weight.dtype can be != self.ia3_l[self.active_adapters].dtype
            # e.g. bf16 vs fp32. Is that okay?
            interm = (x * ia3_scaling).to(self.base_layer.weight.dtype)
            result = self.base_layer(interm, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result


class LoraModel(nn.Module, ABC):
    prefix = 'lora_'

    def __init__(self, model, peft_config, adapter_name):
        super().__init__()
        self.model = model
        self.peft_config = peft_config
        self.inject_adapter(self.model, adapter_name)
        self.model.peft_config = peft_config

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def _create_module(self, target, adapter_name):
        new_module = Linear_Lora(target,
                                adapter_name)
        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)

    def _create_and_replace(
        self,
        peft_config: LoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ) -> None:
        current_key = optional_kwargs["current_key"]
        if isinstance(target, torch.nn.Linear):
            new_module = self._create_module(target,
                                             adapter_name,
                                             peft_config.is_feedforward,
                                             peft_config.init_ia3_weights)
            self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def inject_adapter(self, model, adapter_name):
        peft_config = self.peft_config
        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            if not check_target_module_exists(peft_config, key):
                continue

            parent, target, target_name = get_submodules(model, key)
            optional_kwargs = {
                "current_key": key,
            }
            self._create_and_replace(peft_config, adapter_name, target, target_name, parent, **optional_kwargs)

        self._mark_only_adapters_as_trainable(model)

        if self.peft_config.inference_mode:
            for n, p in model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = False
