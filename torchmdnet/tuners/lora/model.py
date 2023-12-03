from typing import Any, Union
from abc import ABC, abstractmethod
import torch
import math
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
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
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
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

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

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(self.base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        result = self.base_layer(x, *args, **kwargs)
        lora_A = self.lora_A[self.adapter_name]
        lora_B = self.lora_B[self.adapter_name]
        dropout = self.lora_dropout[self.adapter_name]
        scaling = self.scaling[self.adapter_name]
        x = x.to(lora_A.weight.dtype)
        result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


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

    def _create_module(
        self,
        target,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
    ):
        new_module = Linear_Lora(target,
                                 adapter_name,
                                 r=r,
                                 lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout,
                                 init_lora_weights=init_lora_weights)
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
                                             r=peft_config.r,
                                             lora_alpha=peft_config.lora_alpha,
                                             lora_dropout=peft_config.lora_dropout,
                                             init_lora_weights=peft_config.init_lora_weights)
            self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model) -> None:
        for n, p in self.model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        bias = self.peft_config.bias
        if bias == "all":
            for n, p in self.model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in self.model.modules():
                if isinstance(m, Linear_Lora) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        elif bias == "none":
            return
        else:
            raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

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
