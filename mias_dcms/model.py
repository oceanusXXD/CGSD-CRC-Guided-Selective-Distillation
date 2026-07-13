from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .utils import disable_tokenizer_thinking, read_json, write_json
LORA_MODES = {'lora_attention_mlp'}
ALL_MODES = sorted(LORA_MODES)
LAST_LAYER_LORA_PATTERN = 'layers'
LORA_LAYER_SCOPES = {'last1', 'last4', 'all'}
LORA_TARGET_GROUPS: dict[str, list[str]] = {'qv': ['q_proj', 'v_proj'], 'qkvo': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'mlp': ['gate_proj', 'up_proj', 'down_proj'], 'attention_mlp': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
LORA_TARGETS: dict[str, list[str]] = {'lora_attention_mlp': LORA_TARGET_GROUPS['attention_mlp']}

def set_use_cache_false(module: nn.Module) -> None:
    config = getattr(module, 'config', None)
    if config is not None and hasattr(config, 'use_cache'):
        config.use_cache = False

def get_last_layer_index_from_config(config: Any) -> int:
    if hasattr(config, 'num_hidden_layers'):
        return int(config.num_hidden_layers) - 1
    text_config = getattr(config, 'text_config', None)
    if text_config is not None and hasattr(text_config, 'num_hidden_layers'):
        return int(text_config.num_hidden_layers) - 1
    raise AttributeError('Model config does not expose num_hidden_layers or text_config.num_hidden_layers')

def resolve_lora_layers_to_transform(config: Any, layer_scope: str) -> int | list[int] | None:
    if layer_scope not in LORA_LAYER_SCOPES:
        raise ValueError(f'Unsupported LoRA layer scope: {layer_scope}')
    if layer_scope == 'all':
        return None
    last_layer = get_last_layer_index_from_config(config)
    if layer_scope == 'last1':
        return last_layer
    first_layer = max(0, last_layer - 3)
    return list(range(first_layer, last_layer + 1))

def sparse_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, class_token_weights: dict[int, float] | None=None, sample_weights: torch.Tensor | None=None) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    active_positions = shift_labels.ne(-100)
    if not torch.any(active_positions):
        return logits.sum() * 0.0
    active_logits = shift_logits[active_positions].float()
    active_labels = shift_labels[active_positions].to(active_logits.device)
    per_token_loss = F.cross_entropy(active_logits, active_labels, reduction='none')
    token_weights = torch.ones_like(per_token_loss)
    if class_token_weights:
        for token_id, weight in class_token_weights.items():
            token_weights = torch.where(active_labels.eq(int(token_id)), torch.as_tensor(float(weight), dtype=per_token_loss.dtype, device=per_token_loss.device), token_weights)
    if sample_weights is not None:
        shifted_sample_weights = sample_weights.to(active_logits.device, dtype=per_token_loss.dtype)
        if shifted_sample_weights.ndim != 1 or shifted_sample_weights.size(0) != labels.size(0):
            raise ValueError('sample_weights must be a 1D tensor with one value per batch row')
        row_indices = torch.arange(labels.size(0), device=labels.device).unsqueeze(1).expand_as(shift_labels)
        token_weights = token_weights * shifted_sample_weights[row_indices[active_positions]]
    return (per_token_loss * token_weights).mean()

class QwenGenerativeModel(nn.Module):

    def __init__(self, model_path: str, mode: str, lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05, lora_target_modules: str='attention_mlp', lora_layer_scope: str='last1', torch_dtype: Any='auto', trust_remote_code: bool=True, adapter_path: str | Path | None=None, adapters_trainable: bool=True) -> None:
        super().__init__()
        if mode not in ALL_MODES:
            raise ValueError(f'Unsupported mode: {mode}')
        self.model_path = str(model_path)
        self.mode = mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if lora_target_modules not in LORA_TARGET_GROUPS:
            raise ValueError(f'Unsupported LoRA target module group: {lora_target_modules}')
        if lora_layer_scope not in LORA_LAYER_SCOPES:
            raise ValueError(f'Unsupported LoRA layer scope: {lora_layer_scope}')
        self.lora_target_modules = lora_target_modules
        self.lora_layer_scope = lora_layer_scope
        self.trust_remote_code = trust_remote_code
        loaded_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, local_files_only=True)
        set_use_cache_false(loaded_model)
        self.lora_layers_to_transform = resolve_lora_layers_to_transform(loaded_model.config, self.lora_layer_scope)
        self.backbone = self._build_lora_backbone(base_model=loaded_model, adapter_path=adapter_path, adapters_trainable=adapters_trainable)

    def _build_lora_backbone(self, base_model: nn.Module, adapter_path: str | Path | None, adapters_trainable: bool) -> nn.Module:
        try:
            from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError('LoRA modes require the peft package.') from exc
        if adapter_path is not None:
            return PeftModel.from_pretrained(base_model, adapter_path, is_trainable=adapters_trainable)
        config_kwargs = {'task_type': TaskType.CAUSAL_LM, 'r': self.lora_r, 'lora_alpha': self.lora_alpha, 'target_modules': LORA_TARGET_GROUPS[self.lora_target_modules], 'lora_dropout': self.lora_dropout, 'bias': 'none'}
        if self.lora_layers_to_transform is not None:
            config_kwargs['layers_to_transform'] = self.lora_layers_to_transform
            config_kwargs['layers_pattern'] = LAST_LAYER_LORA_PATTERN
        config = LoraConfig(**config_kwargs)
        return get_peft_model(base_model, config)

    def forward(self, input_ids: Any, attention_mask: Any, labels: Any | None=None, class_token_weights: dict[int, float] | None=None, sample_weights: Any | None=None, **kwargs: Any) -> Any:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=None, return_dict=True, **kwargs)
        if labels is None:
            return outputs
        return CausalLMOutputWithPast(loss=sparse_causal_lm_loss(outputs.logits, labels, class_token_weights=class_token_weights, sample_weights=sample_weights), logits=outputs.logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone.generate(*args, **kwargs)

    def checkpoint_config(self) -> dict[str, Any]:
        return {'model_path': self.model_path, 'mode': self.mode, 'lora_r': self.lora_r, 'lora_alpha': self.lora_alpha, 'lora_dropout': self.lora_dropout, 'lora_target_modules': self.lora_target_modules, 'lora_targets': LORA_TARGET_GROUPS[self.lora_target_modules], 'lora_layer_scope': self.lora_layer_scope, 'lora_layers_to_transform': self.lora_layers_to_transform, 'lora_layers_pattern': LAST_LAYER_LORA_PATTERN if self.lora_layers_to_transform is not None else None, 'trust_remote_code': self.trust_remote_code}

    def save_checkpoint(self, output_dir: str | Path, tokenizer: Any | None=None, extra_config: dict[str, Any] | None=None) -> None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        config = self.checkpoint_config()
        if extra_config:
            config.update({key: value for key, value in extra_config.items() if value is not None})
        write_json(config, target_dir / 'model_config.json')
        if self.mode in LORA_MODES:
            self.backbone.save_pretrained(target_dir / 'adapter')
        if tokenizer is not None:
            disable_tokenizer_thinking(tokenizer)
            tokenizer.save_pretrained(target_dir)
            chat_template_path = target_dir / 'chat_template.jinja'
            if chat_template_path.exists():
                chat_template_path.unlink()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir: str | Path, torch_dtype: Any='auto', map_location: Any='cpu', model_path: str | Path | None=None) -> 'QwenGenerativeModel':
        checkpoint_path = Path(checkpoint_dir)
        config = read_json(checkpoint_path / 'model_config.json')
        mode = str(config['mode'])
        adapter_path = checkpoint_path / 'adapter' if mode in LORA_MODES else None
        base_model_path = str(model_path) if model_path is not None else str(config['model_path'])

        def config_value(key: str, default: Any) -> Any:
            value = config.get(key, default)
            return default if value is None else value
        model = cls(model_path=base_model_path, mode=mode, lora_r=int(config_value('lora_r', 8)), lora_alpha=int(config_value('lora_alpha', 16)), lora_dropout=float(config_value('lora_dropout', 0.05)), lora_target_modules=str(config_value('lora_target_modules', 'attention_mlp')), lora_layer_scope=str(config_value('lora_layer_scope', 'last1')), torch_dtype=torch_dtype, trust_remote_code=bool(config_value('trust_remote_code', True)), adapter_path=adapter_path, adapters_trainable=False)
        return model
