
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from unsloth_zoo.loss_utils import fused_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def uncompiled_cross_entropy_loss(self, hidden_states,):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n\n'\
    "import os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "... trainer.train() ..."

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


from torch import Tensor
import torch
from torch.nn import functional as F
from transformers.models.mllama.modeling_mllama import (F, math, List, Optional, Tuple, Union, torch, nn, PreTrainedModel, ACT2FN, Cache, GenerationMixin, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings, MllamaTextConfig, MllamaVisionConfig, __name__, MllamaPreTrainedModel, MLLAMA_START_DOCSTRING, MLLAMA_INPUTS_DOCSTRING, MllamaTextModel)

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_cross_attention_mask(cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape so it can be used by attn module
    batch_size, text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
    cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)

    # invert the mask
    inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min)

    # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = ((cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None])
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask!=torch.finfo(cross_attention_mask.dtype).min, full_text_row_masked_out_mask

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def _prepare_aspect_ratio_attention_mask(aspect_ratio_mask: torch.Tensor,
    num_patches: int,
    target_length: int,
    dtype: torch.dtype,) -> torch.Tensor:
    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.view(batch_size, max_num_tiles, 1, 1).to(dtype)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(batch_size, max_num_tiles * target_length, 1)
    attention_mask = attention_mask @ attention_mask.transpose(-1, -2) * torch.finfo(dtype).min
    attention_mask = attention_mask.unsqueeze(1)

    return attention_mask!=torch.finfo(attention_mask.dtype).min

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    embeddings = self.embedding(aspect_ratio_ids)
    embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

    if self.is_gated:
        embeddings = embeddings * self.gate.tanh()

    hidden_state = hidden_state + embeddings
    return hidden_state

class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size)
        if is_gated:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedAspectRatioEmbedding_forward(self, hidden_state, aspect_ratio_ids)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaPrecomputedPositionEmbedding_forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
    # position embeddings
    gated_position_embedding = (1 - self.gate.tanh()) * self.embedding
    hidden_state = hidden_state + gated_position_embedding.view(1, 1, self.num_patches, self.hidden_size)

    # precomputed tile position embeddings
    tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
    batch_size = hidden_state.shape[0]
    tile_position_embedding = tile_position_embedding.reshape(
        batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
    )
    gated_tile_position_embedding = self.gate.tanh() * tile_position_embedding
    hidden_state = hidden_state + gated_tile_position_embedding

    return hidden_state

class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = nn.Parameter(torch.zeros(1))

        # position embedding
        position_embedding = torch.randn(self.num_patches, self.hidden_size)
        self.embedding = nn.Parameter(self.scale * position_embedding)

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.num_patches * self.hidden_size
        )

    def forward(self, hidden_state: torch.Tensor, aspect_ratio_ids: torch.Tensor) -> torch.Tensor:
        return MllamaPrecomputedPositionEmbedding_forward(self, hidden_state, aspect_ratio_ids)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaVisionMLP_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class MllamaVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return MllamaVisionMLP_forward(self, hidden_states)


@torch.compiler.disable(recursive = False)
def MllamaVisionAttention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = None,
) -> torch.Tensor:
    query = self.q_proj(hidden_state)
    key = self.k_proj(hidden_state)
    value = self.v_proj(hidden_state)

    batch_size, q_seq_len, _ = query.shape
    _, kv_seq_len, _ = key.shape

    query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

    output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return output, attn_weights

class MllamaVisionAttention(nn.Module):
    def __init__(self, config: MllamaVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads

        self.q_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        return MllamaVisionAttention_forward(self, hidden_state, attention_mask, output_attentions)


@torch.compiler.disable(recursive = False)
def MllamaVisionSdpaAttention_forward(
    self,
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = None,
) -> torch.Tensor:
    # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
    if output_attentions: raise RuntimeError('Unsloth: Not supported')

    query = self.q_proj(hidden_state)
    key = self.k_proj(hidden_state)
    value = self.v_proj(hidden_state)

    batch_size, q_seq_len, _ = query.shape
    _, kv_seq_len, _ = key.shape

    query = query.view(batch_size, q_seq_len, self.num_heads, self.head_dim)
    key = key.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)
    value = value.view(batch_size, kv_seq_len, self.num_heads, self.head_dim)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

    output = self.o_proj(attn_output)

    return output, None

class MllamaVisionSdpaAttention(MllamaVisionAttention):
    # Adapted from MllamaVisionAttention
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ) -> torch.Tensor:
        return MllamaVisionSdpaAttention_forward(self, hidden_state, attention_mask, output_attentions)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def MllamaTextRMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

class MllamaTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MllamaTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return MllamaTextRMSNorm_forward(self, hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compiler.disable(recursive = False)
def MllamaTextCrossAttention_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = self.q_norm(query_states)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = self.k_norm(key_states)
        if past_key_value is not None:
            # if we have a new image + new tokens, we only computed key_states on that new image
            # we still update the cross key states, past_image, new_image. And use it!
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
    elif cache_position[0] != 0:
        key_states, value_states = (
            past_key_value.key_cache[self.layer_idx],
            past_key_value.value_cache[self.layer_idx],
        )
    else:
        raise ValueError(
            "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
        )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[MllamaTextConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MllamaTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return MllamaTextCrossAttention_forward(self, hidden_states, cross_attention_states, past_key_value, attention_mask, output_attentions, use_cache, cache_position)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.compiler.disable(recursive = False)
def MllamaTextCrossSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    cross_attention_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    if output_attentions: raise RuntimeError('Unsloth: Not supported')

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = self.q_norm(query_states)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            # if we have a new image + new tokens, we only computed key_states on that new image
            # we still update the cross key states, past_image, new_image. And use it!
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
    elif cache_position[0] != 0:
        key_states, value_states = (
            past_key_value.key_cache[self.layer_idx],
            past_key_value.value_cache[self.layer_idx],
        )
    else:
        raise ValueError(
            "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
        )

    key_states = self.k_norm(key_states)

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        pass


    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if attention_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0, enable_gqa=self.num_key_value_groups != 1,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

class MllamaTextCrossSdpaAttention(MllamaTextCrossAttention):
    """
    Mllama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MllamaTextCrossAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MllamaTextCrossAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return MllamaTextCrossSdpaAttention_forward(self, hidden_states, cross_attention_states, past_key_value, attention_mask, output_attentions, use_cache, cache_position)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compiler.disable(recursive = False)
def MllamaTextSelfAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: torch.Tensor,
    output_attentions: bool = False,
    use_cache: bool = False,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

class MllamaTextSelfAttention(nn.Module):
    def __init__(self, config: MllamaTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        return MllamaTextSelfAttention_forward(self, hidden_states, attention_mask, position_embeddings, output_attentions, use_cache, past_key_value, cache_position, **kwargs)


@torch.compiler.disable(recursive = False)
def MllamaTextSelfSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings: torch.Tensor,
    output_attentions: bool = False,
    use_cache: bool = False,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    if output_attentions: raise RuntimeError('Unsloth: Not supported')

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        pass


    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.dropout if self.training else 0.0, enable_gqa=self.num_key_value_groups != 1,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value

class MllamaTextSelfSdpaAttention(MllamaTextSelfAttention):
    # Adapted from MllamaTextSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        return MllamaTextSelfSdpaAttention_forward(self, hidden_states, attention_mask, position_embeddings, output_attentions, use_cache, past_key_value, cache_position, **kwargs)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def MllamaTextMLP_forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MllamaTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Ignore copy
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return MllamaTextMLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
def MllamaRotaryEmbedding_forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class MllamaRotaryEmbedding(nn.Module):
    def __init__(self, config: MllamaTextConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len


    def forward(self, x, position_ids):
        return MllamaRotaryEmbedding_forward(self, x, position_ids)


@torch.compiler.disable(recursive = False)
@add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="MllamaTextConfig")
def MllamaForCausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cross_attention_states: Optional[torch.LongTensor] = None,
    cross_attention_mask: Optional[torch.LongTensor] = None,
    full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, MllamaForCausalLM

    >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
    >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

    >>> prompt = "If I had to write a haiku, it would be:"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
    >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    >>> print(result)
    If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
    I love the idea of snowflakes gently falling, each one
    ```
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        cross_attention_states=cross_attention_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cross_attention_mask=cross_attention_mask,
        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = EMPTY_LOGITS
    loss = None
    if not self.training and labels is None:
        logits = self.lm_head(hidden_states)
    elif True and self.training and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        n_items = loss_kwargs.get("num_items_in_batch", None) or loss_kwargs.get("n_items", None)
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states,
            lm_weight          = self.lm_head.weight,
            labels             = labels,
            num_items_in_batch = n_items,
            logit_softcapping  = getattr(self.config, "final_logit_softcapping", 0),
        )
    else:
        logits = self.lm_head(hidden_states)
        loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

@add_start_docstrings(
    """The Mllama Text Model with a language modeling head on top.""",
    MLLAMA_START_DOCSTRING,
)
class MllamaForCausalLM(MllamaPreTrainedModel, GenerationMixin):
    config_class = MllamaTextConfig
    _supports_static_cache = True  # only the LLM without cross attn can do compile
    base_model_prefix = "language_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = MllamaTextModel._from_config(self.text_config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return MllamaForCausalLM_forward(self, input_ids, attention_mask, position_ids, cross_attention_states, cross_attention_mask, full_text_row_masked_out_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, num_logits_to_keep, **loss_kwargs)
