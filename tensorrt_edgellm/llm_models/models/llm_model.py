# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LLM Model Implementation for Causal Language Modeling

This module provides the main LLM model implementation for efficient
accelerated generation. The model supports standard models, EAGLE3
variants, and Qwen3VL with deepstack processing.

The module contains:
- EdgeLLMModel: Main LLM model class with decoder layers and normalization
- EdgeLLMModelForCausalLM: Wrapper for causal language modeling tasks
"""

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from .. import model_utils
from ..layers.gather_nd import custom_gather_nd
from ..layers.layers import (EdgeLLMDecoderLayer, EdgeLLMGatedDeltaNetLayer,
                             EdgeLLMNemotronHBlock)
from ..layers.reduced_lm_head import reduce_lm_head


class BaseEdgeLLMCausalLMWrapper(nn.Module):
    """
    Shared utilities for export-time CausalLM wrappers.

    This base class intentionally does not define a unified `forward` signature.
    Different model families expose different state inputs/outputs, and ONNX
    export relies on explicit, stable argument ordering per wrapper.
    """

    def _init_causal_lm_common(
        self,
        hf_model: nn.Module,
        language_model: nn.Module,
        config: Any,
        reduced_vocab_size: Optional[int] = None,
        vocab_map: Optional[torch.Tensor] = None,
    ) -> None:
        self.torch_dtype = hf_model.dtype
        self.config = config

        # Some wrappers expose `embed_tokens`, some only expose `get_input_embeddings()`.
        embed_layer = model_utils.resolve_input_embeddings(language_model)
        self.embed_tokens = embed_layer.to(self.torch_dtype)

        if reduced_vocab_size is not None and vocab_map is not None:
            print(
                f"Reducing vocabulary size from "
                f"{model_utils.resolve_output_embeddings(hf_model, language_model).out_features}"
                f" to {reduced_vocab_size}")
            if vocab_map.shape[0] != reduced_vocab_size:
                raise ValueError(
                    f"vocab_map size {vocab_map.shape[0]} does not match "
                    f"reduced_vocab_size {reduced_vocab_size}")
            self.lm_head = reduce_lm_head(
                model_utils.resolve_output_embeddings(hf_model, language_model),
                reduced_vocab_size, vocab_map)
        else:
            self.lm_head = model_utils.resolve_output_embeddings(
                hf_model, language_model)

    @property
    def device(self):
        return next(self.parameters()).device

    def _compute_logits(self, hidden_states: torch.Tensor,
                        last_token_ids: torch.Tensor) -> torch.Tensor:
        last_hidden_state_gathered = custom_gather_nd(hidden_states,
                                                      last_token_ids, 1)
        logits = self.lm_head(last_hidden_state_gathered)
        return logits.to(torch.float32)


class EdgeLLMModel(nn.Module):
    """
    EdgeLLM Model for causal language modeling.
    
    This model implements the main component for language modeling, supporting
    standard models and EAGLE3 variants. It processes input through
    decoder layers with proper normalization and can output hidden states
    for EAGLE variants.
    
    Attributes:
        config: Model configuration object
        padding_idx: Padding token index
        vocab_size: Size of the vocabulary
        layers: List of decoder layers
        norm: RMS normalization layer
        rotary_emb: Rotary embedding layer
        is_eagle_base: Whether this is an EAGLE3 base model
    """

    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False) -> None:
        """
        Initialize the EdgeLLM model.
        
        Args:
            hf_model: The original model (LlamaForCausalLM, Qwen2ForCausalLM, etc.)
            is_eagle_base: Whether this is an EAGLE3 base model
        """
        super().__init__()

        # Copy all the basic attributes
        self.config = hf_model.config
        self.vocab_size = self.config.vocab_size
        self.is_eagle_base = is_eagle_base

        # Keep all the original components
        self.torch_dtype = hf_model.dtype

        # embed_tokens is optional (e.g., Talker/CodePredictor use projected embeddings as input)
        if hasattr(hf_model, 'embed_tokens'):
            self.embed_tokens = hf_model.embed_tokens.to(self.torch_dtype)
        else:
            self.embed_tokens = None

        self.norm = hf_model.norm.to(self.torch_dtype)

        # Replace decoder layers with our custom ones
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayer(hf_layer, self.torch_dtype, eagle3_draft=False)
            for hf_layer in hf_model.layers
        ])

        # Set max_position_embeddings on attention modules from the model's config
        for layer in self.layers:
            layer.self_attn.max_position_embeddings = self.config.max_position_embeddings

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.FloatTensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Optional[Tuple[
            torch.Tensor, ...]]]:
        """
        Forward pass of the EdgeLLM model.
        
        Args:
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_size)
            past_key_values: Past KV cache, list of (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE embeddings (batch_size, seq_len, head_dim)
            context_lengths: Current position in cache (batch_size,)
            kvcache_start_index: Start index of KV cache (batch_size,)
            position_ids: Position IDs (batch_size, seq_len), optional
            attention_mask: Attention mask (batch_size, seq_len, seq_len + past_len), optional
            deepstack_visual_embeds: Deepstack visual embeddings for Qwen3VL, list of 3 tensors, each (batch_size, seq_len, hidden_size), optional
            output_hidden_states: Whether to output hidden states from all layers
            
        Returns:
            (hidden_states, present_key_values, all_hidden_states)
        """

        hidden_states = inputs_embeds
        present_key_values = ()
        all_hidden_states = () if output_hidden_states else None

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Get the past_key_value for this specific layer
            past_key_value = past_key_values[idx] if isinstance(
                past_key_values, (list, tuple)) else past_key_values

            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                kvcache_start_index=kvcache_start_index,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            present_key_values += (present_key_value, )

            # Apply deepstack processing for Qwen3VL and Qwen3OmniThinker
            if deepstack_visual_embeds is not None and idx in range(
                    len(deepstack_visual_embeds)):
                assert self.config.model_type in [
                    "qwen3_vl_text", "qwen3_omni_text"
                ], "Qwen3VLTextModel or Qwen3OmniTextModel is required for deepstack processing"
                hidden_states = hidden_states + deepstack_visual_embeds[idx]

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states, )

        return hidden_states, present_key_values, all_hidden_states


class EdgeLLMNemotronHModel(nn.Module):
    """EdgeLLM model implementation specialized for Nemotron-H.

    Unlike :class:`EdgeLLMModel` which assumes uniform attention+MLP layers,
    this class reads ``config.layers_block_type`` to wrap each block
    appropriately and routes state (KV cache for attention, SSM state for
    Mamba) through the correct blocks.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()

        self.config = hf_model.config
        self.vocab_size = self.config.vocab_size
        self.torch_dtype = hf_model.dtype
        norm_layer = getattr(hf_model, 'norm', None) or hf_model.norm_f
        self.norm = norm_layer.to(self.torch_dtype)

        self.block_types: List[str] = list(self.config.layers_block_type)

        self.layers = nn.ModuleList([
            EdgeLLMNemotronHBlock(hf_layer, self.torch_dtype)
            for hf_layer in hf_model.layers
        ])

        # Set max_position_embeddings on attention mixers
        for layer in self.layers:
            if layer.block_type == "attention":
                layer.mixer.max_position_embeddings = (
                    self.config.max_position_embeddings)

        # Pre-compute index maps for attention / mamba layers
        self.attn_layer_indices: List[int] = [
            i for i, bt in enumerate(self.block_types) if bt == "attention"
        ]
        self.mamba_layer_indices: List[int] = [
            i for i, bt in enumerate(self.block_types) if bt == "mamba"
        ]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_attention_layers(self) -> int:
        return len(self.attn_layer_indices)

    @property
    def num_linear_attn_layers(self) -> int:
        return len(self.mamba_layer_indices)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[
            torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns:
            (hidden_states, present_key_values, present_conv_states, present_recurrent_states)
        """
        hidden_states = inputs_embeds
        present_key_values: Tuple[torch.Tensor, ...] = ()
        present_conv_states: Tuple[torch.Tensor, ...] = ()
        present_recurrent_states: Tuple[torch.Tensor, ...] = ()

        attn_idx = 0
        mamba_idx = 0

        for idx, layer in enumerate(self.layers):
            bt = self.block_types[idx]

            if bt == "mamba":
                hidden_states, conv_state_out, ssm_state_out = layer.forward_mamba(
                    hidden_states, conv_states[mamba_idx],
                    recurrent_states[mamba_idx], context_lengths)
                present_conv_states += (conv_state_out, )
                present_recurrent_states += (ssm_state_out, )
                mamba_idx += 1

            elif bt == "attention":
                hidden_states, present_kv = layer.forward_attention(
                    hidden_states,
                    past_key_values[attn_idx],
                    rope_rotary_cos_sin,
                    context_lengths,
                    kvcache_start_index,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                present_key_values += (present_kv, )
                attn_idx += 1

            elif bt == "mlp":
                hidden_states = layer.forward_mlp(hidden_states)

            elif bt == "moe":
                hidden_states = layer.forward_moe(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values, present_conv_states, present_recurrent_states


class EdgeLLMHybridModelForCausalLM(BaseEdgeLLMCausalLMWrapper):
    """
    Causal LM wrapper for hybrid LinearAttention+Attention architectures.
    
    This wrapper provides a consistent interface for different types of hybrid
    models, including Nemotron-H and Qwen3-5. It handles model structure differences
    and provides uniform forward pass behavior.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        reduced_vocab_size: Optional[int] = None,
        vocab_map: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        language_model, config = model_utils.prepare_language_model_and_config(
            hf_model)
        self._init_causal_lm_common(hf_model, language_model, config,
                                    reduced_vocab_size, vocab_map)

        if config.model_type == "nemotron_h":
            self.model = EdgeLLMNemotronHModel(language_model)
        elif config.model_type == "qwen3_5_text":
            self.model = EdgeLLMQwen3_5Model(language_model)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[
            torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Returns:
            (logits, present_key_values, present_conv_states, present_recurrent_states)
        """
        hidden_states, present_key_values, present_conv_states, present_recurrent_states = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            conv_states=conv_states,
            recurrent_states=recurrent_states,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        logits = self._compute_logits(hidden_states, last_token_ids)

        return logits, tuple(present_key_values), tuple(
            present_conv_states), tuple(present_recurrent_states)


class EdgeLLMModelForCausalLM(BaseEdgeLLMCausalLMWrapper):
    """
    EdgeLLM Model for Causal Language Modeling.
    
    This wrapper provides a consistent interface for different types of language
    models, including standard models and EAGLE variants. It handles model
    structure differences and provides uniform forward pass behavior.
    
    Attributes:
        model: The underlying EdgeLLM model
        lm_head: Language model head for token prediction
        config: Model configuration object
        is_eagle_base: Whether this is an EAGLE3 base model
        embed_tokens: Token embedding layer
    """

    def __init__(self,
                 hf_model: nn.Module,
                 is_eagle_base: bool = False,
                 reduced_vocab_size: Optional[int] = None,
                 vocab_map: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the EdgeLLM model for causal LM.
        
        Args:
            hf_model: The original model (LlamaForCausalLM, Qwen2ForCausalLM, etc.)
            is_eagle_base: Whether this is an EAGLE3 base model
            reduced_vocab_size: Size of the reduced vocabulary (optional)
            vocab_map: Tensor of shape (reduced_vocab_size,) with int32 indices for vocabulary reduction (optional)
        """
        super().__init__()

        language_model, config = model_utils.prepare_language_model_and_config(
            hf_model)
        self._init_causal_lm_common(hf_model, language_model, config,
                                    reduced_vocab_size, vocab_map)

        # Create EdgeLLMModel with the original model
        self.model = EdgeLLMModel(language_model, is_eagle_base)

        self.is_eagle_base = is_eagle_base

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], Tuple[
            torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_size)
            past_key_values: Past KV cache, tuple of (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            rope_rotary_cos_sin: RoPE embeddings (batch_size, seq_len, head_dim)
            context_lengths: Current position in cache (batch_size,)
            last_token_ids: Indices of last tokens to extract (batch_size,)
            kvcache_start_index: Start index of KV cache (batch_size,)
            position_ids: Position IDs (batch_size, seq_len), optional
            attention_mask: Attention mask (batch_size, seq_len, seq_len + past_len), optional
            deepstack_visual_embeds: Deepstack visual embeddings for Qwen3VL, list of 3 tensors, each (batch_size, seq_len, hidden_size), optional

        Returns:
            For standard: (logits, past_key_values)
            For EAGLE3 base: (logits, past_key_values, hidden_states)
        """
        # Determine output configuration based on model type
        # Enable hidden states output for EAGLE base and Qwen3-Omni Thinker
        is_qwen3_omni_thinker = self.config.model_type == "qwen3_omni_text"
        output_hidden_states = self.is_eagle_base or is_qwen3_omni_thinker

        # Forward pass through the model
        hidden_states, present_key_values, all_hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            rope_rotary_cos_sin=rope_rotary_cos_sin,
            context_lengths=context_lengths,
            kvcache_start_index=kvcache_start_index,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        logits = self._compute_logits(hidden_states, last_token_ids)

        # Handle different model types
        if self.is_eagle_base:
            # EAGLE3 base model: return concatenated hidden states from specific layers
            idx = [
                2, ((len(all_hidden_states) - 1) // 2),
                len(all_hidden_states) - 4
            ]
            hidden_states_0 = all_hidden_states[idx[0]]
            hidden_states_1 = all_hidden_states[idx[1]]
            hidden_states_2 = all_hidden_states[idx[2]]
            hidden_states = torch.cat(
                [hidden_states_0, hidden_states_1, hidden_states_2],
                dim=-1).to(self.torch_dtype)
            return logits, hidden_states, tuple(present_key_values)

        elif is_qwen3_omni_thinker:
            # Qwen3-Omni Thinker: return accept_hidden_layer hidden states for Talker
            # accept_hidden_layer (e.g. 14): thinker_hidden (used for hidden_projection in Talker)
            # Note: Layer 0 (thinker_embed) is the same as inputs_embeds, already available in runtime
            # Output shape: [batch_size, seq_len, hidden_size]

            # Read accept_hidden_layer from config (auto from talker_config or default 14)
            accept_layer = getattr(self.config, 'accept_hidden_layer', 14)
            if hasattr(self.config, 'talker_config') and hasattr(
                    self.config.talker_config, 'accept_hidden_layer'):
                accept_layer = self.config.talker_config.accept_hidden_layer

            if accept_layer >= len(all_hidden_states):
                raise ValueError(
                    f"accept_hidden_layer ({accept_layer}) exceeds number of layers ({len(all_hidden_states)})"
                )

            hidden_states_output = all_hidden_states[accept_layer].to(
                self.torch_dtype)

            return logits, hidden_states_output, tuple(present_key_values)

        # Standard model: return only logits and kv cache (original behavior)
        return logits, tuple(present_key_values)


class EdgeLLMQwen3_5Model(nn.Module):
    """
    EdgeLLM model implementation specialized for Qwen3.5.

    This model implements the main component for Qwen3.5. It processes input through
    decoder layers with proper normalization and can output hidden states
    for Qwen3.5 variants.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()

        self.config = hf_model.config
        self.vocab_size = self.config.vocab_size
        self.torch_dtype = hf_model.dtype
        self.norm = hf_model.norm.to(self.torch_dtype)
        self.layer_types = self.config.layer_types

        self.layers = nn.ModuleList([])
        for hf_layer in hf_model.layers:
            if hf_layer.layer_type == "full_attention":
                layer = EdgeLLMDecoderLayer(hf_layer,
                                            self.torch_dtype,
                                            eagle3_draft=False)
                layer.self_attn.max_position_embeddings = self.config.max_position_embeddings
            elif hf_layer.layer_type == "linear_attention":
                layer = EdgeLLMGatedDeltaNetLayer(hf_layer, self.torch_dtype)
            else:
                raise ValueError(
                    f"Unsupported layer type: {hf_layer.layer_type}")

            self.layers.append(layer)

    @property
    def num_attention_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_linear_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
        conv_states: Tuple[torch.Tensor, ...],
        recurrent_states: Tuple[torch.Tensor, ...],
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[
            torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the Qwen3.5 model.
        
        Args:
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_size)
            past_key_values: Past KV cache, tuple of (batch_size, 2, num_kv_heads, max_position_embeddings, head_dim)
            conv_states: Conv states, tuple of (batch_size, conv_channels, conv_kernel)
            recurrent_states: Recurrent states, tuple of (batch_size, hv, kdim, vdim)
            rope_rotary_cos_sin: RoPE embeddings (batch_size, seq_len, head_dim)
            context_lengths: Current position in cache (batch_size,)
            kvcache_start_index: Start index of KV cache (batch_size,)
            position_ids: Not used in Qwen3.5 model
            attention_mask: Not used in Qwen3.5 model

        Returns:
            (hidden_states, present_key_values, present_conv_states, present_recurrent_states)
        """
        hidden_states = inputs_embeds
        present_key_values: Tuple[torch.Tensor, ...] = ()
        present_conv_states: Tuple[torch.Tensor, ...] = ()
        present_recurrent_states: Tuple[torch.Tensor, ...] = ()

        attn_idx = 0
        gdn_idx = 0
        for layer in self.layers:
            if isinstance(layer, EdgeLLMDecoderLayer):
                hidden_states, present_key_value = layer(
                    hidden_states=hidden_states,
                    past_key_value=past_key_values[attn_idx],
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    context_lengths=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                present_key_values += (present_key_value, )
                attn_idx += 1
            else:
                hidden_states, conv_state_out, recurrent_state_out = layer(
                    hidden_states=hidden_states,
                    conv_state=conv_states[gdn_idx],
                    recurrent_state=recurrent_states[gdn_idx],
                    context_lengths=context_lengths,
                )
                present_conv_states += (conv_state_out, )
                present_recurrent_states += (recurrent_state_out, )
                gdn_idx += 1

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values, present_conv_states, present_recurrent_states
