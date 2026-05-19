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

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (LlamaRMSNorm,
                                                      LlamaRotaryEmbedding)

try:
    import modelopt.torch.opt as mto
except ImportError:
    mto = None

from .. import model_utils
from ..layers.gather_nd import custom_gather_nd
from ..layers.layers import EdgeLLMDecoderLayerTRTNative
from ..layers.reduced_lm_head import reduce_lm_head


class EdgeLLMModelTRTNative(nn.Module):
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
    """

    def __init__(
        self,
        hf_model: nn.Module,
        is_eagle_base: bool = False,
        reduced_vocab_size: Optional[int] = None,
        vocab_map: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        language_model, config = model_utils.prepare_language_model_and_config(
            hf_model)
        self.torch_dtype = hf_model.dtype
        self.language_model = language_model
        self.config = config
        self.is_eagle_base = is_eagle_base

        # Handle lm_head with optional vocabulary reduction
        self.lm_head = model_utils.resolve_output_embeddings(
            hf_model, language_model)
        if reduced_vocab_size is not None and vocab_map is not None:
            self.lm_head = reduce_lm_head(self.lm_head, reduced_vocab_size,
                                          vocab_map)

        self.embed_tokens = model_utils.resolve_input_embeddings(
            language_model).to(self.torch_dtype)
        self.norm = language_model.norm.to(self.torch_dtype)

        # Replace decoder layers with our custom ones
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayerTRTNative(layer, self.torch_dtype)
            for layer in language_model.layers
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
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        k_caches: Tuple[torch.Tensor, ...],
        v_caches: Tuple[torch.Tensor, ...],
        kvcache_start_index: torch.Tensor,
        position_ids: Union[torch.Tensor, None],
        attention_mask: Union[torch.Tensor, None],
        deepstack_visual_embeds: Union[list[torch.Tensor], None],
    ):
        """
        Forward pass of the model.
        
        Args:
            inputs_embeds: Input embeddings, shape (batch_size, seq_len, hidden_size)
            rope_rotary_cos_sin: RoPE rotary embeddings, shape (batch_size, seq_len, rotary_dim)
            context_lengths: Context length tensor indicating current position in cache, shape (batch_size,)
            last_token_ids: Indices of the last tokens to extract, shape (batch_size,)
            k_caches: Key caches for TensorRT native mode (batch, num_heads, capacity, head_dim)
            v_caches: Value caches for TensorRT native mode (batch, num_heads, capacity, head_dim)
            kvcache_start_index: Start index of KV cache of shape (batch_size)
            position_ids: Position IDs for positional encoding, shape (batch_size, seq_len), optional
            attention_mask: Attention mask, shape (batch_size, seq_len, seq_len + past_len), optional
            deepstack_visual_embeds: List of deepstack visual embeddings tensors, each with shape (visual_seqlen, hidden_size), optional (used with deepstack processing)
        Returns:
            Union[Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], torch.Tensor]]: Model outputs
                - For standard models: (logits, k_caches, v_caches)
                - For EAGLE3 base: (logits, k_caches, v_caches, hidden_states)
        """

        # Determine output configuration based on model type
        output_hidden_states = self.is_eagle_base

        hidden_states = inputs_embeds
        present_k_caches = ()
        present_v_caches = ()
        all_hidden_states = () if output_hidden_states else None

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            k_cache = k_caches[idx]
            v_cache = v_caches[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states, )

            hidden_states, present_k_cache, present_v_cache = decoder_layer(
                hidden_states=hidden_states,
                k_cache=k_cache,
                v_cache=v_cache,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                kvcache_start_index=kvcache_start_index,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            present_k_caches += (present_k_cache, )
            present_v_caches += (present_v_cache, )

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

        # Extract last token hidden states and compute logits
        # Use custom_gather_nd for all models to support batch dimensions
        last_hidden_state_gathered = custom_gather_nd(hidden_states,
                                                      last_token_ids, 1)

        logits = self.lm_head(last_hidden_state_gathered)
        logits = logits.to(torch.float32)

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
            hidden_states_output = torch.cat(
                [hidden_states_0, hidden_states_1, hidden_states_2],
                dim=-1).to(self.torch_dtype)
            return logits, tuple(present_k_caches), tuple(
                present_v_caches), hidden_states_output

        # Standard model: return logits and caches
        return logits, tuple(present_k_caches), tuple(present_v_caches)

    def prepare_onnx_required_arguments(
        self, model_config, device
    ) -> Tuple[List[Union[torch.Tensor, Tuple[torch.Tensor], None]], List[str],
               Dict[str, Dict[int, str]]]:
        """
        Prepare the required arguments for ONNX export.
        The order should align with the order of the arguments in the forward method.

        Args:
            model_config: Model configuration object
            device: Device to run the model on
        Returns:
            Tuple[List[Union[torch.Tensor, Tuple[torch.Tensor], None]], List[str], Dict[str, Dict[int, str]]]: (dummy_inputs, input_names, dynamic_axes, output_names)
        """

        dummy_inputs = []
        input_names = []
        dynamic_axes = {}
        output_names = []

        # Dynamic axes, using dummy shapes
        dummy_batch_size = 1
        dummy_seq_len = 1
        dummy_past_len = 1
        dummy_image_token_len = 1
        dummy_num_selected_tokens = 1

        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        max_position_embeddings = model_config.max_position_embeddings
        max_kv_cache_capacity = 4096  # TRT KVCacheUpdate layer requires a static value for capacity

        # Use head_dim from config if available, otherwise calculate from hidden_size
        if hasattr(model_config, 'head_dim'):
            head_dim = model_config.head_dim
        else:
            head_dim = hidden_size // num_heads

        # Determine rotary dimension from partial_rotary_factor if provided
        partial_rotary_factor = getattr(model_config, 'partial_rotary_factor',
                                        1.0)
        rotary_dim = int(head_dim * float(partial_rotary_factor))
        if rotary_dim <= 0 or rotary_dim > head_dim:
            rotary_dim = head_dim

        # inputs_embeds
        shape = (dummy_batch_size, dummy_seq_len, hidden_size)
        inputs_embeds = torch.randn(shape, dtype=torch.float16, device=device)
        dummy_inputs.append(inputs_embeds)
        input_names.append('inputs_embeds')
        dynamic_axes['inputs_embeds'] = {
            0: 'batch_size',
            1: 'seq_len',
        }

        # rope_rotary_cos_sin
        shape = (dummy_batch_size, max_position_embeddings, rotary_dim)
        rope_rotary_cos_sin = torch.randn(shape,
                                          dtype=torch.float32,
                                          device=device)
        dummy_inputs.append(rope_rotary_cos_sin)
        input_names.append('rope_rotary_cos_sin')
        dynamic_axes['rope_rotary_cos_sin'] = {
            0: 'rope_batch_size',
            1: 'max_position_embeddings'
        }

        # context_lengths
        shape = (dummy_batch_size, )
        context_lengths = torch.zeros(shape, dtype=torch.int32, device=device)
        dummy_inputs.append(context_lengths)
        input_names.append('context_lengths')
        dynamic_axes['context_lengths'] = {0: 'batch_size'}

        # last_token_ids
        shape = (dummy_batch_size, dummy_num_selected_tokens)
        last_token_ids = torch.zeros(shape, dtype=torch.int64, device=device)
        dummy_inputs.append(last_token_ids)
        input_names.append('last_token_ids')
        if self.is_eagle_base:
            dynamic_axes['last_token_ids'] = {
                0: 'batch_size',
                1: 'num_selected_tokens'
            }
        else:
            dynamic_axes['last_token_ids'] = {0: 'batch_size'}

        # k_caches
        shape = (dummy_batch_size, num_kv_heads, max_kv_cache_capacity,
                 head_dim)
        k_caches = [torch.zeros(shape, dtype=torch.float16, device=device)
                    ] * num_layers
        dummy_inputs.append(tuple(k_caches))
        k_caches_names = [f'k_cache_{i}' for i in range(num_layers)]
        input_names.extend(k_caches_names)
        dynamic_axes.update({
            k_caches_names[i]: {
                0: 'batch_size',
            }
            for i in range(num_layers)
        })

        # v_caches
        shape = (dummy_batch_size, num_kv_heads, max_kv_cache_capacity,
                 head_dim)
        v_caches = [torch.zeros(shape, dtype=torch.float16, device=device)
                    ] * num_layers
        dummy_inputs.append(v_caches)
        v_caches_names = [f'v_cache_{i}' for i in range(num_layers)]
        input_names.extend(v_caches_names)
        dynamic_axes.update({
            v_caches_names[i]: {
                0: 'batch_size',
            }
            for i in range(num_layers)
        })

        # kvcache_start_index
        shape = (dummy_batch_size, )
        kvcache_start_index = torch.zeros(shape,
                                          dtype=torch.int32,
                                          device=device)
        dummy_inputs.append(kvcache_start_index)
        input_names.append('kvcache_start_index')
        dynamic_axes['kvcache_start_index'] = {0: 'batch_size'}

        # position_ids
        if self.is_eagle_base:
            shape = (dummy_batch_size, dummy_seq_len)
            position_ids = torch.zeros(shape, dtype=torch.int32, device=device)
            dummy_inputs.append(position_ids)
            input_names.append('attention_pos_id')
            dynamic_axes['attention_pos_id'] = {0: 'batch_size', 1: 'seq_len'}
        else:
            # Vanilla decoding do not use this, adding a placeholder for ONNX export alignment
            dummy_inputs.append(None)

        # attention_mask
        if self.is_eagle_base:
            shape = (dummy_batch_size, 1, dummy_seq_len,
                     dummy_seq_len + dummy_past_len)
            attention_mask = torch.zeros(shape,
                                         dtype=torch.bool,
                                         device=device)
            dummy_inputs.append(attention_mask)
            input_names.append('attention_mask')
            dynamic_axes['attention_mask'] = {
                0: 'batch_size',
                2: 'seq_len',
                3: 'seq_len + past_len'
            }
        else:
            # Vanilla decoding do not use this, adding a placeholder for ONNX export alignment
            dummy_inputs.append(None)

        # deepstack_visual_embeds
        if model_config.model_type == "qwen3_vl_text":
            shape = (dummy_image_token_len, hidden_size)
            num_deepstack_features = 3
            deepstack_visual_embeds = [
                torch.zeros(shape, dtype=torch.float16, device=device)
            ] * num_deepstack_features
            dummy_inputs.append(deepstack_visual_embeds)
            deepstack_visual_embeds_names = [
                f'deepstack_feature.{i}' for i in range(num_deepstack_features)
            ]
            input_names.extend(deepstack_visual_embeds_names)
            dynamic_axes.update({
                deepstack_visual_embeds_names[i]: {
                    0: 'image_token_len',
                }
                for i in range(num_deepstack_features)
            })
        else:
            dummy_inputs.append(None)

        # prepare output names
        output_names.append('logits')
        output_names.extend(
            [f'present_k_cache_{i}' for i in range(num_layers)])
        output_names.extend(
            [f'present_v_cache_{i}' for i in range(num_layers)])

        # Add hidden_states output for eagle_base
        if self.is_eagle_base:
            output_names.append('hidden_states')

        return dummy_inputs, input_names, dynamic_axes, output_names


class Eagle3DraftModelTRTNative(nn.Module):
    """
    EAGLE3 Draft Model with TensorRT Native Operations.
    
    This model implements the draft component of EAGLE3 for speculative decoding
    using TensorRT native operations. It predicts multiple tokens ahead to accelerate
    generation with an enhanced architecture using separate K/V caches.
    
    Attributes:
        config: Model configuration object
        padding_idx: Padding token index
        vocab_size: Size of the vocabulary
        embed_tokens: Token embedding layer
        draft_vocab_size: Size of the draft vocabulary (may differ from vocab_size)
        lm_head: Language model head for token prediction
        target_hidden_size: Target hidden size for fusion
        hidden_size: Model hidden size
        fc: Fusion layer for combining different hidden states
        layers: List of decoder layers with native ops
        norm: RMS normalization layer
    """

    def __init__(
        self,
        config: any,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize the EAGLE3 draft model with native ops.
        
        Args:
            config: Model configuration object containing model parameters
            torch_dtype: Data type for the model
        """
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size",
                                        config.vocab_size)
        self.register_buffer(
            "d2t", torch.empty(self.draft_vocab_size, dtype=torch.int32))

        # Handle target hidden size for fusion
        self.target_hidden_size = config.target_hidden_size if hasattr(
            config, "target_hidden_size") else config.hidden_size
        self.hidden_size = config.hidden_size

        # Fusion layer for combining hidden states
        bias = getattr(config, "bias", False)
        self.fc = nn.Linear(self.target_hidden_size * 3,
                            self.hidden_size,
                            bias=bias)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         padding_idx=config.pad_token_id)

        # Decoder layers using EdgeLLMDecoderLayerTRTNative with eagle3_draft=True
        self.layers = nn.ModuleList([
            EdgeLLMDecoderLayerTRTNative(config,
                                         self.torch_dtype,
                                         index,
                                         eagle3_draft=True)
            for index in range(config.num_hidden_layers)
        ])

        # RMS normalization layer
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Language model head for token prediction
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.draft_vocab_size,
                                 bias=False)

        # Set max_position_embeddings on attention modules from the model's config
        for layer in self.layers:
            layer.self_attn.max_position_embeddings = self.config.max_position_embeddings

        # This logic is adapted from the transformers implementation for Qwen2.5-VL
        # See:https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py#L262
        if config.rope_scaling is not None and "type" in config.rope_scaling:
            if config.rope_scaling["type"] == "mrope":
                config.rope_scaling["type"] = "default"
            config.rope_scaling["rope_type"] = config.rope_scaling["type"]
        # Set default rope theta to 10000 if not specified
        # See: https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py#L111
        if config.rope_scaling is None and not hasattr(config, "rope_theta"):
            print(
                "Warning: rope_theta is not specified, setting default rope_theta to 10000 for EAGLE3 draft model"
            )
            config.rope_theta = 10000.0
        # We use the LlamaRotaryEmbedding for both Qwen2.5-VL and Llama because our quantization process only deals with text inputs.
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    @property
    def device(self):
        """Get the device of the model's parameters."""
        return next(self.parameters()).device

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        last_token_ids: torch.Tensor,
        k_caches: Tuple[torch.Tensor, ...],
        v_caches: Tuple[torch.Tensor, ...],
        kvcache_start_index: torch.Tensor,
        hidden_states_from_base: torch.Tensor,
        hidden_states_from_draft: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Tuple[
            torch.Tensor, ...]]:
        """
        Forward pass of the EAGLE3 draft model with native ops.
        
        Args:
            inputs_embeds: Input embeddings (batch_size, seq_len, hidden_size)
            rope_rotary_cos_sin: RoPE embeddings (batch_size, seq_len, head_dim)
            context_lengths: Current position in cache (batch_size,)
            last_token_ids: Indices of last tokens to extract (batch_size,)
            k_caches: Key caches for TensorRT native mode (batch, num_heads, capacity, head_dim)
            v_caches: Value caches for TensorRT native mode (batch, num_heads, capacity, head_dim)
            kvcache_start_index: Start index of KV cache (batch_size,)
            hidden_states_from_base: Hidden states from base model (batch_size, seq_len, target_hidden_size * 3)
            hidden_states_from_draft: Hidden states from draft (batch_size, seq_len, hidden_size)
            position_ids: Position IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len, seq_len + past_len)
            
        Returns:
            (logits, hidden_states, present_k_caches, present_v_caches)
        """

        # Fuse hidden states and combine with draft hidden states
        hidden_states = self.fc(hidden_states_from_base)

        # TODO: WAR for INT4 ONNX export
        hidden_states_from_draft = hidden_states_from_draft.to(torch.float16)
        hidden_states = hidden_states.to(torch.float16)
        hidden_states = hidden_states_from_draft + hidden_states

        present_k_caches = ()
        present_v_caches = ()

        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            k_cache = k_caches[idx]
            v_cache = v_caches[idx]

            hidden_states, present_k_cache, present_v_cache = decoder_layer(
                hidden_states=hidden_states,
                k_cache=k_cache,
                v_cache=v_cache,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                context_lengths=context_lengths,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kvcache_start_index=kvcache_start_index,
            )
            present_k_caches += (present_k_cache, )
            present_v_caches += (present_v_cache, )

        # Extract last token hidden states using custom_gather_nd to support batch dimensions
        hidden_states = custom_gather_nd(hidden_states, last_token_ids, 1)
        hidden_states_normed = self.norm(hidden_states)
        logits = self.lm_head(hidden_states_normed)
        logits = logits.to(torch.float32)
        logits = self.log_softmax(logits)

        return logits, hidden_states, tuple(present_k_caches), tuple(
            present_v_caches)

    @classmethod
    def from_pretrained(
        cls,
        draft_model_dir: str,
        base_model_dir: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> "Eagle3DraftModelTRTNative":
        """
        Load a pre-trained EAGLE3 draft model.
        
        Args:
            draft_model_dir: Path to the draft model directory
            base_model_dir: Base model directory to copy weights from if needed
            device: Device to load the model on ("cpu", "cuda", or "cuda:0", "cuda:1", etc.)

        Returns:
            Eagle3DraftModel: Loaded EAGLE3 draft model instance
            
        Raises:
            FileNotFoundError: If model files cannot be found
        """
        # Load configuration
        config = model_utils.get_eagle3_draft_config(draft_model_dir)
        model = cls(config)

        # Load from quantized model if it exists
        quantized_model_path = os.path.join(draft_model_dir,
                                            "modelopt_quantized_model.pth")
        if os.path.exists(quantized_model_path):
            if mto is None:
                raise ImportError(
                    "nvidia-modelopt is required to restore quantized TRT-native models."
                )
            mto.restore(model, quantized_model_path)
            return model

        # Load and prepare weights
        processed_state_dict = model_utils.load_and_prepare_eagle3_draft_weights(
            draft_model_dir, base_model_dir, device)

        # Load state dict into model
        model.load_state_dict(processed_state_dict, strict=False)

        return model

    def prepare_onnx_required_arguments(
        self, model_config, device
    ) -> Tuple[List[Union[torch.Tensor, Tuple[torch.Tensor], None]], List[str],
               Dict[str, Dict[int, str]]]:
        """
        Prepare the required arguments for ONNX export.
        The order should align with the order of the arguments in the forward method.

        Args:
            model_config: Model configuration object
            device: Device to run the model on
        Returns:
            Tuple[List[Union[torch.Tensor, Tuple[torch.Tensor], None]], List[str], Dict[str, Dict[int, str]]]: (dummy_inputs, input_names, dynamic_axes, output_names)
        """

        dummy_inputs = []
        input_names = []
        dynamic_axes = {}
        output_names = []

        # Dynamic axes, using dummy shapes
        dummy_batch_size = 1
        dummy_seq_len = 1
        dummy_past_len = 1

        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        max_position_embeddings = model_config.max_position_embeddings
        max_kv_cache_capacity = 4096  # TRT KVCacheUpdate layer requires a static value for capacity

        # Use head_dim from config if available, otherwise calculate from hidden_size
        if hasattr(model_config, 'head_dim'):
            head_dim = model_config.head_dim
        else:
            head_dim = hidden_size // num_heads

        # Determine rotary dimension from partial_rotary_factor if provided
        partial_rotary_factor = getattr(model_config, 'partial_rotary_factor',
                                        1.0)
        rotary_dim = int(head_dim * float(partial_rotary_factor))
        if rotary_dim <= 0 or rotary_dim > head_dim:
            rotary_dim = head_dim

        # Target hidden size for fusion
        target_hidden_size = model_config.target_hidden_size if hasattr(
            model_config, "target_hidden_size") else hidden_size

        # inputs_embeds
        shape = (dummy_batch_size, dummy_seq_len, hidden_size)
        inputs_embeds = torch.randn(shape, dtype=torch.float16, device=device)
        dummy_inputs.append(inputs_embeds)
        input_names.append('inputs_embeds')
        dynamic_axes['inputs_embeds'] = {
            0: 'batch_size',
            1: 'seq_len',
        }

        # rope_rotary_cos_sin
        shape = (dummy_batch_size, max_position_embeddings, rotary_dim)
        rope_rotary_cos_sin = torch.randn(shape,
                                          dtype=torch.float32,
                                          device=device)
        dummy_inputs.append(rope_rotary_cos_sin)
        input_names.append('rope_rotary_cos_sin')
        dynamic_axes['rope_rotary_cos_sin'] = {
            0: 'rope_batch_size',
            1: 'max_position_embeddings'
        }

        # context_lengths
        shape = (dummy_batch_size, )
        context_lengths = torch.zeros(shape, dtype=torch.int32, device=device)
        dummy_inputs.append(context_lengths)
        input_names.append('context_lengths')
        dynamic_axes['context_lengths'] = {0: 'batch_size'}

        # last_token_ids
        shape = (dummy_batch_size, 1)
        last_token_ids = torch.zeros(shape, dtype=torch.int64, device=device)
        dummy_inputs.append(last_token_ids)
        input_names.append('last_token_ids')
        dynamic_axes['last_token_ids'] = {
            0: 'batch_size',
            1: 'num_selected_tokens'
        }

        # k_caches
        shape = (dummy_batch_size, num_kv_heads, max_kv_cache_capacity,
                 head_dim)
        k_caches = [torch.zeros(shape, dtype=torch.float16, device=device)
                    ] * num_layers
        dummy_inputs.append(tuple(k_caches))
        k_caches_names = [f'k_cache_{i}' for i in range(num_layers)]
        input_names.extend(k_caches_names)
        dynamic_axes.update({
            k_caches_names[i]: {
                0: 'batch_size',
            }
            for i in range(num_layers)
        })

        # v_caches
        shape = (dummy_batch_size, num_kv_heads, max_kv_cache_capacity,
                 head_dim)
        v_caches = [torch.zeros(shape, dtype=torch.float16, device=device)
                    ] * num_layers
        dummy_inputs.append(v_caches)
        v_caches_names = [f'v_cache_{i}' for i in range(num_layers)]
        input_names.extend(v_caches_names)
        dynamic_axes.update({
            v_caches_names[i]: {
                0: 'batch_size',
            }
            for i in range(num_layers)
        })

        # kvcache_start_index
        shape = (dummy_batch_size, )
        kvcache_start_index = torch.zeros(shape,
                                          dtype=torch.int32,
                                          device=device)
        dummy_inputs.append(kvcache_start_index)
        input_names.append('kvcache_start_index')
        dynamic_axes['kvcache_start_index'] = {0: 'batch_size'}

        # hidden_states_from_base
        shape = (dummy_batch_size, dummy_seq_len, target_hidden_size * 3)
        hidden_states_from_base = torch.randn(shape,
                                              dtype=torch.float16,
                                              device=device)
        dummy_inputs.append(hidden_states_from_base)
        input_names.append('hidden_states_input')
        dynamic_axes['hidden_states_input'] = {0: 'batch_size', 1: 'seq_len'}

        # hidden_states_from_draft
        shape = (dummy_batch_size, dummy_seq_len, hidden_size)
        hidden_states_from_draft = torch.randn(shape,
                                               dtype=torch.float16,
                                               device=device)
        dummy_inputs.append(hidden_states_from_draft)
        input_names.append('hidden_states_from_draft')
        dynamic_axes['hidden_states_from_draft'] = {
            0: 'batch_size',
            1: 'seq_len'
        }

        # position_ids
        shape = (dummy_batch_size, dummy_seq_len)
        position_ids = torch.zeros(shape, dtype=torch.int32, device=device)
        dummy_inputs.append(position_ids)
        input_names.append('attention_pos_id')
        dynamic_axes['attention_pos_id'] = {0: 'batch_size', 1: 'seq_len'}

        # attention_mask
        shape = (dummy_batch_size, 1, dummy_seq_len,
                 dummy_seq_len + dummy_past_len)
        attention_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        dummy_inputs.append(attention_mask)
        input_names.append('attention_mask')
        dynamic_axes['attention_mask'] = {
            0: 'batch_size',
            2: 'seq_len',
            3: 'seq_len + past_len'
        }

        # prepare output names
        output_names.append('logits')
        output_names.append('hidden_states')
        output_names.extend(
            [f'present_k_cache_{i}' for i in range(num_layers)])
        output_names.extend(
            [f'present_v_cache_{i}' for i in range(num_layers)])

        return dummy_inputs, input_names, dynamic_axes, output_names
