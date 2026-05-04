#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..geochat_arch import GeoChatMetaModel, GeoChatMetaForCausalLM


class GeoChatConfig(LlamaConfig):
    model_type = "geochat"


class GeoChatLlamaModel(GeoChatMetaModel, LlamaModel):
    config_class = GeoChatConfig

    def __init__(self, config: LlamaConfig):
        super(GeoChatLlamaModel, self).__init__(config)


class GeoChatLlamaForCausalLM(LlamaForCausalLM, GeoChatMetaForCausalLM):
    config_class = GeoChatConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = GeoChatLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _get_soft_label_matrix(self, device, dtype):
        """Lazily build and cache the soft label matrix on first use.

        The matrix is cached and only re-created if device/dtype change,
        which avoids redundant .to() calls on every forward pass.
        """
        needs_rebuild = (
            not hasattr(self, '_soft_label_matrix')
            or self._soft_label_matrix is None
        )
        needs_move = (
            not needs_rebuild
            and (self._soft_label_matrix.device != device
                 or self._soft_label_matrix.dtype != dtype)
        )

        if needs_rebuild:
            from geochat.train.soft_label_loss import (
                build_triangular_soft_matrix,
                build_binomial_soft_matrix,
                build_poisson_soft_matrix,
                build_uniform_soft_matrix,
            )

            digit_token_ids = self.config.digit_token_ids
            dist_type = getattr(self.config, 'soft_label_distribution', 'triangular')
            eta = getattr(self.config, 'soft_label_eta', 0.08)

            if dist_type == 'triangular':
                matrix = build_triangular_soft_matrix(
                    digit_token_ids, self.config.vocab_size, eta
                )
            elif dist_type == 'binomial':
                matrix = build_binomial_soft_matrix(
                    digit_token_ids, self.config.vocab_size, eta
                )
            elif dist_type == 'poisson':
                matrix = build_poisson_soft_matrix(
                    digit_token_ids, self.config.vocab_size, eta
                )
            elif dist_type == 'uniform':
                matrix = build_uniform_soft_matrix(
                    digit_token_ids, self.config.vocab_size, eta
                )
            else:
                raise ValueError(f"Unknown soft label distribution: {dist_type}")

            self._soft_label_matrix = matrix.to(device=device, dtype=dtype)

        elif needs_move:
            self._soft_label_matrix = self._soft_label_matrix.to(
                device=device, dtype=dtype
            )

        return self._soft_label_matrix

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            if getattr(self.config, 'soft_label_enable', False) and self.training:
                # Soft labeling: use distance-aware soft targets for digit tokens
                from geochat.train.soft_label_loss import compute_soft_label_loss
                soft_matrix = self._get_soft_label_matrix(
                    device=shift_logits.device, dtype=shift_logits.dtype
                )
                loss = compute_soft_label_loss(
                    shift_logits,
                    shift_labels,
                    soft_matrix,
                    self.config.digit_token_ids,
                    lambda_weight=getattr(self.config, 'soft_label_lambda', 2.0),
                )
            else:
                # Standard hard-label cross-entropy
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("geochat", GeoChatConfig)
AutoModelForCausalLM.register(GeoChatConfig, GeoChatLlamaForCausalLM)
