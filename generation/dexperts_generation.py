from pathlib import Path
from typing import Union, List, Optional

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, modeling_utils, GPT2PreTrainedModel
from generation.gpt2_generation import GPT2Generation

from utils import utils
from utils.generation_utils import top_k_top_p_filtering

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

class DExpertsGeneration(GPT2Generation): 
    STOP_TOKEN = "<|endoftext|>"

    def __init__(
        self, 
        base_model: Union[str, Path, GPT2PreTrainedModel],
        antiexpert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        expert_model: Union[str, Path, GPT2PreTrainedModel] = None,
        tokenizer: str = 'gpt2', 
        seed: int = 42,
        steering_layer: Optional[int] = None,
        alpha: Optional[float] = 0,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        utils.set_seed(seed, n_gpu)

        self.base_model = GPT2LMHeadModel.from_pretrained(base_model).to(self.device)
        
        if antiexpert_model:
            self.antiexpert = GPT2LMHeadModel.from_pretrained(antiexpert_model).to(self.device)
        else:
            self.antiexpert = None
        
        if expert_model:
            self.expert = GPT2LMHeadModel.from_pretrained(expert_model).to(self.device)
        else:
            self.expert = None
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer, pad_token=self.STOP_TOKEN)
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

        # Prepare model if steer
        if steering_layer is not None:
            self.base_block = self.base_model.transformer.h[steering_layer]
            # TODO: Better copying and loading so that memory gets freed properly
            if self.expert is not None:
                self.expert_block = self.expert.transformer.h[steering_layer]
            else:
                self.expert_block = None
            if self.antiexpert is not None:
                self.anti_expert_block = self.antiexpert.transformer.h[steering_layer]
            else:
                self.anti_expert_block = None

            self.steer_model = self.base_model
            self.steer_model.transformer.set_steering_layer(
                    layer_num=steering_layer,
                    base_block=self.base_block,
                    expert_block=self.expert_block,
                    anti_expert_block=self.anti_expert_block,
                    alpha=alpha,
            )

    def __repr__(self):
        return f'<DExpertsGenerator model_name_or_path="{self.model}">'

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 filter_p: float = 0.9,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 alpha: float = 0.0,
                 layers_to_modify: Optional[List[int]] = None,
                 **model_kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]

        encodings_dict = self.tokenizer.batch_encode_plus(prompt, pad_to_max_length=True, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()
        with torch.no_grad():

            # Sorts layers to modify
            if layers_to_modify is not None:
                layers_to_modify.sort()

            for step in range(max_len):
                if hasattr(self, 'steer_model'):
                    ensemble_logits, steer_past = self.steer_model(
                        input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
                    
                    if filter_p < 1.0:
                        ensemble_logits = top_k_top_p_filtering(ensemble_logits, top_p=filter_p)

                else:
                    if layers_to_modify is None:
                        # base model prediction
                        base_logits, base_past = self.base_model(
                            input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
                    
                    # expert prediction
                    # TODO: Update this
                    if self.expert:
                        if layers_to_modify is not None:
                            raise NotImplementedError('TODO')
                        else:
                            expert_logits, expert_past = self.expert(
                                input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
                    else:
                        if layers_to_modify is not None:
                            raise NotImplementedError('TODO')
                        else:
                            expert_logits = base_logits
                    
                    # antiexpert prediction
                    if self.antiexpert:
                        if layers_to_modify is not None:
                            raise NotImplementedError('TODO')
                        else:
                            antiexpert_logits, antiexpert_past = self.antiexpert(
                                input_ids, attention_mask=attention_mask, position_ids=position_ids, **model_kwargs)
                    else:
                        if layers_to_modify is not None:
                            raise NotImplementedError('TODO')
                        else:
                            antiexpert_logits = base_logits

                    if layers_to_modify is not None:
                        raise NotImplementedError('TODO')
                    
                    if filter_p < 1.0:
                        base_logits = top_k_top_p_filtering(base_logits, top_p=filter_p)
                    
                    # DExperts
                    alpha = torch.tensor(alpha).to(self.device)
                    ensemble_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = ensemble_logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if k > 0 or p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs
