import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import abc
import numpy as np
import logging
import re

from fgrlhf.reward import BasicReward
from fgrlhf.reward_utils import split_text_to_subsentences, split_text_to_sentences
from fgrlhf.evaluators import get_rouge_scores

logging.basicConfig(level=logging.ERROR)


class PreferenceReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 mean = 0.0,
                 std = 1.0,
                 bias = 0.0,
                 scale = 1.0,
                 ):
        
        # use mean and std to normalize the reward
        # use bias and scale to rescale the reward
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.mean = mean
        self.std = std
        
        self.bias = bias
        self.scale = scale
        
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None,
                   ):
        batch_reward_inputs = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input = f"{' '.join(meta['prompt'].split())} answer: {gen_text}"
            batch_reward_inputs.append(reward_input)
        
        # get the reward
        with torch.no_grad():
            # to align with the token classification model
            inputs =self.reward_tokenizer(batch_reward_inputs, 
                                          truncation=True, padding=True, 
                                          return_tensors="pt")
            inputs = inputs.to(self.model.device)
            outputs = self.model(**inputs)
            sequence_level_reward = outputs['logits'].squeeze(-1).tolist() 
        
        # align with generated texts, make it fine-grained
        fine_grained_reward = [
            [0.] * (l-1) + [((r-self.mean)/self.std)*self.scale + self.bias]
            for r, l in zip(sequence_level_reward, torch.sum(generated_attention_mask, dim=1).tolist())
        ]
        
        return fine_grained_reward


class FineGrainedReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 reward_config,
                 kl_coef,
                 sep = "</s>"
                ):
        
        super().__init__(kl_coef)

        self.reward_models = {}
        for rm_name, rm_args in reward_config.items():
        
            self.reward_models[rm_name] = PreferenceReward(tokenizer,
                rm_args.pop('ckpt'),
                **rm_args
            )
        
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_finegrained_reward(self, prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata):

        individual_rewards = {}
        fine_grained_rewards = []

        for rm_name, rm in self.reward_models.items():
            individual_rewards[rm_name] = rm.get_reward(prompts_input_ids, prompts_attention_mask,
                                                        generated_input_ids, generated_attention_mask,
                                                        generated_texts, metadata)

        
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):
            
            fine_grained_reward = [sum(rewards) for rewards in zip(*[individual_rewards[rm_name][text_idx] for rm_name in self.reward_models.keys()])]
            
            fine_grained_rewards.append(fine_grained_reward)
            
        return {"rewards": fine_grained_rewards}
        
        

    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None, 
                   ):
        
        rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        return {'rewards/raw': rewards_output['rewards']}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                metadata=None, 
                ):
        raise NotImplementedError
        output = {}
        
        finegrained_rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        # convert finegrained rewards to portions
        n_sub_sentences = finegrained_rewards_output['n_sub_sentences']
        n_sentences = finegrained_rewards_output['n_sentences']
        
        relevance_ratios = []
        factuality_ratios = []
        completeness_rewards = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            # verbosity reward
            n_sub_sentence = n_sub_sentences[text_idx]
            n_verbosity_correct = finegrained_rewards_output['n_verbosity_correct'][text_idx]
            relevance_ratios.append(n_verbosity_correct/ n_sub_sentence)
            
            # factuality reward
            n_sentence = n_sentences[text_idx]
            n_factuality_correct = finegrained_rewards_output['n_factuality_correct'][text_idx]
            factuality_ratios.append(n_factuality_correct / n_sentence)
            
            # completeness reward
            completeness_rewards.append((finegrained_rewards_output['completeness_rewards'][text_idx][-1]-self.completeness_reward_bias)/self.completeness_reward_scale)
        
        # compute rouge scores
        rouge_scores = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
        
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rouge": rouge_scores,
            "eval/rewards": [np.sum(sublist) for sublist in finegrained_rewards_output['rewards']],
            "eval/relevance_ratios": relevance_ratios,
            "eval/factuality_ratios": factuality_ratios,
            "eval/completeness_rewards": completeness_rewards,
            "eval/n_sub_sentences": n_sub_sentences,
            "eval/n_sentences": n_sentences,
            "eval/lengths": generation_lens
        })
        
        return output
    
    
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        
        # relevance ratios and factual ratios are weighted by the number of (sub)sentences
        
        stats['eval/relevance_ratios'] = (np.sum([row[wandb_table.columns.index('eval/relevance_ratios')] 
                                                  * row[wandb_table.columns.index('eval/n_sub_sentences')] 
                                                  for row in wandb_table.data]) 
                                          / np.sum([row[wandb_table.columns.index('eval/n_sub_sentences')] 
                                                    for row in wandb_table.data]))
        
        stats['eval/factuality_ratios'] = (np.sum([row[wandb_table.columns.index('eval/factuality_ratios')]
                                                   * row[wandb_table.columns.index('eval/n_sentences')]
                                                   for row in wandb_table.data])
                                           / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                                                     for row in wandb_table.data]))
        
        return stats
    
