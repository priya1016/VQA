#!/usr/bin/env python
# coding: utf-8

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
import pandas as pd
from PIL import Image
import torch, gc
import torch.nn as nn
from transformers import (
    # Preprocessing / Common
    AutoTokenizer, AutoFeatureExtractor,
    # Text & Image Models (Now, image transformers like ViTModel, DeiTModel, BEiT can also be loaded using AutoModel)
    AutoModel,            
    # Training / Evaluation
    TrainingArguments, Trainer,
    # Misc
    logging
)

import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, roc_auc_score, cohen_kappa_score
import glob
import json
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from collections import defaultdict

from layer import GELU
from model_uniter_modified import UniterPreTrainedModel, UniterModel, UniterConfig
from torchvision import transforms

version = "uniter_2_modified_20_epochs"


# In[2]:


# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))


# In[4]:


dataset = load_dataset(
    "csv", 
    data_files={
        "train": os.path.join("..", "dataset", "data_train.csv"),
        "test": os.path.join("..", "dataset", "data_eval.csv"),
        "unseen": os.path.join("..", "dataset", "data_test.csv"),
    }
)

with open(os.path.join("..", "dataset", "answer_space.txt")) as f:
    answer_space = f.read().splitlines()

dataset = dataset.map(
    lambda examples: {
        'label': [
            answer_space.index(ans.replace(" ", "").split(",")[0]) # Select the 1st answer if multiple answers are provided
            for ans in examples['answer']
        ]
    },
    batched=True
)


# dataset

@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=60,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text['input_ids'].squeeze(),
            "token_type_ids": encoded_text['token_type_ids'].squeeze(),
            "attention_mask": encoded_text['attention_mask'].squeeze(),
        }

    def preprocess_images(self, images: List[str]):
        
        preprocess = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
        
        processed_images = self.preprocessor(
            images=[Image.open(os.path.join("..", "dataset", "images", image_id + ".png")).convert('RGB') for image_id in images],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
            
    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_id']
                if isinstance(raw_batch_dict, dict) else
                [i['image_id'] for i in raw_batch_dict]
            ),
            'labels': torch.tensor(
                raw_batch_dict['label']
                if isinstance(raw_batch_dict, dict) else
                [i['label'] for i in raw_batch_dict],
                dtype=torch.int64
            ),
        }

class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            nn.LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        # gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False)
        
        pooled_output = self.uniter.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

class MultimodalVQAModel(nn.Module):
    def __init__(
            self, config,
            num_heads: int = 2,
            num_trx_cells: int = 4,
            dropout: float=0.1,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'microsoft/resnet-50'):
     
        super(MultimodalVQAModel, self).__init__()
        
        num_answer = num_labels
        img_dim = 224
        self.uniter = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            nn.LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        # self.apply(self.init_weights)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            mask=None,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).to(device)
        # attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        num_bbs = [f.size(0) for f in pixel_values]
        # print(num_bbs)
        num_bbs = [3, 224, 224]
        # img_feat = pad_tensors(pixel_values, num_bbs)
        # img_pos_feat = pad_tensors(pixel_values, num_bbs)
        img_feat = pixel_values
        
        # img_pos_feat = torch.cat([img_feat, img_feat[:, 4:5]*img_feat[:, 5:]], dim=-1)
        img_pos_feat=pixel_values
        # print('attention_mask:',attention_mask.shape)
        # print('input_ids:',input_ids.shape)
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, #gather_index,
                                      output_all_encoded_layers=False)
        
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.vqa_output(pooled_output)
        
        out = {
            "logits": logits
        }

        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


# In[10]:


def createMultimodalVQACollatorAndModel(model_config,config):
    
    text = model_config['pretrained_text_name']
    image = model_config['pretrained_image_name']
    
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )

    multi_model = MultimodalVQAModel(config = config, num_heads = model_config['num_heads'], num_trx_cells = model_config['num_trx_cells'],
                                     dropout = model_config['dropout'],intermediate_dim = model_config['intermediate_dim'], 
                                     pretrained_text_name=text, pretrained_image_name=image).to(device)
    
    return multi_collator, multi_model


# In[11]:


def wup_measure(a,b,similarity_threshold=0.925):
    """
    Returns Wu-Palmer similarity score.
    More specifically, it computes:
        max_{x \in interp(a)} max_{y \in interp(b)} wup(x,y)
        where interp is a 'interpretation field'
    """
    def get_semantic_field(a):
        weight = 1.0
        semantic_field = wordnet.synsets(a,pos=wordnet.NOUN)
        return (semantic_field,weight)


    def get_stem_word(a):
        """
        Sometimes answer has form word\d+:wordid.
        If so we return word and downweight
        """
        weight = 1.0
        return (a,weight)


    global_weight=1.0

    (a,global_weight_a)=get_stem_word(a)
    (b,global_weight_b)=get_stem_word(b)
    global_weight = min(global_weight_a,global_weight_b)

    if a==b:
        # they are the same
        return 1.0*global_weight

    if a==[] or b==[]:
        return 0


    interp_a,weight_a = get_semantic_field(a) 
    interp_b,weight_b = get_semantic_field(b)

    if interp_a == [] or interp_b == []:
        return 0

    # we take the most optimistic interpretation
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if local_score > global_max:
                global_max=local_score

    # we need to use the semantic fields and therefore we downweight
    # unless the score is high which indicates both are synonyms
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0

    final_score=global_max*weight_a*weight_b*interp_weight*global_weight
    return final_score


# In[12]:


def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)



# In[15]:


def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    preds = logits.argmax(axis=-1)
    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "f1_micro": f1_score(labels, preds, average='micro'),
        "precision": precision_score(labels, preds,average='macro',zero_division=0),
        "recall": recall_score(labels, preds,average='macro',zero_division=0)
        # "roc_auc": roc_auc_score(labels, preds,average='macro',multi_class='ovr')
        # "cohen_kappa_score": cohen_kappa_score(labels, preds)
    }


config = {
    "train_txt_dbs": ["/txt/vqa_train.db",
                      "/txt/vqa_trainval.db",
                      "/txt/vqa_vg.db"],
    "train_img_dbs": ["/img/coco_train2014/", "/img/coco_val2014", "/img/vg/"],
    "val_txt_db": "/txt/vqa_devval.db",
    "val_img_db": "/img/coco_val2014/",
    "checkpoint": "/pretrain/uniter-base.pt",
    "model_config": "/src/config/uniter-base.json",
    "output_dir": "/storage/vqa/default",
    "max_txt_len": 60,
    "conf_th": 0.2,
    "max_bb": 100,
    "min_bb": 10,
    "num_bb": 36,
    "train_batch_size": 5120,
    "val_batch_size": 10240,
    "gradient_accumulation_steps": 5,
    "learning_rate": 8e-05,
    "lr_mul": 10.0,
    "valid_steps": 500,
    "num_train_steps": 6000,
    "optim": "adamw",
    "betas": [
        0.9,
        0.98
    ],
    "dropout": 0.1,
    "weight_decay": 0.01,
    "grad_norm": 2.0,
    "warmup_steps": 600,
    "seed": 42,
    "fp16": True,
    "n_workers": 1,#8
    "pin_mem": True
}

config={
        
     'vocab_size':30000,
     'hidden_size':768,
     'num_hidden_layers':12,
     'num_attention_head':12,
     'intermediate_size':3072,
     'hidden_act':"gelu",
     'hidden_dropout_prob':0.1,
     'attention_probs_dropout_prob':0.1,
     'max_position_embeddings':512,
     'type_vocab_size':2,
     'initializer_range':0.02
}
config=UniterConfig()

args = TrainingArguments(
    output_dir="checkpoint",
    seed=12345, 
    warmup_steps=600,
    lr_scheduler_type='linear', #"cosine_with_restarts",
    learning_rate=5e-3,
    # lr_decay = 10.0,
    adam_beta1=0.9,
    adam_beta2=0.98,
    max_grad_norm=2.0,
    evaluation_strategy="steps",
    eval_steps=200,
    # max_steps=3000,
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,             # Save only the last 3 checkpoints at any given time while training 
    metric_for_best_model='wups',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=20,
    fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    weight_decay=0.01,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=1,#8
    load_best_model_at_end=True,
)

# Inspired from UNITER
# args = TrainingArguments(
#     output_dir="checkpoint",
#     seed=12345, 
#     warmup_steps=600,
#     lr_scheduler_type='linear', #"cosine_with_restarts",
#     learning_rate=8e-02, #8e-05
#     max_grad_norm=0.8,
#     adam_beta1 = 0.9,
#     adam_beta2 = 0.98,
    
#     evaluation_strategy="steps",
#     eval_steps=50,
#     max_steps = 500,
#     logging_strategy="steps",
#     logging_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     save_total_limit=1,             # Save only the last 3 checkpoints at any given time while training 
#     metric_for_best_model='wups',
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
#     remove_unused_columns=False,
#     num_train_epochs=10,
#     fp16=True,
#     # warmup_ratio=0.01,
#     # learning_rate=5e-4,
#     weight_decay=0.01,
#     # gradient_accumulation_steps=2,
#     dataloader_num_workers=8,
#     load_best_model_at_end=True,
# )

#
model_config = {'num_heads': 4,
            'num_trx_cells':  2,
            'dropout': 0.1,
            'intermediate_dim': 512,
            'pretrained_text_name':  'bert-base-uncased',
            'pretrained_image_name':  'microsoft/resnet-50'}

new_ls = list(model_config.values())
config_name = '_'.join(str(e) for e in new_ls)

def createAndTrainModel(dataset, args, config, model_config, multimodal_model='bert_vit'+config_name):
    
    collator, model = createMultimodalVQACollatorAndModel(config=config,model_config = model_config)
    
    multi_args = deepcopy(args)
    multi_args.output_dir = os.path.join("..", "exp" + str(version) , "checkpoint", multimodal_model)
    multi_trainer = Trainer(
        model,
        multi_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    
    train_multi_metrics = multi_trainer.train()
    eval_multi_metrics = multi_trainer.evaluate()
    
    log_history = multi_trainer.state.log_history
    # multi_trainer.save_metrics(split="train",metrics=train_multi_metrics)
    # multi_trainer.save_metrics(split="eval",metrics=eval_multi_metrics)
    
    return collator, model, train_multi_metrics, eval_multi_metrics,log_history


# In[18]:


small_dataset = {}
small_dataset['train'] = dataset['train'].select(range(500))
small_dataset['test'] = dataset['test'].select(range(10))
small_dataset


# In[28]:



gc.collect()
torch.cuda.empty_cache()


# In[29]:


collator, model, train_multi_metrics, eval_multi_metrics,log_history = createAndTrainModel(dataset, args,config=config,model_config = model_config)

# print(log_history)
pd.DataFrame(log_history).to_csv("../exp"+str(version)+"/log_history.csv")
# In[30]:

# train_multi_metrics.save_metrics(split="train")
# eval_multi_metrics.save_metrics(split="eval")
# json.dump(train_multi_metrics, open("train_multi_metrics", 'wb'))
# json.dump(eval_multi_metrics, open("eval_multi_metrics", 'wb'))
# train_multi_metrics.to_csv("train_multi_metrics.csv")
# eval_multi_metrics.to_csv("eval_multi_metrics.csv")
pd.DataFrame([train_multi_metrics[2]]).to_csv("../exp"+str(version)+"/train_multi_metrics.csv")
pd.DataFrame([eval_multi_metrics]).to_csv("../exp"+str(version)+"/eval_multi_metrics.csv")



# In[42]:


# model = MultimodalVQAModel()

# device ='cpu'
# We use the checkpoint giving best results
# checkpoint = os.path.join("..","exp"+str(version), "checkpoint", "bert_vit")
# glob.glob(checkpoint + "/checkpoints-*")
# model.load_state_dict(torch.load(os.path.join("..","exp"+str(version), "checkpoint", "bert_vit", "checkpoint-100", "pytorch_model.bin")))
# model.to(device)


# In[37]:


# sample = collator(dataset["test"][1000:1005])

# input_ids1 = sample["input_ids"].to(device)
# token_type_ids = sample["token_type_ids"].to(device)
# attention_mask = sample["attention_mask"].to(device)
# pixel_values = sample["pixel_values"].to(device)
# labels = sample["labels"].to(device)


# # In[36]:


# model.eval()
# output = model(input_ids, pixel_values, attention_mask, token_type_ids, labels)


# # In[29]:


# preds = output["logits"].argmax(axis=-1).cpu().numpy()
# preds


# # In[22]:


# for i in range(2000, 2005):
#     print("*********************************************************")
#     showExample(train=False, id=i)
#     print("Predicted Answer:\t", answer_space[preds[i-2000]])
#     print("*********************************************************")


# # In[23]:


def countTrainableParameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No. of trainable parameters:\t{0:,}".format(num_params))
    return num_params


# In[24]:


pd.DataFrame([countTrainableParameters(model)]).to_csv("../exp"+str(version)+"/num_train_para.csv") # For BERT-ViT model


# In[ ]:




