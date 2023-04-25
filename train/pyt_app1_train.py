#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
import pandas as pd
from PIL import Image
import torch, gc
import torch.nn as nn
from layer import BertLayer, BertPooler

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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet


from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, roc_auc_score, cohen_kappa_score
import glob
import json


version = "1_pyt_full_wo_init"


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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # print(x.shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
            num_heads: int = 2,
            num_trx_cells: int = 4,
            dropout: float=0.1,
            num_labels: int = len(answer_space),
            intermediate_dim: int = 512,
            pretrained_text_name: str = 'bert-base-uncased',
            pretrained_image_name: str = 'google/vit-base-patch16-224-in21k'):
     
        super(MultimodalVQAModel, self).__init__()
        self.num_labels = num_labels
        self.pretrained_text_name = pretrained_text_name
        self.pretrained_image_name = pretrained_image_name
        
        self.text_encoder = AutoModel.from_pretrained(
            self.pretrained_text_name,
        )
        self.image_encoder = AutoModel.from_pretrained(
            self.pretrained_image_name,
        )
        
        embed_dim = self.text_encoder.config.hidden_size
        trx_ff_dim = intermediate_dim
        num_class = self.num_labels
        
        self.pos_encoder1 = PositionalEncoding(embed_dim, dropout)
        self.pos_encoder2 = PositionalEncoding(self.image_encoder.config.hidden_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim+self.image_encoder.config.hidden_size, num_heads, dim_feedforward=intermediate_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_trx_cells)
        self.encoder = nn.Embedding(num_class, intermediate_dim)
        self.linear_layer = nn.Linear(embed_dim+self.image_encoder.config.hidden_size,intermediate_dim)
        self.d_model = intermediate_dim
        # print(intermediate_dim)
        self.decoder = nn.Linear(intermediate_dim, num_class)
        
#         self.pe_layer_text = PositionalEncoding(embed_dim)
#         self.pe_layer_image = PositionalEncoding(self.image_encoder.config.hidden_size)
        
#         self.te_layer = TransformerEncoder(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size,num_heads,trx_ff_dim,num_trx_cells,dropout)        
        self.output_layer = nn.Linear(embed_dim + self.image_encoder.config.hidden_size, num_class)
        
        # self.fusion = nn.Sequential(
        #     nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # )
        
        # self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
#         self.init_weights()
    
#     def init_weights(self) -> None:
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.linear_layer.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
        
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
            mask=None,
            attention_mask: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None):
        
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
       
        # src = torch.cat([
        #             torch.nn.functional.normalize(encoded_text['pooler_output']),
        #             torch.nn.functional.normalize(encoded_image['pooler_output'])
        #         ],dim=1).to(torch.int64)
        # print(src.shape)
        src1 = encoded_text['pooler_output']
        src2 = encoded_image['pooler_output']
        
        # src1 = torch.transpose(src1,0,1)
        # src2 = torch.transpose(src2,0,1)
        src1 = self.pos_encoder1(src1)
        src2 = self.pos_encoder2(src2)
        # print(src.shape)
        
        # print("src1 ",src1.shape)
        # print("src2 ",src2.shape)
        src = torch.cat([
                    torch.nn.functional.normalize(src1),
                    torch.nn.functional.normalize(src2)
                ],dim=2).to(torch.int64)
        # print("src ",src.shape)
        
        
        # src = self.encoder(src) #* math.sqrt(self.text_encoder.config.hidden_size+self.image_encoder.config.hidden_size)
        
        # print(self.text_encoder.config.hidden_size+self.image_encoder.config.hidden_size)
        # print(src.shape)
        # print(attention_mask.shape)
        
        # print(output.shape)
        output = self.transformer_encoder(src.to(torch.float32), mask)
        # print(output.shape)
        output = self.linear_layer(output)
        # print(output.shape)
        output = self.decoder(output)
        
        # print(encoded_text['pooler_output'])
        # print(encoded_text['pooler_output'].shape)
        # logits_1 = self.pe_layer_text(encoded_text['pooler_output'])
        # logits_2 = self.pe_layer_image(encoded_image['pooler_output'])
        # print(logits_1.shape)
        # print(logits_2.shape)
        # print(torch.cat([logits_1,logits_2],dim=2).shape)

        # logits = self.te_layer(torch.cat([logits_1,logits_2],dim=2),mask)
        
#         logits = self.te_layer(torch.cat(
#                 [
#                     torch.nn.functional.normalize(encoded_text['pooler_output']),
#                     torch.nn.functional.normalize(encoded_image['pooler_output'])
#                 ],
#                 dim=1
#             ),mask)
        # print(output.shape)
        logits = torch.mean(output, dim=1)
        
        # print(avg_pool.shape)
        # logits = self.output_layer(avg_pool)
        # print(logits.shape)
        # print("Embedded shape: ", embedded.size())
        # print("After output layer logits: ", logits.size())
        # logits = nn.Softmax(dim=1)(avg_pool)
        # print(logits.shape)
        
        
        # fused_output = self.fusion(
        #     torch.cat(
        #         [
        #             encoded_text['pooler_output'],
        #             encoded_image['pooler_output'],
        #         ],
        #         dim=1
        #     )
        # )
        # logits = self.classifier(logits)
        # print(logits.shape)
        # logits = torch.transpose(logits,0,1)
        # print(logits.shape)
        
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


# In[10]:


def createMultimodalVQACollatorAndModel(model_config):
    
    text = model_config['pretrained_text_name']
    image = model_config['pretrained_image_name']
    
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )
    # # configuration
    # num_heads= 2
    # num_trx_cells= 4
    # dropout=0.1
    # intermediate_dim = 512
    multi_model = MultimodalVQAModel(num_heads=model_config['num_heads'],num_trx_cells=model_config['num_trx_cells'],dropout=model_config['dropout'],
                                     intermediate_dim=model_config['intermediate_dim'], 
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




# labels = np.random.randint(len(answer_space), size=5)
# preds = np.random.randint(len(answer_space), size=5)

# def showAnswers(ids):
#     print([answer_space[id] for id in ids])

# showAnswers(labels)
# showAnswers(preds)

# print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
# print("Labels vs Labels: ", batch_wup_measure(labels, labels))


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


# In[27]:

# {
#     # "train_txt_dbs": ["/txt/vqa_train.db",
#     #                   "/txt/vqa_trainval.db",
#     #                   "/txt/vqa_vg.db"],
#     # "train_img_dbs": ["/img/coco_train2014/", "/img/coco_val2014", "/img/vg/"],
#     # "val_txt_db": "/txt/vqa_devval.db",
#     # "val_img_db": "/img/coco_val2014/",
#     # "checkpoint": "/pretrain/uniter-base.pt",
#     # "model_config": "/src/config/uniter-base.json",
#     # "output_dir": "/storage/vqa/default",
#     "max_txt_len": 60,
#     "conf_th": 0.2,
#     "max_bb": 100,
#     "min_bb": 10,
#     "num_bb": 36,
#     "train_batch_size": 5120,
#     "val_batch_size": 10240,
#     "gradient_accumulation_steps": 5,
#     "learning_rate": 8e-05,
#     "lr_mul": 10.0,
#     "valid_steps": 500,
#     "num_train_steps": 6000,
#     "optim": "adamw",
#     "betas": [
#         0.9,
#         0.98
#     ],
#     "dropout": 0.1,
#     "weight_decay": 0.01,
#     "grad_norm": 2.0,
#     "warmup_steps": 600,
#     "seed": 42,
#     "fp16": true,
#     "n_workers": 4,
#     "pin_mem": true
# }


args = TrainingArguments(
    output_dir="checkpoint",
    seed=12345, 
    warmup_steps=600,
    lr_scheduler_type='cosine_with_restarts', #"cosine_with_restarts",
    learning_rate=5e-5,
    max_grad_norm=10,
    adam_beta1=0.9,
    adam_beta2=0.98,
    evaluation_strategy="steps",
    eval_steps=200,
    # max_steps=1500,
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,             # Save only the last 3 checkpoints at any given time while training 
    metric_for_best_model='wups',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=10,
    fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=1,
    load_best_model_at_end=True,
)

#
model_config = {'num_heads': 12,
            'num_trx_cells':  12,
            'dropout': 0.1,
            'intermediate_dim': 768, #2*768,
            'pretrained_text_name':  'bert-base-uncased',
            'pretrained_image_name':  'google/vit-base-patch16-224-in21k'}

new_ls = list(model_config.values())
config_name = '_'.join(str(e) for e in new_ls)

def createAndTrainModel(dataset, args, model_config, multimodal_model='bert_vit'+config_name):
    
    collator, model = createMultimodalVQACollatorAndModel(model_config = model_config)
    
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


collator, model, train_multi_metrics, eval_multi_metrics,log_history = createAndTrainModel(dataset, args,model_config = model_config)

print(log_history)
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




