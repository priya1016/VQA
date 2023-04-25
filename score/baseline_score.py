#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
from PIL import Image
import torch,gc
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

# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, roc_auc_score, cohen_kappa_score
import pandas as pd

# In[2]:
version = "xx_baseline"

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

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

# In[ ]:


# from IPython.display import display

# def showExample(train=True, id=None):
#     if train:
#         data = dataset["train"]
#     else:
#         data = dataset["test"]
#     if id == None:
#         id = np.random.randint(len(data))
#     image = Image.open(os.path.join("..", "dataset", "images", data[id]["image_id"] + ".png"))
#     display(image)

#     print("Question:\t", data[id]["question"])
#     print("Answer:\t\t", data[id]["answer"], "(Label: {0})".format(data[id]["label"]))


# showExample()
# In[5]:


@dataclass
class MultimodalCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def tokenize_text(self, texts: List[str]):
        encoded_text = self.tokenizer(
            text=texts,
            padding='longest',
            max_length=24,
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


# In[20]:


class MultimodalVQAModel(nn.Module):
    def __init__(
            self,
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
        self.fusion = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size + self.image_encoder.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
            self,
            input_ids: torch.LongTensor,
            pixel_values: torch.FloatTensor,
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
        # print(encoded_text['pooler_output'].shape)
        # print(encoded_image['pooler_output'].shape)
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text['pooler_output'],
                    encoded_image['pooler_output'],
                ],
                dim=1
            )
        )
        # print(fused_output.shape)
        logits = self.classifier(fused_output)
        # print(logits.shape)
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


# In[7]:


def createMultimodalVQACollatorAndModel(text='bert-base-uncased', image='google/vit-base-patch16-224-in21k'):
    tokenizer = AutoTokenizer.from_pretrained(text)
    preprocessor = AutoFeatureExtractor.from_pretrained(image)

    multi_collator = MultimodalCollator(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
    )


    multi_model = MultimodalVQAModel(pretrained_text_name=text, pretrained_image_name=image).to(device)
    
    return multi_collator, multi_model


# In[8]:


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


# In[9]:


def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)


# In[10]:


import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# In[11]:


# labels = np.random.randint(len(answer_space), size=5)
# preds = np.random.randint(len(answer_space), size=5)

# def showAnswers(ids):
#     print([answer_space[id] for id in ids])

# showAnswers(labels)
# showAnswers(preds)

# print("Predictions vs Labels: ", batch_wup_measure(labels, preds))
# print("Labels vs Labels: ", batch_wup_measure(labels, labels))


# In[12]:


def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    # preds = logits.argmax(axis=-1)
    preds = logits
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


# In[13]:


args = TrainingArguments(
    output_dir="checkpoint",
    seed=12345, 
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,             # Save only the last 3 checkpoints at any given time while training 
    metric_for_best_model='wups',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    num_train_epochs=5,
    fp16=True,
    # warmup_ratio=0.01,
    # learning_rate=5e-4,
    # weight_decay=1e-4,
    # gradient_accumulation_steps=2,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
)


# In[14]:


def createAndTrainModel(dataset, args, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224-in21k', multimodal_model='bert_vit'):
    collator, model = createMultimodalVQACollatorAndModel(text_model, image_model)
    
#     multi_args = deepcopy(args)
#     multi_args.output_dir = os.path.join("..", "exp" + str(version) , "checkpoint", multimodal_model)
#     multi_trainer = Trainer(
#         model,
#         multi_args,
#         train_dataset=dataset['train'],
#         eval_dataset=dataset['test'],
#         data_collator=collator,
#         compute_metrics=compute_metrics
#     )
    
#     train_multi_metrics = multi_trainer.train()
#     eval_multi_metrics = multi_trainer.evaluate()
    
#     log_history = multi_trainer.state.log_history
    
    return collator, model#, train_multi_metrics, eval_multi_metrics,log_history


# In[16]:


small_dataset = {}
small_dataset['train'] = dataset['train'].select(range(1000))
small_dataset['test'] = dataset['test'].select(range(100))
small_dataset


gc.collect()
torch.cuda.empty_cache()
# In[21]:


collator, model= createAndTrainModel(dataset, args)


model = MultimodalVQAModel()

# # We use the checkpoint giving best results
model.load_state_dict(torch.load(os.path.join("..", "exp" + str(version) , "checkpoint", "bert_vit", "checkpoint-2100", "pytorch_model.bin")))
model.to(device) 

step = 2 #number of examples in loop

num_samples = len(dataset["unseen"])

for i in range(0,num_samples, step):
    # print("*****Running ",i,"th iteration*****")
    sample = collator(dataset["unseen"][i:i+step])

    input_ids = sample["input_ids"].to(device)
    token_type_ids = sample["token_type_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    pixel_values = sample["pixel_values"].to(device)
    labels = sample["labels"].to(device)

    # print("input_ids",input_ids)
    # print("token_type_ids",token_type_ids)
    model.eval()
    output = model(input_ids =input_ids, pixel_values=pixel_values, labels=labels)

    preds = output["logits"].argmax(axis=-1).cpu().numpy()
    
    if i == 0:
        labels_array = labels.cpu().numpy()
        preds_array = preds.astype(int)
    else:
        labels_array = np.concatenate((labels_array,labels.cpu().numpy()),axis=0)
        preds_array = np.concatenate((preds_array.astype(int),preds),axis=0)

    
test_tuple = (preds_array,labels_array)
results_unseen = compute_metrics(test_tuple)

pd.DataFrame([results_unseen]).to_csv("../exp"+str(version)+"/results_unseen.csv")
print(results_unseen)




