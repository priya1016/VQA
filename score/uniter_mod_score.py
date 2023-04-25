import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, set_caching_enabled
import numpy as np
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

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from layer import GELU
from model_uniter_modified import UniterPreTrainedModel, UniterModel, UniterConfig

version = "uniter_2_modified_20_epochs"

# SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
os.environ['HF_HOME'] = os.path.join(".", "cache")
# SET ONLY 1 GPU DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_caching_enabled(True)
logging.set_verbosity_error()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    
    
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

def batch_wup_measure(labels, preds):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)


def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_tuple
    # preds = logits.argmax(axis=-1)
    preds = logits
    # print("labels", labels)
    # print("logits",logits)
    # print("preds",preds)

    return {
        "wups": batch_wup_measure(labels, preds),
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro'),
        "f1_micro": f1_score(labels, preds, average='micro'),
        "precision": precision_score(labels, preds,average='macro',zero_division=0),
        "recall": recall_score(labels, preds,average='macro',zero_division=0),
        # "roc_auc": roc_auc_score(labels, preds),
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
gc.collect()
torch.cuda.empty_cache()

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
    # multi_trainer.save_metrics(split="train",metrics=train_multi_metrics)
    # multi_trainer.save_metrics(split="eval",metrics=eval_multi_metrics)
    
    return collator, model#, train_multi_metrics, eval_multi_metrics,log_history


collator, model = createAndTrainModel(dataset, args=args,config=config,model_config=model_config)

model = MultimodalVQAModel(config=config)

# device ='cpu'
# We use the checkpoint giving best results 

model.load_state_dict(torch.load(os.path.join("..","exp"+str(version), "checkpoint", 'bert_vit'+config_name, "checkpoint-1000", "pytorch_model.bin")))
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

    
    # labels = labels.to(torch.int32).cpu()

# preds
# print(labels)
# print(torch.Tensor(preds.astype(int)).to(torch.int32).to(device))

# preds = torch.Tensor(preds.astype(int)).to(torch.int32)

# print(labels_array)
# print(preds_array)
# test_tuple = (np.array(preds_list).astype(int), np.array(labels_list))
test_tuple = (preds_array,labels_array)
results_unseen = compute_metrics(test_tuple)
pd.DataFrame([results_unseen]).to_csv("../exp"+str(version)+"/results_unseen.csv")
print(results_unseen)
