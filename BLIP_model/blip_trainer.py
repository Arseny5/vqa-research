# === install and import required dependencies ================================

import os
import warnings
import json
import numpy as np
import pandas as pd
import evaluate
import requests
import torch
import random
import argparse
import torchvision
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.io import read_image
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipConfig


os.environ["WANDB_PROJECT"]="vqa_binary_research"
parser = argparse.ArgumentParser(description="model parameters")
parser.add_argument("-od", "--output-dir", type=bool, default=True, help="overwrite output dir")
parser.add_argument("-estr", "--eval-strat", type=str, default='steps', help="evaluation strategy")
parser.add_argument("-estep", "--eval-steps", type=int, default=500, help="evaluation steps")
parser.add_argument("-lstr", "--logging-strat", type=str, default='steps', help="logging strategy")
parser.add_argument("-lstep", "--logging-steps", type=int, default=250, help="logging steps")
parser.add_argument("-sstr", "--save-strat", type=str, default='steps', help="save strategy")
parser.add_argument("-sstep", "--save-steps", type=int, default=500, help="save steps")
parser.add_argument("-stl", "--save-total-limit", type=int, default=2, help="save total limit")
parser.add_argument("-lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("-bs", "--batch-size", type=int, default=8, help="batch size on train")
parser.add_argument("-nte", "--num-train-epochs", type=float, default=10.0, help="num train epochs")
parser.add_argument("-lbmae", "--load-best-model-at-end", type=bool, default=True, help="load best model at end")
parser.add_argument("-rt", "--report", type=str, default='wandb', help="report results to")
parser.add_argument("-esp", "--early-stopping-patience", type=int, default=3, help="early stopping patience")
parser.add_argument("-s", "--seed", type=int, default=42, help="random state")
parser.add_argument("-nd", "--num-device", type=int, default=0, help="index of device")
parser.add_argument("-expn", "--name-experiment", type=str, default='blip_vqa_model', help="name of experiment")


args = parser.parse_args()
OVERWRITE_OUTPUT_DIR = args.output_dir
EVALUATION_STRATEGY = args.eval_strat # Evaluation strategy for training: evaluate every eval_steps
EVAL_STEPS = args.eval_steps # Number of update steps between two evaluations
LOGGING_STRATEGY = args.logging_strat
LOGGING_STEPS = args.logging_steps
SAVE_STRATEGY = args.save_strat
SAVE_STEPS = args.save_steps
SAVE_TOTAL_LIMIT = args.save_total_limit # Save only the last 3 checkpoints at any given time while training
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
NUM_TRAIN_EPOCHS = args.num_train_epochs
LOAD_BEST_MODEL_AT_END = args.load_best_model_at_end # Loads the best model based on the evaluation metric at the end of training
REPORT_TO = args.report
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
SEED = args.seed
RUN_NAME = args.name_experiment
device = torch.device("cuda:{}".format(args.num_device) if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

warnings.filterwarnings('ignore')
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



# === load data ================================

DATA_PATH = '/home/jovyan/ars/vqa-research/VQA/data/'
ANNOTATIONS_PATH = DATA_PATH + 'abstract_v002_train2017_annotations.json'
QUESTIONS_PATH = DATA_PATH + 'OpenEnded_abstract_v002_train2017_questions.json'
VQA_PATH = DATA_PATH + 'VQA_train.csv'
IMAGE_DIR = DATA_PATH + 'scene_img_abstract_v002_train2017/'
IMAGE_PREFIX = 'abstract_v002_train2015_'
IMAGE_FORMAT = '.png'

BLIP_MODEL_PRETRAIN = 'Salesforce/blip-vqa-base' # get model from HF vilt-32b
MODELS_DIR = '/home/jovyan/ars/vqa-research/VQA/BLIP_model/model_blip_checkpoints'

dataset = pd.read_csv(VQA_PATH, index_col=0)

with open(ANNOTATIONS_PATH) as f:
    annotations = json.load(f)['annotations']

with open(QUESTIONS_PATH) as f:
    questions = json.load(f)['questions']

# lemmatize and lowercase with questions and answers
dataset[['question', 'answer']].apply(lambda x: x.str.lower(), axis=0)

# we should fix image paths for correct version
dataset['image_path'] = IMAGE_PREFIX + (12 - dataset['image_id'].astype('str').str.len()).apply(lambda x: x * '0') + dataset['image_id'].astype('str') + IMAGE_FORMAT


# === preprocess data ================================

config = BlipConfig.from_pretrained(BLIP_MODEL_PRETRAIN)
dataset['label'] = dataset['answer'].apply(lambda x: 1 if x == 'yes' else 0)

class BlipDataset(torch.utils.data.Dataset):

    def __init__(self, data, processor, img_dir, max_length=32):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.loc[idx, ['question', 'answer', 'image_path', 'label']]

        img = read_image(self.img_dir + str(item['image_path']))[:3,:,:]

        encoding = self.processor(
            img,
            item['question'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )

        encoding['label'] = self.processor.tokenizer.encode(
            item['answer'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )

        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding
    
processor = BlipProcessor.from_pretrained(BLIP_MODEL_PRETRAIN)
blip_ds = BlipDataset(dataset, processor, IMAGE_DIR)

count_samples_train = int(np.round((dataset.shape[0] * 80 ) / 100))
count_samples_val = int((dataset.shape[0] - count_samples_train) / 2)
count_samples_test = int(dataset.shape[0] - count_samples_train - count_samples_val)
blip_train_ds, blip_val_ds, blip_test_ds = torch.utils.data.random_split(blip_ds, [count_samples_train, 
                                                                                   count_samples_val, 
                                                                                   count_samples_test])

model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL_PRETRAIN, id2label=config.id2label, label2id=config.label2id)
model.to(device)

args = TrainingArguments(
    output_dir=MODELS_DIR,
    overwrite_output_dir=OVERWRITE_OUTPUT_DIR,
    evaluation_strategy=EVALUATION_STRATEGY,
    eval_steps=EVAL_STEPS,
    logging_strategy=LOGGING_STRATEGY,
    logging_steps=LOGGING_STEPS,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=SAVE_TOTAL_LIMIT,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    report_to=REPORT_TO,
)

collator = DataCollatorWithPadding(processor.tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=blip_train_ds,
    eval_dataset=blip_val_ds,
    data_collator=blip_test_ds,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)


trainer.train()


# === inference and evaluate the model ================================

finetunned_model = BlipForQuestionAnswering.from_pretrained(
    MODELS_DIR + "/checkpoint-9500",
    id2label=config.id2label,
    label2id=config.label2id 
).to(device)

processor = BlipProcessor.from_pretrained(BLIP_MODEL_PRETRAIN)
collator = DataCollatorWithPadding(processor.tokenizer)

blip_test_dl = DataLoader(
    blip_test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)

def calculate_metric_on_test_ds(model, processor, test_dl, metric, device):
    for batch in tqdm(test_dl):
        batch = batch.to(device)
        labels = batch['labels']
        batch.pop('labels', None)

        input_ids = batch.input_ids
        pixel_values = batch.pixel_values

        preds = []
        refs = []
        for i in range(8):
            try:
                input_ids_i = input_ids[i].unsqueeze(0)
                pixel_values_i = pixel_values[i].unsqueeze(0)

                output = finetunned_blip_model.generate(input_ids=input_ids_i,
                                                        pixel_values=pixel_values_i,
                                                        labels=labels)

                pred = blip_processor.decode(output[0], skip_special_tokens=True)
                ref = blip_processor.decode(labels[i], skip_special_tokens=True)
                preds.append(pred)
                refs.append(ref)
            except:
                break

        metric.add_batch(predictions=preds, references=refs)
    return metric.compute()

rouge_metric = evaluate.load('rouge')

rouge_score = calculate_metric_on_test_ds(finetunned_model, processor, blip_test_dl, rouge_metric, device)
print(f'ROUGE: {rouge_score}')