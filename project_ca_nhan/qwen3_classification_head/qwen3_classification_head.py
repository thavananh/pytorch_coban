import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import os
import unsloth
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F


MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
LR = 4e-5
BATCH_SIZE = 8
NUM_EPOCHS = 2
MAX_LEN = 1024
TEST_SIZE = 0.2
TASK_TYPE = "aspect_cateogry_analysis"
DETECT_METHOD = "sigmoid_prob"
BASE_MODEL_NAME = MODEL_NAME.replace("/", "_")
LORA_FULL_PATH = "/kaggle/input/finetuned_qwen3_14b_uit_vsfc_absa_15062025/other/default/1/opensloth_lora"
FOLDER_DIR = f'{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}'
os.makedirs(FOLDER_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(FOLDER_DIR, f'best_{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}.pth')
ACCURACY_PLOT_NAME = os.path.join(FOLDER_DIR, f'accuracy_plot_{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}.png')
LOSS_PLOT_NAME = os.path.join(FOLDER_DIR, f'loss_plot_{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}.png')
CLASSIFICATION_REPORT_NAME = os.path.join(FOLDER_DIR, f'classification_report_{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}.txt')
CONFUSION_MATRIX_NAME = os.path.join(FOLDER_DIR, f'confusion_matrix_{BASE_MODEL_NAME}_{TASK_TYPE}_{DETECT_METHOD}.png')


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        logging.info(f"Using {num_gpus} GPUs.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    return {
        "device": device,
        "num_gpus": num_gpus if 'num_gpus' in locals() else 0
    }


import torch
import torch.nn as nn

class LLMForClassification(nn.Module):
    def __init__(self, base_model, lora_path, hidden_size=4096, num_classes=3):
        super().__init__()
        
        # Load base + LoRA
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=1024,
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)

        for param in self.model.parameters():
            param.requires_grad = False
        
        hidden_size = self.model.config.hidden_size if hasattr(self.model.config, "hidden_size") else hidden_size
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

    