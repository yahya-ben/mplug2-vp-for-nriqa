import yaml
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
import torchvision.transforms as T
import os
from torchvision.utils import save_image
import uuid


# This function receives a tensor of logits in the form (B,L,V): (batch_size, sequence_length:
# which is the number of tokens in the sequence (input + output), vocab_size: which is the number of classes)
# we select the last token logits and then extract some particular token logits.
def logits_to_quality_scores(logits, positive_token_ids, negative_token_ids):

    # Logits comes in the form (B,L,V), we select the last token logits (this can contain the logits of quality related tokens)
    last_token_logits = logits[:, -1, :] # (B: batch_size, V: vocab_size)

    # Extract the logits of the quality related tokens
    positive_logits = last_token_logits[:, positive_token_ids]
    negative_logits = last_token_logits[:, negative_token_ids]

    exp_positive = torch.exp(positive_logits)  # (B, P)
    exp_negative = torch.exp(negative_logits)  # (B, N)

    sum_positive = exp_positive.sum(dim=1)     # (B,)
    sum_negative = exp_negative.sum(dim=1)     # (B,)

    q_score = sum_positive / (sum_positive + sum_negative)  # (B,)

    return q_score


def load_yaml_config(file_path: str) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration.
              Returns an empty dictionary if the file is not found or an error occurs.
    """
    config = {}
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file case
            config = {}
            print(f"Warning: Config file at {file_path} is empty. Returning empty config.")
        else:
            print(f"Successfully loaded configuration from: {file_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {file_path}. Returning empty config.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file at {file_path}: {e}. Returning empty config.")
    except Exception as e:
        print(f"An unexpected error occurred while reading {file_path}: {e}. Returning empty config.")
    return config

# Returns srcc, plcc, eval_metric

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    srocc, _ = spearmanr(predictions, labels)
    plcc, _ = pearsonr(predictions, labels)
    metric = (srocc + plcc) / 2
    return {
        "srocc": srocc,
        "plcc": plcc,
        "metric": metric
    }

# The minimal generic PyTroch transforms (inspired by CLIP processing)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def apply_generic_transforms(images, image_size):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        _convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])(images)


# Utilities for saving prompted images

def save_prompted_images(image_tensor, output_dir="prompted_images"):
    """
    Save each image in the batch to the specified directory.
    
    Args:
    - image_tensor (torch.Tensor): The batch of images to save.
    - output_dir (str): The directory where images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(image_tensor.size(0)):
        # Generate a unique filename for each image
        filename = f"{uuid.uuid4().hex}.png"
        # Save the image
        save_image(image_tensor[i].cpu(), os.path.join(output_dir, filename))


def verify_trainable_parameters(model, model_name="Model"):
    """
    Verify which parameters are trainable in the model.
    This is useful for debugging freezing issues in visual prompt tuning.
    
    Args:
        model: The model to check
        model_name: Name of the model for logging
    
    Returns:
        dict: Dictionary with parameter statistics
    """
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    print(f"\n{'='*50}")
    print(f"Parameter Analysis for {model_name}")
    print(f"{'='*50}")
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✅ TRAINABLE: {name} | Shape: {list(param.shape)} | Params: {param.numel():,}")
        else:
            frozen_params += param.numel()
            # Only print first few frozen parameters to avoid spam
            if frozen_params < 5 * param.numel():  # Only show first few
                print(f"❄️  FROZEN: {name} | Shape: {list(param.shape)} | Params: {param.numel():,}")
    
    print(f"\n📊 Parameter Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"   Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    print(f"{'='*50}\n")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percentage': 100*trainable_params/total_params
    }