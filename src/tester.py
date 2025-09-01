from transformers import Trainer, TrainingArguments
import torch.nn as nn
from torchvision import transforms
from utils import load_yaml_config, compute_metrics, verify_trainable_parameters
from vlms_plus_prompters import VLMWithVisualPrompt
from datasets import IQADataset
from collators import IQACollator
from transforms import get_val_test_transform
import os
import torch


def main():

    # Load config
    config_path = "configs/final_mplug_owl2_configs/SGD_mplug2_exp_01_kadid_padding_10px_add.yaml"
    config = load_yaml_config(config_path)

    # Load custom model (VLM + Visual Prompt)
    my_model = VLMWithVisualPrompt(
        vlm_name=config["model"]["vlm_name"],
        visual_prompt_type=config["model"]["visual_prompt"]["type"],
        visual_prompt_args=config["model"]["visual_prompt"]["args"],
        quality_score_type=config["quality_score_type"],
    )

    # Load the saved visual prompt if present
    checkpoint_best = "checkpoint-best" # example: "checkpoint-1000"
    visual_prompt_path = os.path.join(config["training"]["output_dir"], checkpoint_best, "visual_prompt.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(visual_prompt_path) and hasattr(my_model, "visual_prompter") and my_model.visual_prompter is not None:
        state_dict = torch.load(visual_prompt_path, map_location=device)
        my_model.visual_prompter.load_state_dict(state_dict)
        my_model.visual_prompter.to(device)
        print(f"✅ Loaded visual prompt from {visual_prompt_path}")
    else:
        print(f"⚠️ Visual prompt checkpoint not found at: {visual_prompt_path}")

    # Verify parameter freezing is working correctly
    model_name_for_verification = config["model"].get("vlm_name", "mplug2")
    verify_trainable_parameters(my_model, f"{model_name_for_verification.upper()} with Visual Prompt")

    # Define TrainingArguments for testing
    my_training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        seed=config["training"]["seed"],
        fp16=True,
    )

    # Get model-specific transform for testing (no augmentation)
    my_transform = get_val_test_transform()

    # Define a test dataset
    my_test_dataset = IQADataset(
        root=config["dataset"]["test"]["path"],
        dataset=config["dataset"]["test"]["name"],
        split=config["dataset"]["test"]["split"],
        transform=my_transform
    )

    # Define a data collator which is used to customly form a batch from a list of examples
    my_data_collator = IQACollator()

    # Define a custom trainer class
    class MyTrainer(Trainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            Perform a prediction step and return predictions and labels for metric computation.
            """
            inputs = self._prepare_inputs(inputs)
            
            with torch.no_grad():
                pixel_values, labels = inputs["pixel_values"], inputs["labels"]
                predictions = model(pixel_values)
                
                if prediction_loss_only:
                    return (None, None, None)
                
                # Return predictions and labels as torch tensors
                return (None, predictions.detach(), labels.detach())

    # Define a trainer
    trainer = MyTrainer(
        model=my_model,
        args=my_training_args,
        data_collator=my_data_collator,
        eval_dataset=my_test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    metrics = trainer.evaluate()

    print(metrics)

if __name__ == "__main__":
    main() 