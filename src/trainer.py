from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import load_yaml_config, compute_metrics, verify_trainable_parameters
from vlms_plus_prompters import VLMWithVisualPrompt
from datasets import IQADataset
from collators import IQACollator
from transforms import get_train_transform, get_val_test_transform
import os
import torch
import argparse


def main(experiment_num):

    # 0. Load config

    final_mplug_owl2_exps = [
        # Model mPLUG-Owl2
        # Operation: Addition (add)
            # Padding -- 10px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_01_kadid_padding_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_02_koniq_padding_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_03_agiqa_padding_10px_add.yaml",
        
            # Padding -- 30px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_04_kadid_padding_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_05_koniq_padding_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_06_agiqa_padding_30px_add.yaml",
        
            # Center Patch -- 10px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_07_kadid_patch_center_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_08_koniq_patch_center_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_09_agiqa_patch_center_10px_add.yaml",
        
            # Center Patch -- 30px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_10_kadid_patch_center_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_11_koniq_patch_center_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_12_agiqa_patch_center_30px_add.yaml",
        
            # Top Left Patch -- 10px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_13_kadid_patch_topLeft_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_14_koniq_patch_topLeft_10px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_15_agiqa_patch_topLeft_10px_add.yaml",
        
            # Top Left Patch -- 30px
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_16_kadid_patch_topLeft_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_17_koniq_patch_topLeft_30px_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_18_agiqa_patch_topLeft_30px_add.yaml",
        
            # Full Overlay
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_19_kadid_fullOverlay_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_20_koniq_fullOverlay_add.yaml",
        "configs/final_mplug_owl2_configs/SGD_mplug2_exp_21_agiqa_fullOverlay_add.yaml",
    ]


    config = load_yaml_config(final_mplug_owl2_exps[experiment_num])

    print("✅ Running experiment NUM: ", experiment_num)
    print("✅ Running experiment: ", config["experiment_name"])

    # 1. Load custom model (VLM + Visual Prompt)
    my_model = VLMWithVisualPrompt(
        vlm_name=config["model"]["vlm_name"],
        visual_prompt_type=config["model"]["visual_prompt"]["type"],
        visual_prompt_args=config["model"]["visual_prompt"]["args"],
        quality_score_type=config["quality_score_type"],
    )

    # Verify parameter freezing is working correctly
    model_name_for_verification = config["model"].get("vlm_name", "mplug2")
    verify_trainable_parameters(my_model, f"{model_name_for_verification.upper()} with Visual Prompt")

    # 2. Define training arguments
    my_training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        optim=config["training"]["optim"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        save_strategy=config["training"]["save_strategy"],
        remove_unused_columns=False,
        fp16=True,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model="metric",
        greater_is_better=config["training"]["greater_is_better"],
        seed=config["training"]["seed"],
    )
    
    # Transforms
    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()

    # 3. Define a train dataset
    my_train_dataset = IQADataset(
        root=config["dataset"]["train"]["path"],
        dataset=config["dataset"]["train"]["name"],
        split=config["dataset"]["train"]["split"],
        transform=train_transform
    )

    # 4. Define a eval dataset
    my_eval_dataset = IQADataset(
        root=config["dataset"]["eval"]["path"],
        dataset=config["dataset"]["eval"]["name"],
        split=config["dataset"]["eval"]["split"],
        transform=val_test_transform
    )


    # 5. Define a data collator which is used to customly form a batch from a list of examples
    my_data_collator = IQACollator()

    # 6. Define a custom trainer class
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, **kwargs):
            pixel_values, labels = inputs["pixel_values"], inputs["labels"]
            predictions = model(pixel_values)
            loss_func = nn.MSELoss()
            loss = loss_func(predictions.to(labels.device), labels)
            return loss
        
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
        
        # Save visual prompt and regression head for VMs, only visual prompt for VLMs
        def save_model(self, output_dir=None, _internal_call=False):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            if hasattr(self.model, "visual_prompter") and self.model.visual_prompter is not None:
                prompt_state = self.model.visual_prompter.state_dict()
                torch.save(prompt_state, os.path.join(output_dir, "visual_prompt.pth"))


    ## Special for resuming training from checkpoint (Example)
    # visual_prompt_ckpt = "final_mplug_owl2_exps/exp-01-kadid-padding-10px-add/checkpoint-1000/visual_prompt.pth"
    # prompt_state = torch.load(visual_prompt_ckpt, map_location="cpu")
    # my_model.visual_prompter.load_state_dict(prompt_state)

    # 7. Define a trainer
    trainer = MyTrainer(
        model=my_model,
        args=my_training_args,
        train_dataset=my_train_dataset,
        eval_dataset=my_eval_dataset,
        data_collator=my_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )

    # 8. Train the model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_num', type=int, help='Experiment number')
    args = parser.parse_args()
    main(args.experiment_num)