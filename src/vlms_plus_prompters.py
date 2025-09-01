import torch.nn as nn
from prompters import load_padding_prompter, load_fixed_patch_prompter, load_full_prompter, load_replace_pad_prompter
from utils import logits_to_quality_scores
from vlms import MPlugOwl2
from transformers import AutoTokenizer

class VLMWithVisualPrompt(nn.Module):
    def __init__(self, vlm_name, visual_prompt_type, visual_prompt_args, quality_score_type, use_visual_prompt=True):
        super().__init__() 
        self.vlm_name = vlm_name
        self.visual_prompt_type = visual_prompt_type
        self.visual_prompt_args = visual_prompt_args
        self.quality_score_type = quality_score_type
        self.use_visual_prompt = use_visual_prompt
        self.positive_token_ids, self.negative_token_ids = self._cache_quality_token_indices()
        
        self.vlm = self.load_vlm()
        self.image_size = self.vlm.image_size
        
        self.freeze_vlm()
        self.vlm.eval()

        if self.visual_prompt_type is not None and self.use_visual_prompt:
            self.visual_prompter = self.load_visual_prompter()
        else:
            self.visual_prompter = None

        self.vlm.set_visual_prompter(self.visual_prompter)

    def forward(self, pixel_values, **ignored):

        images = pixel_values

        # The visual prompter is applied in the forward function of the VLM (internally)
        output_logits = self.vlm(pixel_values=images)

        quality_scores = logits_to_quality_scores(output_logits, 
                                                 self.positive_token_ids, 
                                                 self.negative_token_ids)        
        return quality_scores
    
    def load_vlm(self):
        # Keep a minimal scaffold for future extensions
        return MPlugOwl2()
        
    def load_visual_prompter(self):
        if self.visual_prompt_type == "padding":
            return load_padding_prompter(self.visual_prompt_args, self.image_size)
        elif self.visual_prompt_type == "fixed_patch":
            return load_fixed_patch_prompter(self.visual_prompt_args)
        elif self.visual_prompt_type == "full":
            return load_full_prompter(self.visual_prompt_args, self.image_size)
        elif self.visual_prompt_type == "replace_pad":
            return load_replace_pad_prompter(self.visual_prompt_args, self.image_size)
        else:
            raise ValueError(f"Invalid visual prompt type: {self.visual_prompt_type}")

    def _cache_quality_token_indices(self):

        _COMPARISONS_DEFINITION = {
            "good_vs_bad": ([" good"], [" bad"]),
            "good_vs_poor": ([" good"], [" poor"]),
            "fine_vs_bad":  ([" fine"], [" bad"]),
            "high_vs_low":  ([" high"], [" low"]),
            "good+high_vs_poor+low": ([" good", " high"], [" poor", " low"]),
            "good+fine_vs_poor+bad": ([" good", " fine"], [" poor", " bad"]),
            "good+high+fine_vs_poor+low+bad": ([" good", " high", " fine"], [" poor", " low", " bad"]),
        }
        
        # mPLUG-Owl2 tokenizer only
        tokenizer = AutoTokenizer.from_pretrained("MAGAer13/mplug-owl2-llama2-7b")

        pos_tokens, neg_tokens = _COMPARISONS_DEFINITION[self.quality_score_type]

        positive_token_ids = [tokenizer.encode(t, add_special_tokens=False)[1] for t in pos_tokens]
        negative_token_ids = [tokenizer.encode(t, add_special_tokens=False)[1] for t in neg_tokens]

        print(f"[Sanity Check] Positive tokens decoded: {tokenizer.decode(positive_token_ids)}")
        print(f"[Sanity Check] Negative tokens decoded: {tokenizer.decode(negative_token_ids)}")

        return positive_token_ids, negative_token_ids

    def freeze_vlm(self):
        for param in self.vlm.parameters():
            param.requires_grad = False
   
    # Print the model
    def __str__(self):
        return f"VLMWithVisualPrompt(vlm_name={self.vlm_name}, visual_prompt_type={self.visual_prompt_type}, use_visual_prompt={self.use_visual_prompt}, quality_score_type={self.quality_score_type})"