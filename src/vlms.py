import torch
import torch.nn as nn


class MPlugOwl2(nn.Module):
    def __init__(self):
        super(MPlugOwl2, self).__init__()

        from mplug_owl2.model.builder import load_pretrained_model
        from mplug_owl2.mm_utils import get_model_name_from_path
        from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from mplug_owl2.conversation import conv_templates
        

        self.model_id = "MAGAer13/mplug-owl2-llama2-7b"
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX   = IMAGE_TOKEN_INDEX
        self.conv_template       = conv_templates["mplug_owl2"]

        self.model_name = get_model_name_from_path(self.model_id)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_id, None, self.model_name,
            load_8bit=False, load_4bit=False,
            torch_dtype=torch.float16
        )

        self.image_size = 448

    def set_visual_prompter(self, visual_prompter):
        self.visual_prompter = visual_prompter

    def forward(self, pixel_values):
        
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token
        from torchvision.transforms.functional import to_pil_image
        from transforms import normalize_batch_after_prompt

        batch_size = pixel_values.size(0)
        
        input_id_list = []
        for _ in range(batch_size):
            conv = self.conv_template.copy()
            conv.append_message(
                conv.roles[0],
                self.DEFAULT_IMAGE_TOKEN + "Rate the technical quality of the image."
            )
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                self.IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            )
            input_id_list.append(ids.squeeze(0))

        input_ids = torch.stack(input_id_list).to(self.model.device)

        image_tensor = self.visual_prompter(pixel_values) if hasattr(self, "visual_prompter") and self.visual_prompter is not None else pixel_values
        
        # Apply normalization AFTER prompting to ensure consistent preprocessing
        image_tensor = normalize_batch_after_prompt(image_tensor)

        outputs = self.model(
            input_ids=input_ids,
            images=image_tensor,
            use_cache=False             
        )
        
        return outputs.logits