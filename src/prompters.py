### This code has been copy pasted from: https://github.com/hjbahng/visual_prompting
### I have made some changes to it, but the overall structure is the same.
### Thanks to the original author for the code for their great work. 

import torch
import torch.nn as nn


def load_padding_prompter(args, image_size):
    return PadPrompter(args, image_size)

def load_fixed_patch_prompter(args):
    return FixedPatchPrompter(args)

def load_full_prompter(args, image_size):
    return FullPrompter(args, image_size)

def load_replace_pad_prompter(args, image_size):
    return ReplacePadPrompter(args, image_size)

class PadPrompter(nn.Module):
    def __init__(self, args, image_size):
        super(PadPrompter, self).__init__()
        
        self.image_size = image_size
        self.pad_size = args["size"]
        self.delta = args["delta"]
        self.mode = args["mode"]
        self.base_size = image_size - self.pad_size*2
       
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]) * 0.01) #<--  # to keep the random values in a small range
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]) * 0.01)
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]) * 0.01)
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - self.pad_size*2, self.pad_size]) * 0.01)


    def forward(self, x):
        device = x.device

        # delta is a scalar that controls the range of the values in the prompt

        base = torch.zeros(1, 3, self.base_size, self.base_size, device=device, dtype=x.dtype)
        up   = self.delta * torch.tanh(self.pad_up) # tanh will constrain the values to be between -1 and 1 on its own
        down = self.delta * torch.tanh(self.pad_down)
        left = self.delta * torch.tanh(self.pad_left)
        right= self.delta * torch.tanh(self.pad_right)


        top_bot    = torch.cat([left, base, right], dim=3)
        prompt     = torch.cat([up, top_bot, down], dim=2)

        prompt = torch.cat(x.size(0) * [prompt])

        if self.mode == "add":
            composite = x + prompt
        elif self.mode == "mul":
            composite = x * prompt
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return composite.clamp(0,1) 
    
        # We do clamp just to keep all the values inside 0,1. 
        # The rational is that we want to feed a normal looking image to the MLLM (not that
        # the other method didn't work (i mean before clamping and tanh), it did,
        # but intuitivly didn't sit well with me, and also was very sensible to low LRs.
        # and batch size variations. Now im able to train on b_size of 16 with a lowish or large learning rate
        # and still have stable consistent results.

        # If values of prompter pixels are beyond 0,1 then I intuitivly think that
        # they don't mean anything in the sense of an image.
        # the problem I had with the large learning rate, will eventually
        # be mitigated with tanh (will remap every pixel value down to -1 or 1)
        # and with clamp (which eventually you'd need cuz tanh can output -1).


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.psize      = args["size"]
        self.is_center  = args["is_center"]
        self.delta      = args["delta"]
        self.mode       = args["mode"]

        # small random start to keep tanh in linear zone
        self.patch = nn.Parameter(torch.randn(1, 3, self.psize, self.psize) * 0.01)

    def forward(self, x):
        B, _, H, W = x.shape
        device     = x.device

        # Decide where to place the patch
        if self.is_center:
            center = H // 2
            top    = center - self.psize // 2
            left   = center - self.psize // 2
        else:
            top  = 0
            left = 0

        # Build the prompt canvas
        prompt = torch.zeros(1, 3, H, W, device=device)
        prompt[:, :, top:top+self.psize, left:left+self.psize] = \
            self.delta * torch.tanh(self.patch)

        # Broadcast the prompt to the full batch
        prompt = prompt.repeat(B, 1, 1, 1)

        if self.mode == "add":
            composite = x + prompt
        elif self.mode == "mul":
            composite = x * prompt
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return composite.clamp(0, 1)

# This is a prompter that has the same size as the image.
class FullPrompter(nn.Module):
    def __init__(self, args, image_size):
        super().__init__()
        self.image_size = image_size
        self.delta = args["delta"]
        self.mode = args["mode"]

        self.prompt = nn.Parameter(torch.randn(1, 3, self.image_size, self.image_size) * 0.01)

    def forward(self, x):

        # apply tanh to the prompt
        prompt = self.delta * torch.tanh(self.prompt)
        # repeat the prompt to the same size as the image
            # If batch size is 6 for example, the prompt will be repeated 6 times
        prompt = prompt.repeat(x.size(0), 1, 1, 1)
        if self.mode == "add":
            composite = x + prompt
        elif self.mode == "mul":
            composite = x * prompt
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return composite.clamp(0, 1)

# Replace Pad because it doesn't add the visual prompt and the image, but it puts the image at the center, and pads the rest
# Concretely, it removes parts of the image and replaces them with the visual prompt
class ReplacePadPrompter(nn.Module):
    """
    Visual prompter that replaces padding areas with learned visual prompts.
    Instead of adding/multiplying prompts, it zeroes out the padding areas
    and places the visual prompt there, preserving the center image content.
    """
    def __init__(self, args, image_size):
        super(ReplacePadPrompter, self).__init__()
        
        self.image_size = image_size
        self.pad_size = args["size"]
        self.delta = args["delta"]
        self.base_size = image_size - self.pad_size * 2
        
        # Learnable visual prompts for each padding area
        self.pad_up = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]) * 0.01)
        self.pad_down = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]) * 0.01)
        self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_size, self.pad_size]) * 0.01)
        self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_size, self.pad_size]) * 0.01)

    def forward(self, x):
        device = x.device
        B, _, H, W = x.shape
        
        # Create the visual prompt by combining all padding areas
        up = self.delta * torch.tanh(self.pad_up)
        down = self.delta * torch.tanh(self.pad_down)
        left = self.delta * torch.tanh(self.pad_left)
        right = self.delta * torch.tanh(self.pad_right)
        
        # Combine padding areas into a full-size prompt
        top_bot = torch.cat([left, torch.zeros(1, 3, self.base_size, self.base_size, device=device, dtype=x.dtype), right], dim=3)
        prompt = torch.cat([up, top_bot, down], dim=2)
        
        # Broadcast to batch size
        prompt = prompt.repeat(B, 1, 1, 1)
        
        # Create mask for padding areas (1 for padding, 0 for center content)
        mask = torch.ones_like(x)
        mask[:, :, self.pad_size:self.pad_size + self.base_size, self.pad_size:self.pad_size + self.base_size] = 0
        
        # Zero out padding areas in original image and add visual prompt
        zeroed_image = x * (1 - mask)  # Keep center content, zero padding
        prompted_padding = prompt * mask  # Apply prompt only to padding areas
        
        # Combine: center content + visual prompt in padding
        composite = zeroed_image + prompted_padding
        
        return composite.clamp(0, 1)