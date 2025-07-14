# PixelPrompt: Parameter-Efficient NR-IQA via Pixel-Space Visual Prompts in Multimodal LLMs

This repository implements **PixelPrompt**, a novel parameter-efficient adaptation method for No-Reference Image Quality Assessment (NR-IQA) using visual prompts optimized in pixel-space. Unlike full fine-tuning of Multimodal Large Language Models (MLLMs), our approach optimizes a negligible number of learnable parameters while keeping the base MLLM entirely frozen.

## 🔥 Key Features

- **Parameter-Efficient**: Only 52K trainable parameters vs 7B+ for full fine-tuning
- **Competitive Performance**: Achieves 0.91 SROCC on KADID-10k dataset
- **Multiple Visual Prompt Types**: Padding, Fixed Patches (Center/Top-Left), Full Overlay
- **Multiple MLLM Support**: mPLUG-Owl2-7B and LLaVA-1.5-7B
- **Comprehensive Evaluation**: Supports KADID-10k, KonIQ-10k, and AGIQA-3k datasets

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Visual Prompt Types](#visual-prompt-types)
- [Results](#results)
- [Citation](#citation)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX A6000)
- PyTorch
- HuggingFace Transformers

### Setup Environment

#### For mPLUG-Owl2:
```bash
# Clone and setup mPLUG-Owl2
git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG-Owl/mPLUG-Owl2
conda create -n mplug_owl2 python=3.10 -y
conda activate mplug_owl2
pip install --upgrade pip
pip install -e .
pip install 'numpy<2'
pip install protobuf
```

#### For LLaVA-1.5:
```bash
pip install transformers scipy pandas torch accelerate torchvision
```

#### Additional Dependencies:
```bash
pip install PyYAML scikit-learn tqdm
```

## 📊 Dataset Setup

### Download Datasets

Download the required IQA datasets:

```bash
# KonIQ-10k
wget https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/koniq10k.tgz
tar -xzf koniq10k.tgz

# KADID-10k  
wget https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/kadid10k.tgz
tar -xzf kadid10k.tgz

# AGIQA-3K
wget https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/AGIQA-3K.zip
unzip AGIQA-3K.zip
```

### Dataset Structure

Organize your datasets in the `data/` folder:
```
data/
├── split_agiqa3k.csv
├── split_kadid10k.csv
├── kadid10k/
├── koniq10k/
└── AGIQA-3K/
```

## 🏃 Quick Start

### 1. Training a Visual Prompt

```bash
cd src
python trainer.py 1
```

This runs experiment #1 from the predefined experiment list. The experiment ID maps to specific configurations in `trainer.py`.

### 2. Custom Training

Create a custom configuration file based on the examples in `configs/`:

```bash
# Copy an existing config
cp configs/other_models_configs/mplug2_experiment.yaml configs/my_experiment.yaml
# Edit the configuration
# Run training by modifying the config path in trainer.py
```

### 3. Inference

```bash
cd src
python tester.py
```

Modify the config path and checkpoint path in `tester.py` for your specific experiment.

## 🎯 Training

### Training Process

The training uses HuggingFace's `Trainer` with custom components:

1. **Model**: `VLMWithVisualPrompt` - combines frozen MLLM + trainable visual prompt
2. **Loss**: Mean Squared Error (MSE) between predicted and ground truth quality scores  
3. **Metrics**: SROCC and PLCC correlation coefficients
4. **Optimization**: SGD optimizer with configurable learning rates

### Key Training Parameters

- **Batch Size**: 32 (KADID-10k), 4 (KonIQ-10k, AGIQA-3k)
- **Learning Rate**: 60 (KADID-10k), varies by dataset
- **Epochs**: 25 (KADID-10k), 35 (AGIQA-3k)
- **Optimizer**: SGD

### Training Command

```bash
cd src
python trainer.py <experiment_number>
```

Where `<experiment_number>` corresponds to predefined experiments (1-45):
- Experiments 1-21: Addition mode prompts
- Experiments 22-42: Multiplication mode prompts  
- Experiments 43-45: Other model variants

## 🔍 Inference

### Running Inference

1. **Modify** `src/tester.py` to specify:
   - Configuration file path
   - Checkpoint path for trained visual prompt
   
2. **Run inference**:
```bash
cd src
python tester.py
```

### Output

The inference script outputs:
- SROCC (Spearman Rank Correlation Coefficient)
- PLCC (Pearson Linear Correlation Coefficient)
- Individual predictions vs ground truth

## ⚙️ Configuration

### Configuration File Structure

```yaml
experiment_name: "my_experiment"
model:
  vlm_name: "mplug2"  # or "llava15"
  visual_prompt:
    type: "padding"    # padding, patch_center, patch_topleft, overlay
    args:
      size: 30         # prompt size in pixels
      delta: 1         # scaling factor
      mode: "add"      # add or mul
      is_center: true  # for patch prompts
quality_score_type: "good+fine_vs_poor+bad"
training:
  output_dir: "outputs/my_experiment"
  num_train_epochs: 25
  learning_rate: 0.01
  per_device_train_batch_size: 32
  # ... other training parameters
dataset:
  train:
    path: "./data"
    name: "kadid"     # kadid, koniq, agiqa
    split: "train"
  # ... eval and test splits
```

### Quality Score Types

Available quality score extraction methods:
- `"good+fine_vs_poor+bad"`
- `"excellent+good_vs_poor+bad"`  
- `"good_vs_bad"`
- `"good_vs_poor"`
- `"fine_vs_poor"`
- `"excellent_vs_bad"`

## 🎨 Visual Prompt Types

### 1. Padding
Adds learnable pixels around image borders.
- **Sizes**: 10px, 30px
- **Parameters**: `3 × 2S(W + H - S)` where S=size, W=width, H=height

### 2. Fixed Patch (Center)
Places a learnable square patch at image center.
- **Sizes**: 10px, 30px  
- **Parameters**: `3 × S²`

### 3. Fixed Patch (Top-Left)
Places a learnable square patch at top-left corner.
- **Sizes**: 10px, 30px
- **Parameters**: `3 × S²`

### 4. Full Overlay
Applies learnable prompt across entire image.
- **Parameters**: `3 × H × W`

### Combination Modes
- **Addition**: `prompted_image = image + prompt`
- **Multiplication**: `prompted_image = image × prompt`

## 📈 Results

### Best Performance (30px Padding + Addition)

| Dataset | SROCC | PLCC | Parameters |
|---------|-------|------|------------|
| KADID-10k | 0.924 | 0.926 | 52K |
| KonIQ-10k | 0.852 | 0.874 | 52K |
| AGIQA-3k | 0.810 | 0.860 | 52K |

### Comparison with Full Fine-tuning

| Method | Parameters | KADID-10k SROCC |
|--------|------------|------------------|
| PixelPrompt | 52K | 0.910 |
| Q-Align | 7B | 0.919 |
| Q-Instruct | 7B | 0.706 |

## 🗂️ Repository Structure

```
├── src/
│   ├── trainer.py              # Main training script
│   ├── tester.py               # Inference script  
│   ├── vlms_plus_prompters.py  # Combined VLM+Prompt model
│   ├── vlms.py                 # Individual VLM implementations
│   ├── prompters.py            # Visual prompt modules
│   ├── datasets.py             # Dataset loading
│   ├── collators.py            # Data collation
│   └── utils.py                # Utility functions
├── configs/
│   ├── mplug_owl2_configs/     # mPLUG-Owl2 configurations
│   └── other_models_configs/   # Other model configurations
├── data/                       # Dataset files and splits
├── outputs/                    # Training outputs and checkpoints
└── README.md
```

## 🛠️ Key Implementation Details

### HuggingFace Trainer Integration
- Custom `VLMWithVisualPrompt` model combining frozen MLLM + trainable prompts
- Custom loss computation using MSE regression
- Custom metrics computation (SROCC, PLCC)

### Gradient Flow Preservation  
- Removed standard image processors to maintain gradient flow through pixels
- PyTorch-only preprocessing pipeline
- Visual prompts applied via tensor addition/multiplication

### Quality Score Extraction
- Extracts logits for quality-related tokens ("good", "bad", etc.)
- Applies softmax normalization to compute scalar quality scores
- Supports multiple token ensemble strategies

## ⚠️ Important Notes

1. **GPU Memory**: Training requires significant GPU memory due to MLLM size
2. **Gradient Flow**: Standard image processors break gradients - our implementation preserves them
3. **Model Selection**: mPLUG-Owl2 significantly outperforms LLaVA-1.5 in our experiments
4. **Hyperparameters**: Learning rates and batch sizes are dataset-specific

## 🔬 Reproducing Results

To reproduce the paper results:

1. **Setup environment** following installation instructions
2. **Download datasets** and place in `data/` folder  
3. **Run predefined experiments**:
   ```bash
   # Best results: 30px padding with addition
   python trainer.py 5   # KonIQ-10k + 30px padding
   python trainer.py 4   # KADID-10k + 30px padding  
   python trainer.py 6   # AGIQA-3k + 30px padding
   ```
4. **Evaluate** using `tester.py` with corresponding checkpoints

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{pixelprompt2024,
  title={PixelPrompt: Parameter-Efficient NR-IQA via Pixel-Space Visual Prompts in Multimodal LLMs},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes  
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl) for the base multimodal LLM
- [LLaVA](https://github.com/haotian-liu/LLaVA) for the alternative MLLM implementation
- HuggingFace Transformers for the training framework
- IQA-PyTorch-Datasets for providing standardized dataset splits 