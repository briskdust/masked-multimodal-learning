# Masked Multimodal Training Framework

A PyTorch-based framework for masked multimodal training that supports joint training of image and text modalities. This framework utilizes ViT and BERT as backbone encoders, combining contrastive learning with masked pretraining.

## Features

- Multimodal training support for images and text
- Patch-level image masking and token-level text masking
- Contrastive learning for multimodal alignment
- Mixed precision training support
- Multi-GPU training support
- Complete training and validation pipeline
- Training monitoring with Tensorboard
- Gradient accumulation and clipping
- Cosine learning rate scheduling

## Project Structure

```
.
├── config/
│ └── training_config.py # Training configuration class
├── data/
│ └── dataset.py # Dataset implementation
├── models/
│ └── encoder.py # Multimodal encoder model
├── trainer/
│ └── trainer.py # Trainer implementation
├── utils/
│ └── training.py # Training utility functions
└── main.py # Main training script
```

## Installation

```bash
pip install torch torchvision
pip install transformers
pip install tensorboard
pip install tqdm
```

## Quick Start

1. Prepare your dataset:

    ```python
    train_dataset = MultimodalDataset(
    image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
    texts=['text1', 'text2'],
    transform=your_transform
    )
    ```

2. Configure training parameters:

    ```python
    from config.training_config import TrainingConfig
    config = TrainingConfig(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0
    )
    ```

3. Start training:

    ```python
    from main import train_model
    train_model(config, train_dataset)
    ```


## Core Components

### MultimodalEncoder

The multimodal encoder class combines ViT and BERT:

- Feature extraction for both images and text
- Learnable mask tokens
- Projection layers for common space mapping
- Dropout and layer normalization implementation

### MaskedMultimodalTrainer

The trainer class implements core training logic:

- Random mask generation
- Contrastive learning loss computation
- Training step handling

### MultimodalDataset

The dataset class:

- Support for image-text pair loading
- Customizable data transformations
- Device management support

## Training Configuration

Configurable parameters through `TrainingConfig`:

- `num_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `warmup_steps`: Number of warmup steps
- `gradient_accumulation_steps`: Gradient accumulation steps
- `max_grad_norm`: Maximum gradient norm

## Training Monitoring

Monitor training progress with Tensorboard:
```bash
tensorboard --logdir runs/
```

Monitored metrics include:

- Training loss
- Validation loss
- Learning rate changes

## Model Checkpointing

Best model is automatically saved during training:

- Save path: `best_model.pth`
- Saved contents:
  - Model state
  - Optimizer state
  - Training epoch
  - Best loss value

## Important Notes

1. Data Preparation:
   - Implement specific image loading logic
   - Ensure correct dataset formatting

2. Hardware Requirements:
   - GPU recommended for training
   - Adjust batch size according to GPU memory

3. Training Optimization:
   - Adjustable masking ratios
   - Configurable temperature parameter
   - Customizable learning rate scheduling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome!
