from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.encoder import MultimodalEncoder
from trainer.trainer import MaskedMultimodalTrainer
from data.dataset import MultimodalDataset
from utils.training import setup_training, validate
from config.training_config import TrainingConfig

def train_model(config: TrainingConfig, train_dataset, val_dataset=None):
    # Create data loader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    # Initialize model and trainer
    model = MultimodalEncoder()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trainer = MaskedMultimodalTrainer(model, tokenizer)
    
    # Setup training components
    device, model, optimizer, scheduler, scaler, writer = setup_training(
        config, model, train_dataloader
    )
    
    # Training loop
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        config=config,
        device=device
    )

if __name__ == "__main__":
    config = TrainingConfig()
    # Create dataset
    train_dataset = MultimodalDataset(
        image_paths=[],  # Add actual image paths
        texts=[],        # Add actual text data
    )
    train_model(config, train_dataset)
