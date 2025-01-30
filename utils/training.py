import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def setup_training(config, model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()
    
    return device, model, optimizer, scheduler, scaler, writer

@torch.no_grad()
def validate(model, val_dataloader, trainer, device):
    model.eval()
    total_loss = 0
    
    for batch in val_dataloader:
        images, texts = batch
        images = images.to(device)
        
        with torch.cuda.amp.autocast():
            loss = trainer.training_step(images, texts)
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_dataloader)
    model.train()
    return avg_loss
