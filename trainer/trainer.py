import torch
import torch.nn.functional as F
from tqdm import tqdm

class MaskedMultimodalTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        image_mask_ratio=0.4,
        text_mask_ratio=0.15,
        temperature=0.07
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_mask_ratio = image_mask_ratio
        self.text_mask_ratio = text_mask_ratio
        self.temperature = temperature
        
    def create_masks(self, batch_size, seq_length, mask_ratio):
        mask = torch.zeros(batch_size, seq_length)
        for i in range(batch_size):
            n_mask = int(seq_length * mask_ratio)
            mask_indices = torch.randperm(seq_length)[:n_mask]
            mask[i, mask_indices] = 1
        return mask.bool()
    
    def compute_loss(self, image_embeddings, text_embeddings):
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        similarity = torch.matmul(image_embeddings, text_embeddings.transpose(1, 2))
        similarity = similarity / self.temperature
        
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        loss = (
            F.cross_entropy(similarity, labels) +
            F.cross_entropy(similarity.transpose(1, 2), labels)
        ) / 2
        
        return loss
    
    def training_step(self, images, texts):
        batch_size = images.shape[0]
        
        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(images.device)
        
        image_mask = self.create_masks(
            batch_size,
            self.model.module.image_encoder.config.num_patches if hasattr(self.model, 'module') 
            else self.model.image_encoder.config.num_patches,
            self.image_mask_ratio
        ).to(images.device)
        
        text_mask = self.create_masks(
            batch_size,
            text_tokens.input_ids.shape[1],
            self.text_mask_ratio
        ).to(images.device)
        
        image_embeddings, text_embeddings = self.model(
            images=images,
            text_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            image_mask=image_mask,
            text_mask=text_mask
        )
        
        return self.compute_loss(image_embeddings, text_embeddings)

    def train(
        self,
        model,
        train_dataloader,
        val_dataset,
        optimizer,
        scheduler,
        scaler,
        writer,
        config,
        device
    ):
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                images, texts = batch
                images = images.to(device)
                
                with torch.cuda.amp.autocast():
                    loss = self.training_step(images, texts)
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.max_grad_norm
                    )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            
            epoch_loss = epoch_loss / len(train_dataloader)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_model.pth')
            
            print(f'Epoch {epoch+1} average loss: {epoch_loss:.4f}')
            
            if val_dataset:
                val_loss = validate(model, val_dataset, self, device)
                print(f'Validation loss: {val_loss:.4f}')
                writer.add_scalar('val/loss', val_loss, epoch)
        
        writer.close()
