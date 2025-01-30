import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.encoder import MultimodalEncoder
from trainer.trainer import MaskedMultimodalTrainer
from data.dataset import MultimodalDataset
from utils.training import setup_training, validate
from config.training_config import TrainingConfig

class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        image_encoder_name="google/vit-base-patch16-224",
        text_encoder_name="bert-base-uncased",
        projection_dim=512,
        dropout_rate=0.1
    ):
        super().__init__()
        # 添加dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 添加层归一化
        self.image_layer_norm = nn.LayerNorm(projection_dim)
        self.text_layer_norm = nn.LayerNorm(projection_dim)
        
        # 图像编码器
        self.image_encoder = ViTModel.from_pretrained(image_encoder_name)
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        
        # 投影层
        self.image_projection = nn.Linear(768, projection_dim)
        self.text_projection = nn.Linear(768, projection_dim)
        
        # Mask token embeddings
        self.image_mask_token = nn.Parameter(torch.randn(1, 1, 768))
        self.text_mask_token = nn.Parameter(torch.randn(1, 1, 768))

    def forward(self, images, text_ids, attention_mask, image_mask=None, text_mask=None):
        batch_size = images.shape[0]
        
        # 处理图像
        image_features = self.image_encoder(images).last_hidden_state  # [B, N, D]
        if image_mask is not None:
            # 应用patch级别的掩码
            mask_tokens = self.image_mask_token.expand(batch_size, -1, -1)
            image_features = torch.where(
                image_mask.unsqueeze(-1), 
                mask_tokens, 
                image_features
            )
        
        # 处理文本
        text_features = self.text_encoder(
            input_ids=text_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # [B, L, D]
        
        if text_mask is not None:
            # 应用token级别的掩码
            mask_tokens = self.text_mask_token.expand(batch_size, -1, -1)
            text_features = torch.where(
                text_mask.unsqueeze(-1),
                mask_tokens,
                text_features
            )
        
        # 添加dropout和层归一化
        image_embeddings = self.image_layer_norm(self.dropout(self.image_projection(image_features)))
        text_embeddings = self.text_layer_norm(self.dropout(self.text_projection(text_features)))
        
        return image_embeddings, text_embeddings

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
        """创建随机掩码"""
        mask = torch.zeros(batch_size, seq_length)
        for i in range(batch_size):
            n_mask = int(seq_length * mask_ratio)
            mask_indices = torch.randperm(seq_length)[:n_mask]
            mask[i, mask_indices] = 1
        return mask.bool()
    
    def compute_loss(self, image_embeddings, text_embeddings):
        """计算对比损失"""
        # 归一化embeddings
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(image_embeddings, text_embeddings.transpose(1, 2))
        similarity = similarity / self.temperature
        
        # 创建标签（对角线为正例）
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        # 计算对比损失
        loss = (
            F.cross_entropy(similarity, labels) +
            F.cross_entropy(similarity.transpose(1, 2), labels)
        ) / 2
        
        return loss
    
    def training_step(self, images, texts):
        """单个训练步骤"""
        batch_size = images.shape[0]
        
        # 对文本进行tokenize
        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(images.device)
        
        # 创建掩码
        image_mask = self.create_masks(
            batch_size,
            self.model.image_encoder.config.num_patches,
            self.image_mask_ratio
        ).to(images.device)
        
        text_mask = self.create_masks(
            batch_size,
            text_tokens.input_ids.shape[1],
            self.text_mask_ratio
        ).to(images.device)
        
        # 前向传播
        image_embeddings, text_embeddings = self.model(
            images=images,
            text_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            image_mask=image_mask,
            text_mask=text_mask
        )
        
        # 计算损失
        loss = self.compute_loss(image_embeddings, text_embeddings)
        
        return loss

# 示例数据集类
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 这里需要实现具体的图像加载逻辑
        image = None  # load_image(self.image_paths[idx])
        text = self.texts[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 确保图像张量在正确的设备上
        if isinstance(image, torch.Tensor):
            image = image.to(device)
            
        return image, text

def train_model(config: TrainingConfig, train_dataset, val_dataset=None):
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    # 初始化模型和训练器
    model = MultimodalEncoder()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trainer = MaskedMultimodalTrainer(model, tokenizer)
    
    # 设置训练组件
    device, model, optimizer, scheduler, scaler, writer = setup_training(
        config, model, train_dataloader
    )
    
    # 训练循环
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images, texts = batch
            images = images.to(device)
            
            with torch.cuda.amp.autocast():
                loss = trainer.training_step(images, texts)
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
            val_loss = validate(model, val_dataset, trainer, device)
            print(f'Validation loss: {val_loss:.4f}')
            writer.add_scalar('val/loss', val_loss, epoch)
    
    writer.close()

if __name__ == "__main__":
    config = TrainingConfig()
    # 创建数据集
    train_dataset = MultimodalDataset(
        image_paths=[],  # 添加实际的图像路径
        texts=[],        # 添加实际的文本数据
    )
    train_model(config, train_dataset)
