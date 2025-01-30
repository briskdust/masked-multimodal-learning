import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
