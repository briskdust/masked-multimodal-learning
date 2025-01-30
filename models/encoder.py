import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class MultimodalEncoder(nn.Module):
    def __init__(
        self,
        image_encoder_name="google/vit-base-patch16-224",
        text_encoder_name="bert-base-uncased",
        projection_dim=512,
        dropout_rate=0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.image_layer_norm = nn.LayerNorm(projection_dim)
        self.text_layer_norm = nn.LayerNorm(projection_dim)
        
        self.image_encoder = ViTModel.from_pretrained(image_encoder_name)
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        
        self.image_projection = nn.Linear(768, projection_dim)
        self.text_projection = nn.Linear(768, projection_dim)
        
        self.image_mask_token = nn.Parameter(torch.randn(1, 1, 768))
        self.text_mask_token = nn.Parameter(torch.randn(1, 1, 768))

    def forward(self, images, text_ids, attention_mask, image_mask=None, text_mask=None):
        batch_size = images.shape[0]
        
        image_features = self.image_encoder(images).last_hidden_state
        if image_mask is not None:
            mask_tokens = self.image_mask_token.expand(batch_size, -1, -1)
            image_features = torch.where(image_mask.unsqueeze(-1), mask_tokens, image_features)
        
        text_features = self.text_encoder(
            input_ids=text_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        if text_mask is not None:
            mask_tokens = self.text_mask_token.expand(batch_size, -1, -1)
            text_features = torch.where(text_mask.unsqueeze(-1), mask_tokens, text_features)
        
        image_embeddings = self.image_layer_norm(self.dropout(self.image_projection(image_features)))
        text_embeddings = self.text_layer_norm(self.dropout(self.text_projection(text_features)))
        
        return image_embeddings, text_embeddings
