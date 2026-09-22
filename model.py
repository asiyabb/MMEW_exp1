import torch
import torch.nn as nn
import torchvision.models as models
import config

class LearnableGateFusion(nn.Module):
    """
    Applies learnable gating to combine ViT spatial features (Dv) 
    and Optical Flow motion features (Dm). Output shape: [B, T, Dv + Dm]
    """
    def __init__(self, d_v=config.VIT_DIM, d_m=config.MOTION_DIM):
        super().__init__()
        self.gate_v = nn.Sequential(
            nn.Linear(d_v, d_v),
            nn.Sigmoid()
        )
        self.gate_m = nn.Sequential(
            nn.Linear(d_m, d_m),
            nn.Sigmoid()
        )
        
    def forward(self, feat_v, feat_m):
        # feat_v: [B, T, Dv], feat_m: [B, T, Dm]
        gated_v = feat_v * self.gate_v(feat_v)
        gated_m = feat_m * self.gate_m(feat_m)
        fused = torch.cat([gated_v, gated_m], dim=-1)  # [B, T, Dv + Dm]
        return fused

class MicroToMacroGuidedTransformer(nn.Module):
    """
    Dual-path spatial-temporal Transformer architecture guided by fine-grained micro-features
    to improve macro-expression recognition.
    """
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        # 1. Spatial Branch: Vision Transformer (ViT-B/16)
        vit_backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        vit_backbone.heads = nn.Identity()
        self.vit_encoder = vit_backbone  # Outputs 768-D spatial embeddings
        
        # 2. Dynamic Motion Encoder (Optical Flow Representation)
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # Accepts 2-channel optical flow (u, v)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config.MOTION_DIM)  # Outputs 256-D motion vector
        )
        
        # 3. Learnable Fusion Gate
        self.fusion_gate = LearnableGateFusion(d_v=config.VIT_DIM, d_m=config.MOTION_DIM)
        
        # 4. Spatio-Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.FUSED_DIM, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=0.3, 
            batch_first=True
        )
        
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 5. Micro-Expression Recognition Head
        self.micro_head = nn.Sequential(
            nn.Linear(config.FUSED_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 6. Micro-Guided Cross-Attention Block for Macro-Expression Recognition
        self.micro_to_macro_attention = nn.MultiheadAttention(
            embed_dim=config.FUSED_DIM, 
            num_heads=8, 
            batch_first=True
        )
        
        # 7. Macro-Expression Recognition Head
        self.macro_head = nn.Sequential(
            nn.Linear(config.FUSED_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, frames, optical_flows):
        """
        frames: [B, T, 3, 224, 224]
        optical_flows: [B, T, 2, 224, 224]
        """
        B, T, C, H, W = frames.shape
        
        # Step 1: Process Frames via Vision Transformer (ViT)
        frames_flat = frames.view(B * T, C, H, W)
        vit_feats = self.vit_encoder(frames_flat).view(B, T, config.VIT_DIM)
        
        # Step 2: Process Motion via Optical Flow Encoder
        flows_flat = optical_flows.view(B * T, 2, H, W)
        motion_feats = self.motion_encoder(flows_flat).view(B, T, config.MOTION_DIM)
        
        # Step 3: Learnable Gated Feature Fusion
        fused_seq = self.fusion_gate(vit_feats, motion_feats)  # [B, T, Dv + Dm]
        
        # Step 4: Temporal Context via Transformer
        temporal_tokens = self.temporal_transformer(fused_seq)  # [B, T, 1024]
        
        # Step 5: Micro-Expression Prediction (Fine-Grained Guidance Source)
        micro_pooled = torch.mean(temporal_tokens, dim=1)  # [B, 1024]
        micro_logits = self.micro_head(micro_pooled)       # [B, NUM_CLASSES]
        
        # Step 6: Micro-Guided Macro-Expression Feature Refinement
        # Query: Global Temporal Tokens | Key/Value: Micro Spatial-Temporal Tokens
        guided_macro_tokens, _ = self.micro_to_macro_attention(
            query=temporal_tokens,
            key=temporal_tokens,
            value=temporal_tokens
        )
        macro_pooled = torch.mean(guided_macro_tokens, dim=1)
        macro_logits = self.macro_head(macro_pooled)       # [B, NUM_CLASSES]
        
        return micro_logits, macro_logits