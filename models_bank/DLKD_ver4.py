"""
DLKD version 4
- intermediate fused features use mse loss
- penultimate features use mse
- logits use KL divergence
- classwise decoder change second last layers to series of conv integrating classwise features from multiple class heads
"""

from typing import final
import torch.nn as nn
from models_bank.module.conv_encoder_decoder import cnn_block, tcnn_block
import torch
import torch.nn.functional as F
from models_bank.module.layers import general_conv2d_prenorm, fusion_prenorm_2d
import math


gf_dim = 32
TRANSFORMER_BASIC_DIMS = 256
MLP_DIM = 256
NUM_HEADS = 8
DEPTH = 2


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, return_seg_logits=True):
        super(Encoder, self).__init__()
        
        self.return_seg_logits = return_seg_logits
        
        self.e1 = cnn_block(input_channels,gf_dim,4,2,1, first_layer = True)
        self.e2 = cnn_block(gf_dim,gf_dim*2,4,2,1,)
        self.e3 = cnn_block(gf_dim*2,gf_dim*4,4,2,1,)
        self.e4 = cnn_block(gf_dim*4,gf_dim*8,4,2,1,first_layer=True) # (batch_size, 256, 32, 32)
        
        if return_seg_logits:
            self.d5 = tcnn_block(gf_dim*8,gf_dim*4,4,2,1)
            self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,4,2,1)
            self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,4,2,1)
            self.d8 = tcnn_block(gf_dim*1*2,output_channels,4,2,1, first_layer = True)
            self.softmax = nn.Softmax(dim=1)
            
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1,0.2))
        e3 = self.e3(F.leaky_relu(e2,0.2))
        e4 = self.e4(F.leaky_relu(e3,0.2))
        
        if self.return_seg_logits:
            d5 = torch.cat([self.d5(F.relu(e4)),e3],1)
            d6 = torch.cat([self.d6(F.relu(d5)),e2],1)
            d7 = torch.cat([self.d7(F.relu(d6)),e1],1)
            seg_pred = self.softmax(self.d8(F.relu(d7)))

            return e1, e2, e3, e4, seg_pred
        else:
            return e1, e2, e3, e4
        
        
class FusionTransformerBlock(nn.Module):
    def __init__(self, in_dim, patch_size, num_heads=8):
        super(FusionTransformerBlock, self).__init__()
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.mlp = nn.Linear(int(in_dim/2), in_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=MLP_DIM,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=DEPTH
        )
        self.rgb_pos = nn.Parameter(torch.zeros(1, patch_size**2, in_dim))
        self.ndsm_pos = nn.Parameter(torch.zeros(1, patch_size**2, in_dim))
        
    def forward(self, rgb_feat, ndsm_feat, prev_fusion=None):
        B, C, H, W = rgb_feat.shape
        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        ndsm_tokens = ndsm_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        fusion_token = self.fusion_token.expand(B, -1, -1)  # (B, 1, C)
        
        seq = torch.cat([rgb_tokens, ndsm_tokens], dim=1)  # (B, H*W*2, C)])
        seq_pos = torch.cat([self.rgb_pos.expand(B, -1, -1), self.ndsm_pos.expand(B, -1, -1)], dim=1)  # (B, H*W*2, C)
        seq = seq + seq_pos  # Add positional encoding
        
        if prev_fusion is not None:
            prev_fusion = self.mlp(prev_fusion).unsqueeze(1)  # (B, 1, C)
            
            # print(f"prev_fusion.shape: {prev_fusion.shape}")
            # print(f"fusion_token.shape: {fusion_token.shape}")
            # print(f"seq.shape: {seq.shape}")
            
            seq = torch.cat([fusion_token, seq, prev_fusion], dim=1)  # (B, H*W*2 + 2, C)
        else:
            seq = torch.cat([fusion_token, seq], dim=1) # (B, H*W*2 + 1, C)
        
        out = self.transformer(seq)  # (B, H*W*2 + 2, C)
        
        new_fusion = out[:, 0, :]  # Extract the fusion token output (B, C)
        
        # print(f"new_fusion.shape: {new_fusion.shape}")
        # print(f"out.shape: {out.shape}")
        
        return new_fusion, out
    
    
class AttnPoolToToken(nn.Module):
    def __init__(self, in_ch, out_ch=None, hidden=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, in_ch))
        
        if out_ch is not None:
            self.proj = nn.Linear(in_ch, out_ch)
        self.out_ch = out_ch
            
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1,2)  # (B, H*W, C)
        
        # compute attention scores using dot product
        # attn_scores = (x_flat @ self.query.t()).squeeze(-1)  # (B, H*W)
        
        # compute attention scores using MLP
        attn_scores = self.mlp(x_flat).squeeze(-1)
        
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, H*W, 1)
        
        pooled = (x_flat * attn_weights).sum(dim=1)  # (B, C)
        
        if self.out_ch is not None:
            pooled = self.proj(pooled)  # (B, out_ch)
            
        return pooled
        

class HiererchicalFusion(nn.Module):
    def __init__(self, in_dims, image_size, num_heads=6):
        super(HiererchicalFusion, self).__init__()
        # fuse rgb and ndsm features at each level
        self.conv_fusion_e1 = fusion_prenorm_2d(in_channel=in_dims[0], num_modals=2)
        self.conv_fusion_e2 = fusion_prenorm_2d(in_channel=in_dims[1], num_modals=2)
        self.conv_fusion_e3 = fusion_prenorm_2d(in_channel=in_dims[2], num_modals=2)
        
        # residual connection
        self.conv_res1 = cnn_block(in_dims[0], in_dims[1], 4, 2, 1)
        self.conv_res2 = cnn_block(in_dims[1], in_dims[2], 4, 2, 1)
        
        # pool features for transformer-based fusion
        self.pool_e3 = AttnPoolToToken(in_dims[2], out_ch=in_dims[3])
        
        # transformer-based fusion at level 4
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, in_dims[3]))
        self.patch_size_w = int(image_size[0] // 16) 
        self.patch_size_h = int(image_size[1] // 16)  # assuming image_size is (H, W) 
        self.rgb_pos = nn.Parameter(torch.zeros(1, self.patch_size_w * self.patch_size_h, in_dims[3]))
        self.ndsm_pos = nn.Parameter(torch.zeros(1, self.patch_size_w * self.patch_size_h, in_dims[3]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dims[3],
            nhead=num_heads,
            dim_feedforward=MLP_DIM,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=DEPTH
        )
        self.unify = nn.Conv2d(in_dims[3] * 2, in_dims[3], kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, rgb_feats, ndsm_feats):
        rgb_e1, rgb_e2, rgb_e3, rgb_e4 = rgb_feats
        ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4 = ndsm_feats
        
        fused_e1 = torch.cat((rgb_e1, ndsm_e1), dim=1)
        fused_e2 = torch.cat((rgb_e2, ndsm_e2), dim=1)
        fused_e3 = torch.cat((rgb_e3, ndsm_e3), dim=1)
        
        fused_e1 = self.conv_fusion_e1(fused_e1)
        
        fused_e1_ = self.conv_res1(fused_e1)
        fused_e2 = self.conv_fusion_e2(fused_e2) + fused_e1_
        
        fused_e2_ = self.conv_res2(fused_e2)
        fused_e3 = self.conv_fusion_e3(fused_e3) + fused_e2_
        
        fused_e3_pooled = self.pool_e3(fused_e3)  # (B, C)
        
        # transformer-based fusion at level 4
        B, C, H, W = rgb_e4.shape
        rgb_tokens = rgb_e4.flatten(2).transpose(1, 2)  # (B, H*W, C)
        ndsm_tokens = ndsm_e4.flatten(2).transpose(1, 2)  # (B, H*W, C)
        fusion_token = self.fusion_token.expand(B, -1, -1)  # (B, 1, C)
        seq = torch.cat([rgb_tokens, ndsm_tokens], dim=1)  # (B, H*W*2, C)])
        seq_pos = torch.cat([self.rgb_pos.expand(B, -1, -1), self.ndsm_pos.expand(B, -1, -1)], dim=1)  # (B, H*W*2, C)
        seq = seq + seq_pos  # Add positional encoding
        seq = torch.cat([fusion_token, seq, fused_e3_pooled.unsqueeze(1)], dim=1) # (B, H*W*2 + 1, C)
        fused_e4 = self.transformer(seq)  # (B, H*W*2 + 2, C)
        fused_e4_global = fused_e4[:, 0, :]  # Extract the fusion token output (B, C)
        fused_e4_rgb = fused_e4[:, 1:(self.patch_size_w*self.patch_size_h)+1, :]
        fused_e4_ndsm = fused_e4[:, (self.patch_size_w*self.patch_size_h)+1:-1, :]
        fused_e4_rgb = fused_e4_rgb.reshape(B, self.patch_size_w, self.patch_size_h, -1).permute(0, 3, 1, 2)  # (B, C, patch_size, patch_size)
        fused_e4_ndsm = fused_e4_ndsm.reshape(B, self.patch_size_w, self.patch_size_h, -1).permute(0, 3, 1, 2)  # (B, C, patch_size, patch_size)
        unified_fused_e4 = torch.cat((fused_e4_rgb, fused_e4_ndsm), dim=1)  # (B, C*2, patch_size, patch_size)
        unified_fused_e4 = self.unify(unified_fused_e4)  # (B, C, patch_size, patch_size)
        
        return fused_e1, fused_e2, fused_e3, unified_fused_e4, fused_e4_global

    
class HierarchicalAttnPool(nn.Module):
    def __init__(self, in_dims):
        super(HierarchicalAttnPool, self).__init__()
        self.attn_pools = nn.ModuleList([
            AttnPoolToToken(in_dim) for in_dim in in_dims
        ])
        
    def forward(self, x):
        pooled_outputs = []
        for i in range(len(self.attn_pools)):
            pooled_output = self.attn_pools[i](x[i])
            pooled_outputs.append(pooled_output)
        return pooled_outputs
    
    
def orthogonality_loss(a, b):
    """
    a: (B, D) - e.g., fused tokens
    b: (B, D) - e.g., modality-specific tokens
    Returns mean squared cosine similarity (should be minimized for orthogonality).
    """
    a_norm = F.normalize(a, dim=-1)  # (B, D)
    b_norm = F.normalize(b, dim=-1)  # (B, D)
    # Compute batchwise dot product
    dot_product = (a_norm * b_norm).sum(dim=-1)  # (B,)
    # Squared to penalize both positive and negative correlation
    loss = (dot_product ** 2).mean()  # Scalar
    return loss

def l2_kd_loss(a, b, spatial=False):
    """
    a: (B, D) or (B, C, H, W) - e.g., fused tokens
    b: (B, D) - e.g., modality-specific tokens
    Returns L2 loss
    """
    a_norm = F.normalize(a, p=2, dim=1, eps=1e-6)  # (B, D)
    b_norm = F.normalize(b, p=2, dim=1, eps=1e-6)  # (B, D)
    
    if spatial:
        diff = (a_norm - b_norm).pow(2)
        loss = diff.mean(dim=(0, 1, 2, 3))  # Mean over spatial dimensions
    else:
        loss =  F.mse_loss(a_norm, b_norm, reduction='mean')

    return loss

class ClassWiseSpatialAttention(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(ClassWiseSpatialAttention, self).__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.class_queries = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.class_queries)
        
    def forward(self, feat: torch.Tensor, viz=False):
        """
        feat: (B, C, H, W)
        returns: [B, K, H, W, C] where K=num_classes
        """
        B, C, H, W = feat.shape
        K = self.num_classes
        N = H * W
        
        flat = feat.view(B, C, N).permute(0, 2, 1)
        
        # expand & shape our queries → [B, K, C]
        #    (broadcasting the same class‐query across the batch)
        Q = self.class_queries.unsqueeze(0).expand(B, -1, -1)  # B×K×C
        
        # 3) compute raw dot‐product scores: [B, K, N]
        scores = torch.bmm(Q, flat.transpose(1,2))             # B×K×N
        scores = scores / math.sqrt(C)                         # optional
        
        attn = F.softmax(scores, dim=-1)  
        weighted = attn.unsqueeze(-1) * flat.unsqueeze(1)
        out = weighted.view(B, K, H, W, C)
        
        if viz:
            return out, attn, Q
        else:
            return out


class ClassWiseDecoder(nn.Module):
    def __init__(self, in_dims, out_channels, image_size=(512, 512)):
        super().__init__()
        self.image_size = image_size

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
        self.seg_d4 = nn.Conv2d(in_channels=in_dims[3], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.seg_d4_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_dims[3], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(out_channels)
        ])
        self.seg_d3_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_dims[2], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(out_channels)
        ])
        self.seg_d2_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_dims[1], out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(out_channels)
        ])
        self.seg_d1_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_dims[0]//8, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
            for _ in range(out_channels)
        ])
        
        self.e4_conv_up = general_conv2d_prenorm(in_dims[3], in_dims[2], pad_type='reflect')
        self.e3_conv = general_conv2d_prenorm(in_dims[2] * 2, in_dims[2], pad_type='reflect')
        self.e3_conv_up = general_conv2d_prenorm(in_dims[2], in_dims[1], pad_type='reflect')
        self.e2_conv = general_conv2d_prenorm(in_dims[1] * 2, in_dims[1], pad_type='reflect')
        self.e2_conv_up = general_conv2d_prenorm(in_dims[1], in_dims[0], pad_type='reflect')
        self.e1_conv = general_conv2d_prenorm(in_dims[0]*2, in_dims[0]//8, pad_type='reflect')
        self.e1_conv_up = general_conv2d_prenorm(in_dims[0]//4, in_dims[0]//8, pad_type='reflect')

        self.class_combine_conv = nn.Sequential(
            general_conv2d_prenorm(in_dims[0]//8*out_channels, in_dims[0]*2, pad_type='reflect'),
            general_conv2d_prenorm(in_dims[0]*2, in_dims[0], pad_type='reflect')
        )
        self.final_conv = nn.Conv2d(in_dims[0], out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, e1, e2, e3, classwise_attention, return_logits=False, viz=False):
        """
        e1, e2, e3: features from the encoder
        classwise_attention: [B, K, H, W, C]
        """
        B, K, H, W, C = classwise_attention.shape

        feats = []

        for i in range(K):
            # Extract the class-specific attention map
            class_attn = classwise_attention[:, i, :, :, :]
            class_attn = class_attn.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            
            # Apply seg_d4_layer for each class
            seg_map_e4 = self.seg_d4_layers[i](class_attn)  # (B, 1, H, W)
            if i == 0:
                seg_maps_e4 = seg_map_e4
            else:
                seg_maps_e4 = torch.cat([seg_maps_e4, seg_map_e4], dim=1)
            
            
            e4_up = self.e4_conv_up(self.up2(class_attn))
            e4_to_e3 = torch.cat([e3, e4_up], dim=1)
            e3_dec = self.e3_conv(e4_to_e3)
            seg_map_e3 = self.seg_d3_layers[i](e3_dec)
            if i == 0:
                seg_maps_e3 = seg_map_e3
            else:
                seg_maps_e3 = torch.cat([seg_maps_e3, seg_map_e3], dim=1)
                
            e3_up = self.e3_conv_up(self.up2(e3_dec))
            e3_to_e2 = torch.cat([e2, e3_up], dim=1)
            e2_dec = self.e2_conv(e3_to_e2)
            seg_map_e2 = self.seg_d2_layers[i](e2_dec)
            if i == 0:
                seg_maps_e2 = seg_map_e2
            else:
                seg_maps_e2 = torch.cat([seg_maps_e2, seg_map_e2], dim=1)
                
            e2_up = self.e2_conv_up(self.up2(e2_dec))
            e2_to_e1 = torch.cat([e1, e2_up], dim=1)
            e1_dec = self.e1_conv(e2_to_e1)
            seg_map_e1 = self.seg_d1_layers[i](e1_dec)
            if i == 0:
                seg_maps_e1 = seg_map_e1
            else:
                seg_maps_e1 = torch.cat([seg_maps_e1, seg_map_e1], dim=1)

            e1_dec = self.up2(e1_dec)
            e1_dec = F.interpolate(
                e1_dec,
                size=(self.image_size[0], self.image_size[1]),
                mode='bilinear', align_corners=False
            )
            feats.append(e1_dec)
            
        unified = torch.cat(feats, dim=1)

        # shape of [B, 32, 512, 512]
        
        penultimate_feats = self.class_combine_conv(unified)

        final_seg_logits = self.final_conv(penultimate_feats)
        
        # print(f"seg_maps_e4.shape: {seg_maps_e4.shape}") # [B, K, 16, 16]
        # print(f"seg_maps_e3.shape: {seg_maps_e3.shape}") # [B, K, 32, 32]
        # print(f"seg_maps_e2.shape: {seg_maps_e2.shape}") # [B, K, 64, 64]
        # print(f"seg_maps_e1.shape: {seg_maps_e1.shape}") # [B, K, 128, 128]
        seg_e4 = self.softmax(self.up16(seg_maps_e4))
        seg_e3 = self.softmax(self.up8(seg_maps_e3))
        seg_e2 = self.softmax(self.up4(seg_maps_e2))
        seg_e1 = self.softmax(self.up2(seg_maps_e1))
        final_seg = self.softmax(final_seg_logits)
        # print(f"final_seg.shape: {final_seg.shape}")  # ([B, K, 256, 256])
        
        if return_logits:
            return final_seg, [seg_e4, seg_e3, seg_e2, seg_e1], final_seg_logits, penultimate_feats
        elif viz:
            return final_seg, penultimate_feats, final_seg_logits
        else:
            return final_seg
        
        
class DLKD_ver4(nn.Module):
    def __init__(self, num_classes=5, image_size=(512, 512), ndsm_encoder_channels=1, rgb_encoder_channels=3):
        super(DLKD_ver4, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.ndsm_encoder_channels = ndsm_encoder_channels
        self.rgb_encoder_channels = rgb_encoder_channels

        self.is_training = True
        self.viz = False
        
        self.rgb_encoder = Encoder(self.rgb_encoder_channels, num_classes)
        self.ndsm_encoder = Encoder(self.ndsm_encoder_channels, num_classes)
        self.rgb_else = Encoder(self.rgb_encoder_channels, num_classes, return_seg_logits=False)
        self.ndsm_else = Encoder(self.ndsm_encoder_channels, num_classes, return_seg_logits=False)
        
        in_dims = [gf_dim, gf_dim*2, gf_dim*4, gf_dim*8]
        self.full_fuse = HiererchicalFusion(in_dims, image_size=image_size, num_heads=NUM_HEADS)
        self.full_classwise_attention = ClassWiseSpatialAttention(in_dim=in_dims[3], num_classes=num_classes)
        self.full_classwise_decoder = ClassWiseDecoder(in_dims, num_classes, image_size=image_size)

        self.miss_rgb_fuse = HiererchicalFusion(in_dims, image_size=image_size, num_heads=NUM_HEADS)
        self.miss_rgb_classwise_attention = ClassWiseSpatialAttention(in_dim=in_dims[3], num_classes=num_classes)
        self.miss_rgb_classwise_decoder = ClassWiseDecoder(in_dims, num_classes, image_size=image_size)
        
        self.miss_ndsm_fuse = HiererchicalFusion(in_dims, image_size=image_size, num_heads=NUM_HEADS)
        self.miss_ndsm_classwise_attention = ClassWiseSpatialAttention(in_dim=in_dims[3], num_classes=num_classes)
        self.miss_ndsm_classwise_decoder = ClassWiseDecoder(in_dims, num_classes, image_size=image_size)

        in_dims_lite = [gf_dim, gf_dim*2, gf_dim*4]
        self.miss_rgb_fuse_pool = HierarchicalAttnPool(in_dims_lite)
        self.miss_ndsm_fuse_pool = HierarchicalAttnPool(in_dims_lite)
        
        self.rgb_dist_pool = HierarchicalAttnPool(in_dims_lite)
        self.ndsm_dist_pool = HierarchicalAttnPool(in_dims_lite)
        
        self.rgb_else_pool = HierarchicalAttnPool(in_dims_lite)
        self.ndsm_else_pool = HierarchicalAttnPool(in_dims_lite)
        
        self.full_fusion_pool = HierarchicalAttnPool(in_dims_lite)
        
        self.MISS_NDSM = torch.tensor([True, False])
        self.MISS_RGB = torch.tensor([False, True])
        self.FULL_MODALITY = torch.tensor([True, True])
        self.T = 2
        

    def forward(self, x, masks=None, device='cuda'):
        B = x.shape[0]
        rgb = x[:, :self.rgb_encoder_channels, :, :]
        ndsm = x[:, self.rgb_encoder_channels:self.rgb_encoder_channels+self.ndsm_encoder_channels, :, :]
        
        self.MISS_NDSM = self.MISS_NDSM.to(device)
        self.MISS_RGB = self.MISS_RGB.to(device)
        self.FULL_MODALITY = self.FULL_MODALITY.to(device)
        
        if self.is_training:
            rgb_e1, rgb_e2, rgb_e3, rgb_e4, rgb_pred = self.rgb_encoder(rgb)
            ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4, ndsm_pred = self.ndsm_encoder(ndsm)
            
            rgb_feats = [rgb_e1, rgb_e2, rgb_e3, rgb_e4]
            ndsm_feats = [ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4]
            
            # Begin Full Branch #
            full_fused_e1, full_fused_e2, full_fused_e3, full_unified_fused_e4, full_fused_e4_global = self.full_fuse(rgb_feats, ndsm_feats)
            
            # print(f"full_fused_e1.shape: {full_fused_e1.shape}")  # ([B, 32, 128, 128])
            # print(f"full_fused_e2.shape: {full_fused_e2.shape}")   # ([B, 64, 64, 64])
            # print(f"full_fused_e3.shape: {full_fused_e3.shape}")  # ([B, 128, 32, 32])
            # print(f"Full_unified_fused_e4.shape: {full_unified_fused_e4.shape}")  # ([B, 256, 16, 16])
            # print(f"full_fused_e4_global.shape: {full_fused_e4_global.shape}")  # ([B, 256])
            # End Full Branch #
            
            # Begin Full Class-wise Attention Fusion #
            if self.viz:
                full_classwise_attention, full_attn, full_q = self.full_classwise_attention(full_unified_fused_e4, viz=True)
                full_pred, full_pen_feats, full_logits = self.full_classwise_decoder(full_fused_e1, full_fused_e2, full_fused_e3, full_classwise_attention, return_logits=False, viz=self.viz)
            else:
                full_classwise_attention = self.full_classwise_attention(full_unified_fused_e4)
                # print(f"full_classwise_attention.shape: {full_classwise_attention.shape}")  # ([B, K, H, W, C])
                full_pred, full_scale_preds, full_logits, full_pen_feats = self.full_classwise_decoder(full_fused_e1, full_fused_e2, full_fused_e3, full_classwise_attention, return_logits=True)
            # End Full Class-wise Attention Fusion #
            
            
            # Begin Missing Modality Handling #
            ## Begin NDSM Missing Modality Handling ##
            if (torch.equal(masks[0], self.MISS_NDSM)):
                rgb_else_1, rgb_else_2, rgb_else_3, rgb_else_4 = self.rgb_else(rgb)
                stu_feats = [rgb_else_1, rgb_else_2, rgb_else_3, rgb_else_4]
                
                stu_pooled_features = self.rgb_else_pool(stu_feats)
                stu_dist_pooled = self.rgb_dist_pool([rgb_e1, rgb_e2, rgb_e3, rgb_e4])
                
                miss_fused_e1, miss_fused_e2, miss_fused_e3, miss_unified_fused_e4, miss_fused_e4_global = self.miss_ndsm_fuse(rgb_feats, stu_feats)
                miss_fused_pooled = self.miss_ndsm_fuse_pool([miss_fused_e1, miss_fused_e2, miss_fused_e3])
                
                if self.viz:
                    missing_classwise_attention, missing_attn, missing_q = self.miss_ndsm_classwise_attention(miss_unified_fused_e4, viz=True)
                    missing_pred, missing_pen_feats, missing_logits = self.miss_ndsm_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention, return_logits=False, viz=self.viz)
                else:
                    missing_classwise_attention = self.miss_ndsm_classwise_attention(miss_unified_fused_e4)
                    missing_pred, missing_scale_preds, missing_logits, missing_pen_feats = self.miss_ndsm_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention, return_logits=True)
            ## End NDSM Missing Modality Handling ##

            ## Begin RGB Missing Modality Handling ##
            elif (torch.equal(masks[0], self.MISS_RGB)):
                ndsm_else_1, ndsm_else_2, ndsm_else_3, ndsm_else_4 = self.ndsm_else(ndsm)
                stu_feats = [ndsm_else_1, ndsm_else_2, ndsm_else_3, ndsm_else_4]
                
                stu_pooled_features = self.ndsm_else_pool(stu_feats)
                stu_dist_pooled = self.ndsm_dist_pool([ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4])
                
                miss_fused_e1, miss_fused_e2, miss_fused_e3, miss_unified_fused_e4, miss_fused_e4_global = self.miss_rgb_fuse(stu_feats, ndsm_feats)
                miss_fused_pooled = self.miss_rgb_fuse_pool([miss_fused_e1, miss_fused_e2, miss_fused_e3])
                
                if self.viz:
                    missing_classwise_attention, missing_attn, missing_q = self.miss_rgb_classwise_attention(miss_unified_fused_e4, viz=True)
                    missing_pred, missing_pen_feats, missing_logits = self.miss_rgb_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention, return_logits=False, viz=self.viz)
                else:
                    missing_classwise_attention = self.miss_rgb_classwise_attention(miss_unified_fused_e4)
                    missing_pred, missing_scale_preds, missing_logits, missing_pen_feats = self.miss_rgb_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention, return_logits=True)
            ## End RGB Missing Modality Handling ##
            # End Missing Modality Handling #
            
            # Begin Distillation Loss Calculation #
            miss_fused_pooled.append(miss_fused_e4_global)
            full_fusion_pooled = self.full_fusion_pool([full_fused_e1, full_fused_e2, full_fused_e3])
            full_fusion_pooled.append(full_fused_e4_global)
            
            # KLDivLoss expects (log_probs, probs)
            kl_loss = sum(l2_kd_loss(miss_fused_pooled[i], full_fusion_pooled[i].detach()) for i in range(len(miss_fused_pooled)))
            # monitor loss terms somewhere
            kl_loss += l2_kd_loss(missing_pen_feats, full_pen_feats.detach(), spatial=True)
            # End Distillation Loss Calculation #

            # Begin Diversity Loss Calculation #
            diversity_loss = sum(orthogonality_loss(stu_pooled_features[i], stu_dist_pooled[i]) for i in range(len(stu_pooled_features)))
            # End Diversity Loss Calculation #
            
            # temperature scaling
            full_logits = full_logits.detach() / self.T
            missing_logits = missing_logits / self.T
            p_full = F.softmax(full_logits, dim=1)
            log_q_miss = F.log_softmax(missing_logits, dim=1)
            kl_logits_loss = F.kl_div(log_q_miss, p_full, reduction='mean')
            kl_logits_loss = kl_logits_loss * self.T * self.T  # scale by T^2
            
            kl_loss += kl_logits_loss

            dict_results = {}
            if self.viz:
                dict_results['stu_feats'] = stu_feats
                dict_results['rgb_feats'] = rgb_feats
                dict_results['ndsm_feats'] = ndsm_feats
                dict_results['full_fused_feats'] = [full_fused_e1, full_fused_e2, full_fused_e3, full_unified_fused_e4]
                dict_results['miss_fused_feats'] = [miss_fused_e1, miss_fused_e2, miss_fused_e3, miss_unified_fused_e4]
                dict_results['full_classwise_attention'] = full_attn
                dict_results['missing_classwise_attention'] = missing_attn
                dict_results['full_q'] = full_q
                dict_results['missing_q'] = missing_q
                dict_results['full_pen_feats'] = full_pen_feats
                dict_results['missing_pen_feats'] = missing_pen_feats
                return dict_results
            
            dict_results = {
                'rgb_pred': rgb_pred,
                'ndsm_pred': ndsm_pred,
                'full_pred': full_pred,
                'missing_pred': missing_pred,
                # 'stu_else_pred': stu_else_pred,
                'full_scale_preds': full_scale_preds,
                'missing_scale_preds': missing_scale_preds,
                'kl_loss': kl_loss,
                'diversity_loss': diversity_loss,
            }
            
        else:
            if (torch.equal(masks[0], self.FULL_MODALITY)):
                rgb_e1, rgb_e2, rgb_e3, rgb_e4, rgb_pred = self.rgb_encoder(rgb)
                ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4, ndsm_pred = self.ndsm_encoder(ndsm)
                
                rgb_feats = [rgb_e1, rgb_e2, rgb_e3, rgb_e4]
                ndsm_feats = [ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4]
                
                full_fused_e1, full_fused_e2, full_fused_e3, full_unified_fused_e4, full_fused_e4_global = self.full_fuse(rgb_feats, ndsm_feats)
                full_classwise_attention = self.full_classwise_attention(full_unified_fused_e4)
                full_pred = self.full_classwise_decoder(full_fused_e1, full_fused_e2, full_fused_e3, full_classwise_attention)
                
                dict_results = {
                    'seg_pred': full_pred,
                }
            else:
                if (torch.equal(masks[0], self.MISS_NDSM)):
                    rgb_e1, rgb_e2, rgb_e3, rgb_e4, rgb_logits = self.rgb_encoder(rgb)
                    rgb_feats = [rgb_e1, rgb_e2, rgb_e3, rgb_e4]
                    rgb_else_1, rgb_else_2, rgb_else_3, rgb_else_4 = self.rgb_else(rgb)
                    stu_feats = [rgb_else_1, rgb_else_2, rgb_else_3, rgb_else_4]
                    miss_fused_e1, miss_fused_e2, miss_fused_e3, miss_unified_fused_e4, miss_fused_e4_global = self.miss_ndsm_fuse(rgb_feats, stu_feats)
                    missing_classwise_attention = self.miss_ndsm_classwise_attention(miss_unified_fused_e4)
                    missing_pred = self.miss_ndsm_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention)
                elif (torch.equal(masks[0], self.MISS_RGB)):
                    ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4, ndsm_logits = self.ndsm_encoder(ndsm)
                    ndsm_feats = [ndsm_e1, ndsm_e2, ndsm_e3, ndsm_e4]
                    ndsm_else_1, ndsm_else_2, ndsm_else_3, ndsm_else_4 = self.ndsm_else(ndsm)
                    stu_feats = [ndsm_else_1, ndsm_else_2, ndsm_else_3, ndsm_else_4]
                    miss_fused_e1, miss_fused_e2, miss_fused_e3, miss_unified_fused_e4, miss_fused_e4_global = self.miss_rgb_fuse(stu_feats, ndsm_feats)
                    missing_classwise_attention = self.miss_rgb_classwise_attention(miss_unified_fused_e4)
                    missing_pred = self.miss_rgb_classwise_decoder(miss_fused_e1, miss_fused_e2, miss_fused_e3, missing_classwise_attention)
                
                dict_results = {
                    'seg_pred': missing_pred
                }
        
        return dict_results
            
            

            
        
        
        
        
        
        
        
        
        