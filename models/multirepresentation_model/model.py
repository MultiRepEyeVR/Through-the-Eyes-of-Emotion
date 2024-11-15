import sys
import torch
from torch.nn import functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange, reduce
from math import floor
from modules.util import pair, exists
from multirepresentation_model.vivit import PatchEmbeddingLayer, LearnablePositionalEncoding, Transformer, FactorizedTransformer
from multirepresentation_model.ts_transformer import PositionalEncoding, TransformerBatchNormEncoderLayer
from multirepresentation_model.cross_attention import UnifiedCrossAttention, AttentionFeedForward, LearnableWeightedSum


class MultiRepresentationModel(nn.Module):
    def __init__(self, vit_config, ts_transformer_config):
        super(MultiRepresentationModel, self).__init__()
        self.periocular_encoder = PeriocularEncoder(**vit_config)
        self.gaze_pupil_encoder = GazePupilEncoder(**ts_transformer_config)
        self.cross_attention = RepresentationCrossAttention(eye0_dim=self.periocular_encoder.dim,
                                                          eye1_dim=self.periocular_encoder.dim,
                                                          gaze_dim=self.gaze_pupil_encoder.output_dim,
                                                          dim=256,
                                                          heads=8,
                                                          dropout=0.1)

        self.final_layer = EmotionClassifier(self.cross_attention.dim, 7)

    def forward(self, eye0, eye1, ts):
        eye0 = self.periocular_encoder(eye0)
        eye1 = self.periocular_encoder(eye1)
        ts = self.gaze_pupil_encoder(ts)
        fused = self.cross_attention(eye0, eye1, ts)
        return self.final_layer(fused)
    

class PeriocularEncoder(nn.Module):
    def __init__(
            self,
            image_size,
            image_patch_size,
            frames,
            frame_patch_size,
            dim,
            spatial_depth,
            temporal_depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            enc_dropout=0.,
            emb_dropout=0.,
            variant='factorized_encoder',
    ):
        super(PeriocularEncoder, self).__init__()

        # Each frame must be square
        # Patches must also be square
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.dim = dim

        self.to_patch_embedding = PatchEmbeddingLayer(patch_height, patch_width, frame_patch_size, patch_dim, dim)

        self.learnable_positional_encoding = LearnablePositionalEncoding(num_image_patches * num_frame_patches, dim)

        # self.learnable_positional_encoding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, enc_dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, enc_dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, enc_dropout)

        # self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, enc_dropout)
        # self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, enc_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.variant = variant

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, f, n, _ = x.shape
        x = self.learnable_positional_encoding(x)
        # x = x + self.learnable_positional_encoding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d-> b f 1 d', b=b, f=f)
            x = torch.cat((spatial_cls_tokens, x), dim=2)

        x = self.emb_dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            x = self.spatial_transformer(x)

            x = rearrange(x, '(b f) n d -> b f n d', b=b)

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b=b)
                x = torch.cat((temporal_cls_tokens, x), dim=1)

            x = self.temporal_transformer(x)

            x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)

            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        return self.to_latent(x)
    

class GazePupilEncoder(nn.Module):
    def __init__(
            self,
            feat_dim,
            max_len,
            d_model,
            n_heads,
            num_layers,
            dim_feedforward,
            enc_dropout,
            emb_dropout,
            embedding='linear',
            conv_config=None,
            output_pool=None
    ):
        super(GazePupilEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        if embedding == "linear":
            self.proj_inp = nn.Linear(feat_dim, d_model)
            print(f"Linear embedding: {self.max_len} sequence length.")
        elif embedding == "convolution":
            assert conv_config is not None, "Embedding is chosen as Conv, but conv_config is empty."
            self.proj_inp = nn.Sequential(
                Rearrange('b l d -> b d l'),  # Rearrange input shape to [batch_size, feat_dim, seq_length]
                nn.Conv1d(feat_dim, d_model, kernel_size=conv_config['kernel_size'], stride=conv_config['stride'],
                          padding=conv_config['padding'], dilation=conv_config['dilation']),
                Rearrange('b d l -> b l d')  # Rearrange output shape to [batch_size, seq_length, d_model]
            )
            proj_conv_seq_len = int(floor((self.max_len + 2 * conv_config['padding'] - conv_config['dilation'] * (
                    conv_config['kernel_size'] - 1) - 1) / conv_config['stride'] + 1))
            self.max_len = proj_conv_seq_len
            print(f"Convolutional embedding: {self.max_len} sequence length.")
        else:
            sys.exit("Embedding should be either 'linear' or 'convolution'.")

        self.pos_embedding = PositionalEncoding(self.max_len, d_model)

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, enc_dropout, pre_norm=False)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

        self.act = F.gelu

        self.enc_dropout = nn.Dropout(enc_dropout)

        self.output_pool = output_pool

        self.output_dim = d_model * self.max_len if not self.output_pool else d_model

    def forward(self, x):
        if self.embedding == "linear" or self.embedding == "convolution":
            x = self.proj_inp(x) * (self.d_model ** 0.5)
        else:
            sys.exit("The embedding layer should either be 'linear' or 'convolution")
        b, l, d = x.shape
        x = self.pos_embedding(x)
        x = self.emb_dropout(x)
        x = self.transformer_encoder(x)
        x = self.act(x)
        x = self.enc_dropout(x)
        x = x.reshape(b, -1) if not self.output_pool else reduce(x, 'b l d -> b d', 'mean')
        return x
    

class RepresentationCrossAttention(nn.Module):
    def __init__(self, eye0_dim, eye1_dim, gaze_dim, dim=768, heads=8, dropout=0.1):
        super(RepresentationCrossAttention, self).__init__()
        self.eye0_embedding = nn.Linear(eye0_dim, dim)
        self.eye1_embedding = nn.Linear(eye1_dim, dim)
        self.gaze_embedding = nn.Linear(gaze_dim, dim)
        self.dim = dim

        self.unified_cross_attention = UnifiedCrossAttention(dim, heads, dropout)
        self.ffn = AttentionFeedForward(dim, expansion_factor=4, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.final_layer_norm = nn.LayerNorm(dim)

        self.modality_weights = nn.Parameter(torch.ones(3))
        self.pooling = LearnableWeightedSum(num_modalities=3)

    def forward(self, eye0, eye1, ts):
        eye0 = self.eye0_embedding(eye0)
        eye1 = self.eye1_embedding(eye1)
        ts = self.gaze_embedding(ts)

        weights = F.softmax(self.modality_weights, dim=0)
        eye0 = eye0 * weights[0]
        eye1 = eye1 * weights[1]
        ts = ts * weights[2]

        combined = torch.stack([eye0, eye1, ts], dim=1)

        normalized = self.layer_norm1(combined)
        attended = self.unified_cross_attention(normalized)
        combined = combined + attended

        output = self.layer_norm2(combined)
        ffn_output = self.ffn(output)
        output = combined + ffn_output
        output = self.final_layer_norm(output)

        pooled_output = self.pooling(output)

        return pooled_output

    def log_weights(self, epoch, file_path):
        modality_weights = F.softmax(self.modality_weights, dim=0)
        pooling_weights = F.softmax(self.pooling.weights, dim=0)
        with open(file_path, 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"Modality weights: {modality_weights.tolist()}\n")
            f.write(f"Pooling weights: {pooling_weights.tolist()}\n\n")


class EmotionClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear(x)