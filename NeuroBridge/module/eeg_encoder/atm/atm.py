import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange

from .subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from .subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from .subject_layers.Embed import DataEmbedding

class Config:
    def __init__(self, channels_num):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Feedforward network dimension
        self.activation = 'gelu'           # Activation function
        self.enc_in = channels_num         # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False,  num_subjects=10):
        super(iTransformer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout, joint_train=False, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :self.configs.enc_in, :]      
        # print("enc_out", enc_out.shape)
        return enc_out



class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, configs=None):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (configs.enc_in, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, configs=None):
        super().__init__(
            PatchEmbedding(emb_size, configs),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
        
class ATMS(nn.Module):    
    def __init__(
        self,
        channels_num=63,
        feature_dim=1024,
        eeg_sample_points=250,
        num_subjects=10,
    ):
        super(ATMS, self).__init__()
        default_config = Config(channels_num)
        self.encoder = iTransformer(default_config, num_subjects=num_subjects)
        self.enc_eeg = Enc_eeg(configs=default_config)
        if eeg_sample_points == 200:
            embedding_dim = 1040
        elif eeg_sample_points == 250:
            embedding_dim = 1440
        else:
            raise ValueError(f"unsupported eeg_sample_points={eeg_sample_points}")
        self.proj_eeg = Proj_eeg(embedding_dim=embedding_dim, proj_dim=feature_dim)
        self.num_subjects = num_subjects
         
    def forward(self, x: Tensor, subject_ids: Tensor | None = None):
        if x.ndim == 4:
            x = x.squeeze(1)
        if subject_ids is None:
            subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        subject_ids = subject_ids.long().clamp(0, self.num_subjects - 1)
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out
    
    
if __name__ == "__main__":
    # Example usage
    eeg_sample_points = 250
    channels_num = 17
    feature_dim = 768
    model = ATMS(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    
    # Create a dummy EEG input tensor with shape (batch_size, channels_num, eeg_sample_points)
    batch_size = 8
    dummy_eeg_input = torch.randn(batch_size, channels_num, eeg_sample_points)
    
    # Forward pass through the model
    output = model(dummy_eeg_input)
    print(output.shape)  # Expected output shape: (batch_size, feature_dim)