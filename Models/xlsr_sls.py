import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pytorch_lightning as pl
import fairseq
CHECKPOINTS_DIR = os.environ.get('DF_ARENA_CHECKPOINTS_DIR', '../df_arena_stuff/checkpoints/')

class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        # Create XLSR model structure with xlsr_sls checkpoint weights
        print(f"[green]Creating XLSR model structure (weights from xlsr_sls checkpoint)[/green]")
        
        from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
        from argparse import Namespace
        
        # Use exact XLSR-53 configuration
        args = Namespace()
        args.extractor_mode = 'layer_norm'
        args.encoder_layers = 24
        args.encoder_embed_dim = 1024
        args.encoder_ffn_embed_dim = 4096
        args.encoder_attention_heads = 16
        args.activation_fn = 'gelu'
        args.dropout = 0.0
        args.attention_dropout = 0.0
        args.activation_dropout = 0.0
        args.encoder_layerdrop = 0.0
        args.dropout_input = 0.0
        args.dropout_features = 0.0
        args.final_dim = 768
        args.layer_norm_first = True
        args.conv_feature_layers = '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]'
        args.conv_bias = True
        args.logit_temp = 0.1
        args.quantize_targets = True
        args.quantize_input = False
        args.same_quantizer = False
        args.target_glu = False
        args.feature_grad_mult = 1.0
        args.latent_vars = 320
        args.latent_groups = 2
        args.latent_dim = 0
        args.mask_length = 10
        args.mask_prob = 0.65
        args.mask_selection = 'static'
        args.mask_other = 0.0
        args.no_mask_overlap = False
        args.mask_min_space = 1
        args.mask_channel_length = 10
        args.mask_channel_prob = 0.0
        args.mask_channel_selection = 'static'
        args.mask_channel_other = 0.0
        args.no_mask_channel_overlap = False
        args.mask_channel_min_space = 1
        args.num_negatives = 100
        args.negatives_from_everywhere = False
        args.cross_sample_negatives = 0
        args.codebook_negatives = 0
        args.conv_pos = 128
        args.conv_pos_groups = 16
        args.latent_temp = "[2.0, 0.1, 0.999995]"
        
        # Create model structure
        self.model = Wav2Vec2Model(args)
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.eval()  # Set to eval mode for inference!

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # Get features from feature extractor
            features = self.model.feature_extractor(input_tmp)
            features = features.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            features = self.model.layer_norm(features)
            
            # Apply post extract projection if it exists
            if self.model.post_extract_proj is not None:
                features = self.model.post_extract_proj(features)
            
            # Get representations from all encoder layers
            encoder_padding_mask = None  # Assuming no padding for simplicity
            layer_results = []
            
            x = features.transpose(0, 1)  # (B, T, C) -> (T, B, C) for transformer
            
            for layer in self.model.encoder.layers:
                x, attn = layer(x, encoder_padding_mask)
                # Store (layer_output, attention) in the expected format
                layer_results.append((x, attn))
            
            # Final layer norm
            if self.model.encoder.layer_norm is not None:
                x = self.model.encoder.layer_norm(x)
            
            # Final projection
            if self.model.final_proj is not None:
                x = self.model.final_proj(x)
            
            emb = x.transpose(0, 1)  # (T, B, C) -> (B, T, C)
            layerresult = layer_results
        return emb, layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:

        layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature


class XLSR_SLS(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)

        return output

    

class XLSR_SLS_antispoofing(pl.LightningModule):
    def __init__(self, model):
        super(XLSR_SLS_antispoofing, self).__init__()
        self.model = model
        self.out_score_file_name = None

    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        self._produce_evaluation_file(batch, batch_idx)

    def _produce_evaluation_file(self, batch, batch_idx):
        x, utt_id = batch
        fname_list = []
        score_list = []


        batch_out = self(x)
        batch_score = (batch_out[:, 1]
                    ).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(self.out_score_file_name, "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))

def load_model(model_path,  out_score_file_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pytorch_model = XLSR_SLS(None, device)
    
    if model_path:
        print(f'[bold green] Loading checkpoint from {model_path} [/bold green]')
        pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False), strict=False)
    
    model = XLSR_SLS_antispoofing(pytorch_model)
    
    # Set model to evaluation mode - critical for inference!
    model.eval()
    
    model.out_score_file_name = out_score_file_name

    return model