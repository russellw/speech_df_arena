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
        
        cp_path = os.path.join(CHECKPOINTS_DIR, 'xlsr2_300m.pt')
        print(f"[green]Loading XLSR-300M model from {cp_path}[/green]")
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path], strict=False)
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            out = self.model(input_tmp, mask=False, features_only=True)
            emb = out['x']
            # Handle both old and new fairseq versions
            if 'layer_results' in out:
                layerresult = out['layer_results']
            elif 'features' in out:
                layerresult = out['features']
            else:
                # Create dummy layer results based on the main output
                layerresult = [(emb, None) for _ in range(24)]  # 24 layers for XLSR
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
    
    model.out_score_file_name = out_score_file_name

    return model