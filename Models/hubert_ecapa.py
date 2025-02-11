import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import HubertModel
import pytorch_lightning as pl
from rich import print
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class HubertEcapa(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.out_score_file_name = None
        self.ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.ecapa_tdnn = ECAPA_TDNN(768, lin_neurons=192)
        self.classifier = nn.Linear(192, 2)


    def forward(self, x):
        x_ssl_feat = self.ssl_model(x, output_hidden_states=True)
        x_ssl_feat = x_ssl_feat.last_hidden_state
        x = self.ecapa_tdnn(x_ssl_feat)
        x = self.classifier(x.squeeze())
        return x

    def test_step(self, batch, batch_idx):
        self._produce_evaluation_file(batch, batch_idx)

    def _produce_evaluation_file(self, batch, batch_idx):
        x, utt_id = batch
        fname_list = []
        score_list = []
        out = self(x)
        scores = (out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(scores.tolist())

        with open(self.out_score_file_name , "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))
        fh.close()

def load_model(model_path, out_score_file_name):
    if model_path:
        print(f'[bold green] Loading checkpoint from {model_path} [/bold green]')
        model = HubertEcapa.load_from_checkpoint(model_path, strict=True)
    else:
        model = HubertEcapa(out_score_file_name)
    
    model.out_score_file_name = out_score_file_name

    return model
