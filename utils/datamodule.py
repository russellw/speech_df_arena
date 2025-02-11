import pytorch_lightning as pl
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def generate_filelist(metadata_file):
    """
    Generates a list of spoofed audio file names from a metadata file.

    Args:
        dir_meta (str): Path to the metadata file. The file should contain lines
                        where each line represents a record with fields separated by spaces.
                        The second field in each line is assumed to be the file name.

    Returns:
        list: A list of audio file names.
    """

    df = pd.read_csv(metadata_file, usecols=['file_name','label'])

    return df.file_name.values


class Dataset_Eval(Dataset):
    def __init__(self, file_paths, fix_length=None):
        self.file_paths = file_paths
        self.fix_length = fix_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]


        X, fs = librosa.load(file_path, sr=16000)    

        if self.fix_length:
            X = pad(X)
            
        X = torch.from_numpy(X)

        return X, file_path


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, protocol_file_path, fix_length, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.protocol_file_path = protocol_file_path
        self.fix_length = fix_length
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        files_eval = generate_filelist(
           metadata_file=self.protocol_file_path,
       )

        if stage == "test":
           self.eval = Dataset_Eval(
               file_paths=files_eval,
                fix_length=self.fix_length,
           )

    def test_dataloader(self):
        collate_fn = None if self.fix_length else self._collate_fn_eval
        return DataLoader(
            dataset=self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,

        )
    
    def _collate_fn_eval(self, batch):
       x, file_path = zip(*batch)
       x = torch.nn.utils.rnn.pad_sequence(
           [tensor.squeeze() for tensor in x], batch_first=True, padding_value=0.0
       )
       return x, file_path



