import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from utils.datamodule import DataModule
from Models.model_factory import ModelFactory
import tqdm
from utils.metrics import compute_det_curve, compute_eer, compute_eer_API
from rich import print
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Speech DF Arena Toolkit')
    parser.add_argument('--protocol_file_path', type=str, help='Path to the protocol metadata csv file', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference', required=True)
    parser.add_argument('--model_name', type=str, default='rawgat_st', help='Name of the model', required=True)
    parser.add_argument('--model_path', type=str,  help='Path to the model checkpoint')
    parser.add_argument('--model_config', type=str, help='Path to the model json if required')
    parser.add_argument('--out_score_file_name', type=str, default='scores.txt', help='Name of the output score file', required=True)
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--fix_length', help='Whether to fix the ibput audio length to 4s or not. Used in many models', type=bool, required=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the dataloader",
    )

    args = parser.parse_args()

    model = ModelFactory.get_model(args.model_name, args.model_path, args.model_config, args.out_score_file_name)

    data_module = DataModule(args.batch_size,  args.protocol_file_path, args.fix_length, args.num_workers)
    data_module.prepare_data()
    data_module.setup('test')
    pl.seed_everything(args.seed, workers=True)

    trainer = pl.Trainer(
        accelerator=args.device,
        enable_model_summary=True,
        logger = False
    )

    if os.path.exists(args.out_score_file_name):
        print(f'[bold yellow] WARNING : a score file already exists at {args.out_score_file_name}. New results will be appended to this file. [/bold yellow]')

    print(f'Starting evaluation for {args.model_name}')
    trainer.test(model, datamodule=data_module)

    print("Testing completed")
    eer, th = compute_eer_API(args.out_score_file_name,
                              args.protocol_file_path)
    print("EER (%): {:.4f}".format(eer * 100))

if __name__ == "__main__":
    main()
