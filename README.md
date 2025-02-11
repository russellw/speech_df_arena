# Speech Deep Fake Arena

### A comprehensive benchmark of current Anitspoofing systems on a wide collection of datasets. 

With the growing advent of machine-generated speech, the scientific community is responding with exciting resources to detect deep fakes. With research moving at such a rapid pace, it becomes challenging to keep track of generalizability of SOTA DF detection systems. This tool allows users to compute EER scores on 10 popular countermeasure systems across 16 different anti-spoofing tasks.

Also check out the leaderboard on Huggingface : 



# Usage 							
### 1 Data Preparation 

1. Create metadata.csv for any desired dataset with below format:
```
file_name,label
/path/to/audio1,spoof
/path/to/audio2,bonafide
...

```
NOTE : The labels should contain "spoof" for spoofed samples and "bonafide" for real samples.
       All the file_name paths should be absolute 

### 2 Evaluation

Almost all the models from the leaderboard are supproted. Checkpoints, configs metadata files for 
some datasets can be found here : 

Example usage : 
```py
python evaluation.py  --model_name wavlm_ecapa #Name of a supported model. See Models/ 
                      --batch_size 32 
                      --protocol_file_path #path to metadata csv protocol file
                      --model_path #path to model checkpoint
                      --model_config /path/to/model # path to model config. Leave empty if no config is required
                      --out_score_file_name /path/to/output/scores.txt #path to save the scores
                      --fix_length False  # Whether to trim input length to 4s 
                      -- num_workers 8 # No of dataloader workers
```
