# Speech Deepfake Arena

### A comprehensive benchmark of current Anitspoofing systems on a wide collection of datasets. 

With the growing advent of machine-generated speech, the scientific community is responding with valuable contributions to detect deepfakes. With research moving at such a rapid pace, it becomes challenging to keep track of generalizability of SOTA deepfake detection systems. This tool allows users to compute EER, accuracy and F1 scores on popular countermeasure systems on any dataset provided a standardized protocol format.

The main leaderboard can be found on [Hugging face](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena)


# Usage 							
### 1. Data Preparation 

1. Create metadata.csv for any desired dataset with below format:
```
file_name,label
/path/to/audio1,spoof
/path/to/audio2,bonafide
...

```
NOTE : The labels should contain "spoof" for spoofed samples and "bonafide" for real samples.
       All the file_name paths should be absolute 

### 2. Obtain checkpoints and config files

Checkpoints and configuration files for the 10 systems currently on the leaderboard can be found [here](https://drive.google.com/file/d/1iajJbXtrTDgyvxQYBA44V9_-nd9RaMzj/view?usp=sharing) .The checkpoints must be downloaded since the TCM and XLSR+SLS systems have a [dependancy](https://github.com/Speech-Arena/speech_df_arena/blob/0fdeed13d964356339ab095beed2b552930cd3b4/Models/tcm_add.py#L292) on the XLSR 300M checkpoint present in the zip file mentioned above. The file should be unziped and all the file should be present in the `df_arena_checkpoints/` directory. 
### 3. Evaluation



Example usage : 
```py
python -u evaluation.py --model_name tcm_add         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/tcm_add/sonar.txt        \
                        --fix_length True        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/tcm_add_df_avg_5_best.pth>scores/logs/tcm_add/sonar.txt 
```
Example evaluation commands for all the models are shown in `examples.sh`

NOTE:- Kindly note that the Unispeech and Whisper Mesonet systems are not supported by DF Arena toolkit yet. But the inference code for these two systems has been provided in the same link [mentioned above]([here](https://drive.google.com/file/d/1iajJbXtrTDgyvxQYBA44V9_-nd9RaMzj/view?usp=sharing)) along with instructions


Score files for benchmarked systems can be found [here](https://drive.google.com/file/d/1pI-tvCZt4U__gGGLsCQMdZqLv_QBe4NW/view?usp=sharing)