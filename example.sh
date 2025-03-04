#AASIST

python -u evaluation.py --model_name aasist         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/aasist/sonar.txt        \
                        --fix_length True        \
                        --model_config df_arena_checkpoints/aasist.json \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/aasist.pth>scores/logs/aasist/sonar.txt 

python -u evaluation.py --model_name tcm_add         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/tcm_add/sonar.txt        \
                        --fix_length True        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/tcm_add_df_avg_5_best.pth>scores/logs/tcm_add/sonar.txt 


python -u evaluation.py --model_name wavlm_ecapa         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/wavlm_ecapa/sonar.txt        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/wavlm_ecapa.ckpt>scores/logs/wavlm_ecapa/sonar.txt   \
     

python -u evaluation.py --model_name rawgat_st         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/rawgat_st/sonar.txt        \
                        --model_config df_arena_checkpoints/rawgat_st.yaml         \
                        --fix_length True        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/rawgat_st.pth>scores/logs/rawgat_st/sonar.txt     



python -u evaluation.py --model_name hubert_ecapa         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/hubert_ecapa/sonar.txt        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/hubert_ecapa.ckpt>scores/logs/hubert_ecapa/sonar.txt  


python -u evaluation.py --model_name rawnet_2         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/rawnet_2/sonar.txt        \
                        --model_config df_arena_checkpoints/rawnet.yaml         \
                        --fix_length True        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/rawnet_2.pth>scores/logs/rawnet_2/sonar.txt  


python -u evaluation.py --model_name xlsr_sls         \
                        --batch_size 64        \
                        --protocol_file_path metadata_files/sonar.csv          \
                        --out_score_file_name scores/xlsr_sls/sonar.txt        \
                        --fix_length True        \
                        --num_workers 4 \
                        --model_path df_arena_checkpoints/xlsr_sls.pth>scores/logs/xlsr_sls/sonar.txt 
