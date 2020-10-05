# Why Skip If You Can Combine: A Simple Knowledge Distillation Technique for Intermediate Layers

This is the code for our paper:
>[Why Skip If You Can Combine: A Simple Knowledge Distillation Technique for Intermediate Layers]
>Yimeng Wu\*, Peyman Passban\*, Mehdi Rezagholizadeh, Qun Liu (*These authors contributied equally) 
>Proceedings of EMNLP. 2020.

## Requirements
- Python >=3.6; Pytorch==1.3.0
- Install packages: 
```bash
pip install -r requirements.opt.txt
```

## Data Preparation
- The original data consists of parallel source (src) and target(tgt) tokenized data.
    - src-train.txt
    - tgt-train.txt
    - src-val.txt
    - tgt-val.txt
- First the tokenized train & valid data are processed into three .pt files
    - prefix.train.num_of_features.pt 
    - prefix.valid.num_of_features.pt 
    - prefix.vocab.pt
- Key Features:
    - `data_dir`: directory for the raw data
    - `train_src`: the name of the training source data
    - `train_tgt`: the name of the training target data
    - `valid_src`: the name of the valid source data
    - `valid_tgt`: the name of the valid target data
    - `save_data`: Output file for the prepared data (with prefix)
    - `src_vocab`: Path to an existing source vocabulary. Format: one word per line. 
    - `tgt_vocab`: Path to an existing target vocabulary. Format: one word per line.
    - `share_vocab`: Share source and target vocabulary
    - `src_seq_length`: Maximum source sequence length
    - `tgt_seq_length`: Maximum target sequence length
    - `save_data`: save path with prefix
```
python preprocess.py -data_dir <original_data_dir> -train_src <train_source_data_name> -train_tgt <train_target_data_name> -valid_src <valid_source_data_name> 
-valid_tgt <valid_target_data_name> -save_data <save_path_with_prefix> -src_vocab <soruce_vocab_file_name> -tgt_vocab <target_vocab_file_name> 
-share_vocab 
-src_seq_length <the maximum length of the source language> -tgt_seq_length <the maximum length of the target language>
```
### Example

We upload our 200k ende datasets under data/.The command below will process these tokenized train & valid data into three .pt files: data.train.num_of_features.pt, data.valid.num_of_features.pt, data.vocab.pt. 
```
python preprocess.py
    -data_dir data/200k_vocab_15k/raw_200k_data
    -train_src train_200k.spm.en
    -train_tgt train_200k.spm.de
    -valid_src newstest2013-src.spm.en
    -valid_tgt newstest2013-ref.spm.de
    -save_data: data/200k_vocab_15k/processed
    -src_vocab: ende_200k.vocab
    -tgt_vocab: ende_200k.vocab
    -share_vocab: 'true'
    -src_seq_length: 300
    -tgt_seq_length: 300

```

## Training with or without distillation
- Four training modes are supported for transformer models: NO-KD, RKD, PKD, CKD (regular_comb, cross_comb, overlap_comb, skip_middle)
    - RKD: loss = alpha * soft_loss + (1-alpha)*hard_loss 
    - PKD: loss = alpha * soft_loss + delta * hard_loss + beta * MSE_enc(H^S, H^T) (alpha, beta and delta should sum to 1)
    - Comb_model:
        - Regular_comb: [1,2,3] -> 1, [4,5,6]->2
        - Overlap_comb: [1,2,3,4] -> 1, [3,4,5,6] ->2
        - Skip_middle: [1,2] ->1, [5,6] ->2
        - Cross_comb: [1,3] ->1, [4,6] -> 2

- They code only support to train 12-layer teacher and 4-layer student now.
- Key Features:
    - `data`: Path to the pre-processed .pt files with prefix
    - `config`: config file path (model configs are defined here)
    - `teacher`: If KD is applied then this is the path to the teacher model's state_dict.
    - `logit_distillation`: Whether to do KD on the output logits. If yes then the soft loss willbe added to the loss function. Default to False.
    - `enc_distillation_mode`: Whether to do the intermediate-layer KD on encoder side. If yes then the internal_loss will be added to the loss function. Default to False.
        - Possible choices: None, skip (PKD), regular_comb, cross_comb, overlap_comb, skip_middle
    - `tensorboard`: Whether to use tensorboard during training. Default to False.
    - `tensorboard_log_dir`: Log directory for Tensorboard. This is also the name of the run.
    - `world_size`: how many gpus are used.
    - `save_model`: Path to the saved model. Model filename (the model will be saved as <save_model>_N.pt where N is the number of steps.
    - `val_result_file`: Output validation results to a file under this path. (Used to choose the top best models during inference)
    - `alpha`: the fixed loss weight before softloss. 
    - `beta`: the fixed loss weight before internal_loss. Used when trainable_loss_weight is False. 
- Side Features:
    - `dec_distillation_mode`: Whether to do the intermediate-layer KD on decoder side. If yes then the internal_loss will be added to the loss function. Default to False.
        - Possible choices: None, skip (PKD), regular_comb, cross_comb, overlap_comb, skip_middle
    - `attn_distillation_mode`: Do which distillation on attention matrix (only support for PKD now)
        - Possible choices: only_self, only_context, both_self_context
    - `theta`: the fixed weight before attn_loss

- Before training, make sure you `export CUDA_VISIBLE_DEVICES=0,1,2,3`. For example, If you want to use GPU id 1 and 3 of your OS, you will need to export `CUDA_VISIBLE_DEVICES=1,3`.
    
### Example
Here is an example on how to use regular_comb on processed en -> de dataset. The command below will save the student models in save_model (or train_url) with filename transformer_N.pt (N=step number) and a val_result file to record the validation result. Vocab File is expected to be present in <data>.vocab.pt. 

#### Step 1: Train a transformer_base teacher
```bash
python train.py 
    -config config/distillation/transformer-base.yml
    -world_size 4
    -data data/200k_vocab_15k/processed
    -save_model <Save_path_with_prefix>
    -val_result_file <Path to the saved validation result file> 
```
#### Step 2: Train the student
```bash
python train.py 
    -data data/200k_vocab_15k/processed
    -save_model <Save_path_with_prefix>
    -logit_distillation
    -config config/distillation/transformer-2layer.yml
    -world_size 4
    -enc_distillation_mode regular_comb
    -teacher <Path to the teacher model>
    -val_result_file <Path to the saved validation result file> 
    -alpha 0.2
    -beta 0.7
```

## Inference
- Key Features:
    - `train_dir`: Path to model directory where model is stored.
    - `data_dir`: Path to data directory where data is stored.
    - `src`: Source sequence to decode (one line per sequence); multiple separated with spaces.
    - `tgt`: True target sequence.
    - `best_n_ckpts`: avg best n checkpoints based upon loss.
    - `gpu`: Device to run on.
    - `max_length`: Maximum prediction length.
    - `val_result_file`: Output validation results to a file with this name.
    - `model_prefix`: Model prefix used at the time of training.
    - `vocab_model`: Name of vocab model file for detokenization in data_dir.

```bash
python translate.py \
    -train_dir <Path to the model> \
    -data_dir data/200k_vocab_15k/raw_200k_data \
    -src newstest2014-src.spm.en \
    -tgt newstest2014-ref.de \
    -best_n_ckpts 3 \
    -gpu 0 \
    -max_length 600 \
    -val_result_file val_result.yml \
    -model_prefix transformer \
    -vocab_model ende_vocab.model \
```

## Acknowledgement
This repo is based on (https://github.com/OpenNMT/OpenNMT-py) [Open-NMT-PyTorch]