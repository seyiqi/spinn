#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -N spinn_encoded_large
#PBS -j oe
#PBS -M apd283@nyu.edu
#PBS -l mem=8GB
#PBS -l walltime=24:00:00

module load cuda/7.5.18
module load cudnn/7.0v4.0
module load numpy/intel/1.10.1

MODEL_NAME="spinn_encoded_large"

cd spinn
. .venv-hpc/bin/activate
cd checkpoints

export PYTHONPATH=../python:$PYTHONPATH
export THEANO_FLAGS=allow_gc=False,cuda.root=/usr/bin/cuda,warn_float64=warn,device=gpu,floatX=float32

export MODEL_FLAGS=" \
--batch_size 32 \
--ckpt_path $MODEL_NAME.ckpt_best \
--ckpt_interval_steps 1000 \
--connect_tracking_comp \
--data_type snli \
--embedding_data_path ../glove/glove.840B.300d.txt \
--embedding_keep_rate 0.938514416034 \
--eval_data_path ../snli_1.0/snli_1.0_dev.jsonl \
--eval_seq_length 50 \
--experiment_name $MODEL_NAME \
--init_range 0.005 \
--l2_lambda 2.76018187539e-05 \
--learning_rate 0.00103428201391 \
--log_path ../logs \
--model_dim 600 \
--model_type Model1 \
--num_sentence_pair_combination_layers 1 \
--predict_use_cell \
--semantic_classifier_keep_rate 0.949455648614 \
--seq_length 50 \
--tracking_lstm_hidden_dim 44 \
--training_data_path ../snli_1.0/snli_1.0_train.jsonl \
--transition_cost_scale 0.605159568546 \
--use_tracking_lstm  \
--use_encoded_embeddings \
--enc_embedding_dim 100 \
--word_embedding_dim 300 \
"

echo "THEANO_FLAGS: $THEANO_FLAGS"
echo "python -m spinn.models.fat_classifier $MODEL_FLAGS"

python -m spinn.models.fat_classifier $MODEL_FLAGS
