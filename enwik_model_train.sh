set -e

CPUNUM=8
GPUNUM=4

ATTN="sfl-group"

# enwik8, text8
MAXSTEP=200000
LOGSTEP=200
EVALSTEP=10000
DROPOUT=0.1
DROPATT=0.0
LAYER=9
D_MODEL=256
N_HEAD=8
D_HEAD=32
D_INNER=1024

GPS=4

BATCH=22
LR=0.00025
CLIP=0.25
INITSTD=0.02
PRE=1
FFF=0
TGTLEN=512
MEMLEN=512
EVALTGTLEN=128

DATA="data/enwik8"
DATASET="enwik8"

WORKDIR="model/enwik8"

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--attn_type sfl-group --multi_gpu --gpu0_bsz 4 \
--tgt_len $TGTLEN --mem_len $MEMLEN --eval_tgt_len $EVALTGTLEN \
--lr $LR --max_step $MAXSTEP --log-interval $LOGSTEP --eval-interval $EVALSTEP \
--data $DATA --dataset $DATASET --work_dir=$WORKDIR --cuda \
--n_layer $LAYER --d_model $D_MODEL --n_head $N_HEAD --d_head $D_HEAD --d_inner $D_INNER \
--init_std $INITSTD --dropout $DROPOUT --clip $CLIP --dropatt $DROPATT --batch_size $BATCH \
--n_group $GPS --not_tied
