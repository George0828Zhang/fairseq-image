TASK=debug
DATA=../DATA/data-bin/im2latex

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

export CUDA_VISIBLE_DEVICES=0

fairseq-train --user-dir .. \
    --task image_captioning_lev --noise full_mask \
    --max-tokens 30000 \
    --max-tokens-valid 30000 \
    --update-freq 1 \
    --arch vgg_nat \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --share-decoder-input-output-embed \
    --sg-length-pred \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 746 \
    --lr 5e-4 \
    --clip-norm 10.0  \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --save-interval-updates 100 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-bleu-detok space \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --log-format simple --log-interval 50 \
    --max-update 1000000 \
    --patience 160 \
    --fp16 \
    $DATA 

    # 64, 128, 256, 512
    # --eval-tokenized-bleu \
    # --fp16 \
    # --save-interval-updates 1000 \
    # --keep-interval-updates 1 \