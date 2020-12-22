TASK=levenshtein
DATA=../DATA/data-bin/im2latex
WORKERS=8

mkdir -p checkpoints/$TASK
mkdir -p logdir/$TASK

export CUDA_VISIBLE_DEVICES=0

fairseq-train --user-dir .. \
    --task image_captioning_lev --noise random_delete \
    --max-tokens 30000 \
    --max-tokens-valid 30000 \
    --update-freq 1 \
    --arch vgg_levenshtein_transformer \
    --sampling-for-deletion \
    --encoder-layers 3 \
    --decoder-layers 3 \
    --apply-bert-init \
    --share-decoder-input-output-embed \
    --criterion nat_loss \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --lr 1e-3 \
    --clip-norm 10.0  \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --eval-bleu \
    --eval-bleu-args '{"iter_decode_max_iter": 9, "iter_decode_eos_penalty": 3}' \
    --eval-bleu-detok space \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --validate-interval 1 \
    --save-interval 1 \
    --save-dir checkpoints/$TASK \
    --tensorboard-logdir logdir/$TASK \
    --log-format simple --log-interval 50 \
    --max-update 1000000 \
    --patience 50 \
    --fp16 \
    --num-workers $WORKERS \
    $DATA 
