USER=
FAIRSEQ=/home/${USER}/fairseq
PRETRAIN=/home/${USER}/mbart.cc25/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
DATADIR=/home/${USER}/postprocessed/en-ja
SRC=en_XX
TGT=ja_XX
SAVEDIR=checkpoint_en_ja

python ${FAIRSEQ}/train.py ${DATADIR}  --encoder-normalize-before --decoder-normalize-before  --arch mbart_large --task translation_from_pretrained_bart  --source-lang ${SRC} --target-lang ${TGT} --criterion label_smoothed_cross_entropy --label-smoothing 0.2  --dataset-impl mmap --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --max-update 80000 --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 --max-tokens 768 --update-freq 2 --save-interval 1 --save-interval-updates 8000 --keep-interval-updates 10 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler --restore-file $PRETRAIN --langs $langs --layernorm-embedding  --ddp-backend no_c10d --save-dir ${SAVEDIR}
