USER=
DATA=/home/${USER}/preprocessed
FAIRSEQ=/home/${USER}/fairseq
DEST=/home/${USER}/fairseq
TRAIN=train
VALID=valid
TEST=test
SRC=en_XX
TGT=ja_XX
DEST=/home/${USER}/postprocessed
NAME=en-ja
DICT=/home/${USER}/mbart.cc25/dict.txt

python ${FAIRSEQ}/preprocess.py \
--source-lang ${SRC} \
--target-lang ${TGT} \
--trainpref ${DATA}/${TRAIN}.spm \
--validpref ${DATA}/${VALID}.spm \
--testpref ${DATA}/${TEST}.spm  \
--destdir ${DEST}/${NAME} \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70
