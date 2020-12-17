# mBART English-Japanese finetuning

It can be difficult to follow the [original instructions](https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md) on how to finetune mBART. This is what I did to finetune it for English-Japanese and Japanese-English translation.

Some of these packages may be outdated for the current version of `fairseq`. If you find an issue, contributions are welcome.

## Install dependencies
I used a clean installation of python 3.7 as a start.

```python -m venv nlp
source nlp/bin/activate
pip install pytorch
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

* install sentencepiece from git (git clone and compile from source, [instructions here](https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source))

sentencepiece is found in
`/usr/local/bin/spm_encode`

## Download the data (japanese)

For finetuning on Japanese we use wikimatrix
and jpparacrawl.

```
wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-ja.tsv.gz
wget http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/2.0/bitext/en-ja.tar.gz
```

## Download the checkpoint
```
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
tar -xzvf mbart.CC25.tar.gz
```

## Preprocessing
* Join all data and split by language
`python prepare_data.py`
* Run sentencepiece on the data (check the paths are all correct)
`sh run_sentencepiece.sh`
* Run the fairseq preprocess
`sh fairseq-preprocess.sh`

Make sure that the lang names are `en_XX` and `ja_XX` on files, not `en` and `ja`.

## Train
To train the reverse direction, you need to swap `SRC` and `TGT` languages. Just run:

`sh train.sh`

## Evaluate
See `load_checkpoint.py`
