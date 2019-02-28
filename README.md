# FEVER 2018 System from Team Papelo, NEC Laboratories America

This is the [NEC Labs America](http://www.nec-labs.com) Team Papelo [FEVER 2018 system](https://www.github.com/necla-ml/fever2018) for Fact Extraction and Verification, for the [FEVER shared task at EMNLP](http://fever.ai).  Please cite our [system description paper](http://aclweb.org/anthology/W18-5517) from the EMNLP workshop:

```
@inproceedings{malon2018,
  title={Team Papelo: Transformer Networks at FEVER},
  author={Christopher Malon},
  booktitle={Proceedings of the EMNLP First Workshop on Fact Extraction and Verification},
  year={2018}
}
```

# Preparing the software

```
git clone https://github.com/necla-ml/fever2018
git submodule update --init --recursive
conda create -n fever-papelo python=3.6
source activate fever-papelo
conda install pytorch=0.3.1 torchvision -c torch
cd fever2018
pip install -r requirements.txt
python -m spacy download en
```

# Obtaining the data

```
bash fever2018-retrieval/scripts/download-raw-wiki.sh
bash fever2018-retrieval/scripts/download-processed-wiki.sh
bash fever2018-retrieval/scripts/download-data.sh
```

# Applying the retrieval module

The following retrieves the set of sentences to be classified for each claim.
For the training and development sets, we only retrieve the top 5 sentences
by TFIDF score, but for the test set, we retrieve whole documents for the
best document matches (truncating after 50 sentences each).  If you are using
the pretrained model, you may skip proessing the training and development sets.
If you don't care about reproducing the original results, we recommend you
omit the `--compat` flag, which reproduces earlier behavior.
```
PYTHONPATH=fever2018-retrieval/src python fever2018-retrieval/src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --in-file data/fever-data/train.jsonl --out-file data/fever/train.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --compat
PYTHONPATH=fever2018-retrieval/src python fever2018-retrieval/src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --in-file data/fever-data/dev.jsonl --out-file data/fever/dev.sentences.p5.s5.jsonl --max-page 5 --max-sent 5 --compat
PYTHONPATH=fever2018-retrieval/src python fever2018-retrieval/src/scripts/retrieval/ir.py --db data/fever/fever.db --model data/index/fever-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --in-file data/fever-data/test.jsonl --out-file data/fever/test.sentences.p5.s50.jsonl --max-page 5 --max-sent 50 --whole-docs
```

Then we preprocess the retrieved sentences for use with the entailment module.
For the training and development sets:

```
python retrieved-sentences.py --in_file data/fever/train.sentences.p5.s5.jsonl --out_file data/fever/train-onesentence.jsonl --fever_pages_dir data/wiki-pages
python tokenize-hypotheses.py --in_file data/fever/train-onesentence.jsonl --out_file data/fever/train-tokenized.jsonl
python entailment-title-input.py --in_file data/fever/train-tokenized.jsonl --out_dir data/fever --split train

python retrieved-sentences.py --in_file data/fever/dev.sentences.p5.s5.jsonl --out_file data/fever/dev-onesentence.jsonl --fever_pages_dir data/wiki-pages
python tokenize-hypotheses.py --in_file data/fever/dev-onesentence.jsonl --out_file data/fever/dev-tokenized.jsonl
python entailment-title-input.py --in_file data/fever/dev-tokenized.jsonl --out_dir data/fever --split dev
```

The test set uses different scripts which do not require ground truth labels:
```
python test-retrieved-sentences.py --in_file data/fever/test.sentences.p5.s50.jsonl --out_file data/fever/test-onesentence.jsonl --fever_pages_dir data/wiki-pages
python tokenize-hypotheses.py --in_file data/fever/test-onesentence.jsonl --out_file data/fever/test-tokenized.jsonl
python test-entailment-title-input.py --in_file data/fever/test-tokenized.jsonl --out_dir data/fever --split test
```
Retrieval for the test set took us about three hours.

For our final model, we concatenated training and development data into a
bigger training set.  You may easily do this after running ir.py above.

# Retraining the entailment model

```
cd finetune-transformer-lm
python train.py --dataset entailment --desc entailment --data_dir ../data/fever --n_gpu 3
cd ..
```

This script assumes that training, development, and test data exist
in `data_dir`, in files prefixed by "train", "dev", and "test" as prepared
by `entailment-title-input.py` above.  It will validate on the development
data during training.  If the `--submit` flag is given, it will
run on the test set after training is complete.
Training takes a little over 15 hours on a system with three GTX 1080 Ti
GPU's.

# Applying the entailment model

Instead of retraining as above, you may
[download our pretrained model](https://github.com/necla-ml/fever2018-model).

To output entailment decisions for each premise and claim:
```
cd finetune-transformer-lm
python predict.py --desc entailment --dataset entailment --model_file save/entailment/best_params.jl --test_prefix ../data/fever/test --n_ctx 348 --result_file ../data/fever/test.output.tsv
cd ..
```
If you downloaded the pretrained model, substitute its path for the
`model_file` argument.  Classification on the Fever test set takes
about five and a half hours on our three-GPU system.

If you retrain on a different data set, the `n_ctx` argument may be different.
It is determined by the longest context that appears in the training set
(counted in subword tokens).  From the error message you get with the
wrong `n_ctx` value, you will get the size of the word embedding dictionary
(such as 40829).  This includes the 40478 words in the transformer's language
model, 3 special token types, and one encoding for every possible positional
encoding up to what the value of `n_ctx` should be.  Thus you can recover
the right value for `n_ctx` by subtracting: 40829-40478-3=348.

Finally, reassociate the entailment decisions with the original claims to
formulate the final submission file:
```
python aggregate-decisions.py --original_jsonl data/fever-data/test.jsonl --index_file data/fever/test.index --decision_file data/fever/test.output.tsv --submission_file data/fever/test-predictions.jsonl
```

