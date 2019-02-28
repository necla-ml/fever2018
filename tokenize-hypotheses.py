from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

import unicodedata
import json
import re
import sys
import os
from tqdm import tqdm
import argparse

# from retrieval.fever_doc_db import FeverDocDB

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, required=True)
parser.add_argument("--out_file", type=str, required=True)
args = parser.parse_args()
in_file = args.in_file
out_file = args.out_file

if os.path.exists(out_file):
  raise ValueError("Output already exists")

jsondecoder = json.JSONDecoder()
jsonencoder = json.JSONEncoder()

tokenizer = SimpleWordSplitter()
print("Tokenizing")

with open(in_file, "r") as in_fp:
  with open(out_file, "w") as out_fp:
    for line in tqdm(in_fp.readlines()):
      struct = jsondecoder.decode(line)

      tok = tokenizer.split_words(struct["claim"])
      tokenized = " ".join(map(lambda x: x.text, tok))
      struct["claim"] = tokenized

      result = jsonencoder.encode(struct)
      out_fp.write(result + "\n")

