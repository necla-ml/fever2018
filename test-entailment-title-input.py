from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
from tqdm import tqdm
import unicodedata
import json
import re
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--split", help="which part of dataset", type=str, required=True)
parser.add_argument("--max_sent_len", help="maximum sentence length", type=int, default=200)
args = parser.parse_args()
in_file = args.in_file
out_dir = args.out_dir
split = args.split
max_sent_len = args.max_sent_len

jsondecoder = json.JSONDecoder()

tokenizer = SimpleWordSplitter()

premise_fp = open(out_dir + "/" + split + ".premise", "w")
hypothesis_fp = open(out_dir + "/" + split + ".hypothesis", "w")
label_fp = open(out_dir + "/" + split + ".label", "w")
index_fp = open(out_dir + "/" + split + ".index", "w")

with open(in_file, "r") as in_fp:
  for line in tqdm(in_fp.readlines()):
    struct = jsondecoder.decode(line)

    hypothesis = struct["claim"]

    premise_idx = 0
    for sentence in struct["predicted_sentences"]:
      underlined_title = sentence[0]
      label = 0   # placeholder, but must be a valid index
      premise = sentence[3]

      # Prefix the premise sentence with [ TITLE ] (from source article)
      title = underlined_title.replace("_", " ")
      title_words = tokenizer.split_words(title)
      tokenized_title = " ".join(map(lambda x: x.text, title_words))
      premise = "[ " + tokenized_title + " ] " + premise

      premise_words = premise.split(" ")
      if(len(premise_words) > max_sent_len):
        premise = " ".join(premise_words[0:max_sent_len])

      info = str(struct["id"]) + "\t" + str(premise_idx) + "\t"
      info = info + str(sentence[0]) + "\t" + str(sentence[1])

      premise_fp.write(premise + "\n")
      hypothesis_fp.write(hypothesis + "\n")
      label_fp.write(str(label) + "\n")
      index_fp.write(info + "\n")

      premise_idx = premise_idx + 1

premise_fp.close()
hypothesis_fp.close()
label_fp.close()
index_fp.close()

