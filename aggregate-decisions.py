import sys
import json
import os
import os.path
from operator import itemgetter
from tqdm import tqdm
import argparse

labels = ["SUPPORTS", "NOT ENOUGH INFO", "REFUTES"]
classifications = [[], [], []]
ntype = []

parser = argparse.ArgumentParser()
parser.add_argument("--original_jsonl", type=str, required=True)
parser.add_argument("--index_file", type=str, required=True)
parser.add_argument("--decision_file", type=str, required=True)
parser.add_argument("--submission_file", help="Output", type=str, required=True)
args = parser.parse_args()
original_jsonl = args.original_jsonl
index_file = args.index_file
decision_file = args.decision_file
submission_file = args.submission_file

jsonl_fp = open(original_jsonl, "r")
decision_fp = open(decision_file, "r")
index_fp = open(index_file, "r")

if(os.path.exists(submission_file)):
  raise ValueError("Submission file already exists")

out_fp = open(submission_file, "w")

order = []
outputs = {}
support_evidences = {}
refute_evidences = {}

jsondecoder = json.JSONDecoder()
jsonencoder = json.JSONEncoder()

for line in tqdm(jsonl_fp.readlines()):
  struct = jsondecoder.decode(line)
  qid = str(struct["id"])
  order.append(qid)   # Output answers in the same order as jsonl input
  outputs[qid] = 1    # Initialize to "not enough info, in case we
                      # didn't actually retrieve any sentences for this qid
  refute_evidences[qid]= []
  support_evidences[qid] = []

def update_class(qid, value, h):
  if(qid in h):
    if(value == 2):
      if(h[qid] != 0):
        h[qid] = value
    if(value == 0):
      h[qid] = value
  else:
    h[qid] = value

print("Collecting decisions...")
decision_hdr = decision_fp.readline()
for output in tqdm(decision_fp.readlines()):
  (n, output) = output.rstrip().split()
  output = int(output)
  index = index_fp.readline().rstrip()
  (qid, evidence_number, title, linenum) = index.split()

  if(output != 0 and output != 1 and output != 2):
    raise ValueError("Output format")
  if(not(qid in outputs)):
    raise ValueError("Answer for nonexistent question " + qid)

  update_class(qid, output, outputs)

  if(output == 2):
    refute_evidences[qid].append([title, int(linenum)])
  elif(output == 0):
    support_evidences[qid].append([title, int(linenum)])

print("Writing output...")
for qid in tqdm(order):
  output = outputs[qid]
  label = labels[output]
  struct = {"id": qid, "predicted_label": label}

  evidence = []
  if(output == 0):
    evidence = support_evidences[qid]
  elif(output == 2):
    evidence = refute_evidences[qid]

  # sort by line num then title (perform stable sorts in reverse order)
  evidence.sort(key=itemgetter(0))
  evidence.sort(key=itemgetter(1))

  answer = []
  for i in range(5):
    if(i < len(evidence)):
      answer.append(evidence[i])

  struct["predicted_evidence"] = answer
  out_fp.write(jsonencoder.encode(struct) + "\n")

