import unicodedata
import json
import re
import sys
import os
import argparse

# from retrieval.fever_doc_db import FeverDocDB

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, required=True)
parser.add_argument("--out_file", type=str, required=True)
parser.add_argument("--fever_pages_dir", help="Wikipedia dump", type=str, required=True)
args = parser.parse_args()
in_file = args.in_file
out_file = args.out_file
fever_pages_dir = args.fever_pages_dir

if os.path.exists(out_file):
  raise ValueError("Output already exists")

fever_pages = [ os.path.join(fever_pages_dir, f)
                for f in os.listdir(fever_pages_dir)
                if re.search("\.jsonl$", f) ]
fever_pages = sorted(fever_pages)

# fever_db = FeverDocDB(fever_db_file)

needed = {}
found = {}

jsondecoder = json.JSONDecoder()
jsonencoder = json.JSONEncoder()

with open(in_file, "r") as in_fp:
  for line in in_fp:
    struct = jsondecoder.decode(line)
    for finding in struct["predicted_sentences"]:
      title = unicodedata.normalize('NFD', str(finding[0]))
      linenum = finding[1]
      if(not(title in needed)):
        needed[title] = []
      needed[title].append(linenum)

for pages in fever_pages:
  print("Processing " + pages)
  with open(pages, "r") as in_fp:
    for line in in_fp:
      struct = json.JSONDecoder().decode(line)
      title = unicodedata.normalize('NFD', str(struct["id"]))
      if title in needed:
        found[title] = {}
        lines = struct["lines"].split("\n")
        linenum = 0
        for linerecord in lines:
          fields = linerecord.split("\t")
          if(linenum in needed[title]):
            if(len(fields) < 2):
              print("Problem retrieving from "+title+" line "+str(linenum))
              found[title][linenum] = "This sentence is intentionally left blank ."
            else:
              textline = fields[1]
              found[title][linenum] = textline
          linenum = linenum + 1

print("Filling in answers")

with open(in_file, "r") as in_fp:
  with open(out_file, "w") as out_fp:
    for line in in_fp:
      struct = jsondecoder.decode(line)

      skip = False
      supports = {}
      for eg in struct["evidence"]:
        if(len(eg) != 1):
          skip = True  # multiple supporting statements required
        title = unicodedata.normalize('NFD', str(eg[0][2]))
        if(not(title in supports)):
          supports[title] = []
        supports[title].append(eg[0][3])  # line number

      if(skip == False):
        for finding in struct["predicted_sentences"]:
          title = unicodedata.normalize('NFD', str(finding[0]))
          linenum = finding[1]
          if(finding[0] is None):
            pass
          elif(not(title in found)):
            print("Page not found: " + title)
          elif(not(linenum in found[title])):
            print("Line not found: " + title + " " + str(linenum))
          else:
            sentence = found[title][linenum]
            if(len(finding) != 2):
              print("Bad finding length: " + title + " " + str(linenum))

            label = "NOT ENOUGH INFO"
            if(title in supports):
              if(linenum in supports[title]):
                label = struct["label"] # REFUTES or SUPPORTS

            finding.append(label)
            finding.append(sentence)

        result = jsonencoder.encode(struct)
        out_fp.write(result + "\n")
  
