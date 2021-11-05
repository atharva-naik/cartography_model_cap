"""
Utilities for data handling.
Reading datasets from tsv and jsonl files.
"""
import os
from pathlib import Path
# comment this out except for KGP servers.
os.environ['OPENBLAS_NUM_THREADS'] = "12"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)

import torch
import shutil
import argparse
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import os, re, json, logging, random
from typing import Dict, Union, List
from torch.utils.data import Dataset, DataLoader
# from cartography.data_utils_glue import read_glue_tsv
logger = logging.getLogger(__name__)
try:
    # utilites for getting/printing args etc..
    from cartography_adapters.utils import *
except ImportError:
    from utils import *

def convert_string_to_unique_number(string: str) -> int:
  """
  Hack to convert SNLI ID into a unique integer ID, for tensorizing.
  """
  id_map = {'e': '0', 'c': '1', 'n': '2'}

  # SNLI-specific hacks.
  if string.startswith('vg_len'):
    code = '555'
  elif string.startswith('vg_verb'):
    code = '444'
  else:
    code = '000'

  try:
    number = int(code + re.sub(r"\D", "", string) + id_map.get(string[-1], '3'))
  except:
    number = random.randint(10000, 99999)
    logger.info(f"Cannot find ID for {string}, using random number {number}.")
  return number


def read_glue_tsv(file_path: str,
                  guid_index: int,
                  label_index: int = -1,
                  guid_as_int: bool = False):
  """
  Reads TSV files for GLUE-style text classification tasks.
  Returns:
    - a mapping between the example ID and the entire line as a string.
    - the header of the TSV file.
  """
  tsv_dict = {}

  i = -1
  with open(file_path, 'r') as tsv_file:
    for line in tqdm.tqdm([line for line in tsv_file]):
      i += 1
      if i == 0:
        header = line.strip()
        field_names = line.strip().split("\t")
        continue

      fields = line.strip().split("\t")
      label = fields[label_index]
      if len(fields) > len(field_names):
        # SNLI / MNLI fields sometimes contain multiple annotator labels.
        # Ignore all except the gold label.
        reformatted_fields = fields[:len(field_names)-1] + [label]
        assert len(reformatted_fields) == len(field_names)
        reformatted_line = "\t".join(reformatted_fields)
      else:
        reformatted_line = line.strip()

      if label == "-" or label == "":
        logger.info(f"Skippping line: {line}")
        continue

      if guid_index is None:
        guid = i
      else:
        guid = fields[guid_index] # PairID.
      if guid in tsv_dict:
        logger.info(f"Found clash in IDs ... skipping example {guid}.")
        continue
      tsv_dict[guid] = reformatted_line.strip()

  logger.info(f"Read {len(tsv_dict)} valid examples, with unique IDS, out of {i} from {file_path}")
  if guid_as_int:
    tsv_numeric = {int(convert_string_to_unique_number(k)): v for k, v in tsv_dict.items()}
    return tsv_numeric, header
  return tsv_dict, header

def read_data(file_path: str,
              task_name: str,
              guid_as_int: bool = False):
  """
  Reads task-specific datasets from corresponding GLUE-style TSV files.
  """
  logger.warning("Data reading only works when data is in TSV format, "
                 " and last column as classification label.")

  # `guid_index`: should be 2 for SNLI, 0 for MNLI and None for any random tsv file.
  if task_name == "MNLI":
    return read_glue_tsv(file_path,
                        guid_index=0,
                        guid_as_int=guid_as_int)
  elif task_name == "SNLI":
    return read_glue_tsv(file_path,
                        guid_index=2,
                        guid_as_int=guid_as_int)
  elif task_name == "WINOGRANDE":
    return read_glue_tsv(file_path,
                        guid_index=0)
  elif task_name == "QNLI":
    return read_glue_tsv(file_path,
                        guid_index=0)
  else:
    raise NotImplementedError(f"Reader for {task_name} not implemented.")


def convert_tsv_entries_to_dataframe(tsv_dict: Dict, header: str) -> pd.DataFrame:
  """
  Converts entries from TSV file to Pandas DataFrame for faster processing.
  """
  header_fields = header.strip().split("\t")
  data = {header: [] for header in header_fields}

  for line in tsv_dict.values():
    fields = line.strip().split("\t")
    assert len(header_fields) == len(fields)
    for field, header in zip(fields, header_fields):
      data[header].append(field)

  df = pd.DataFrame(data, columns=header_fields)
  return df


def copy_dev_test(task_name: str,
                  from_dir: os.path,
                  to_dir: os.path,
                  extension: str = ".tsv"):
  """
  Copies development and test sets (for data selection experiments) from `from_dir` to `to_dir`.
  """
  if task_name == "MNLI":
    dev_filename = "dev_matched.tsv"
    test_filename = "dev_mismatched.tsv"
  elif task_name in ["SNLI", "QNLI", "WINOGRANDE"]:
    dev_filename = f"dev{extension}"
    test_filename = f"test{extension}"
  else:
    raise NotImplementedError(f"Logic for {task_name} not implemented.")

  dev_path = os.path.join(from_dir, dev_filename)
  if os.path.exists(dev_path):
    shutil.copyfile(dev_path, os.path.join(to_dir, dev_filename))
  else:
    raise ValueError(f"No file found at {dev_path}")

  test_path = os.path.join(from_dir, test_filename)
  if os.path.exists(test_path):
    shutil.copyfile(test_path, os.path.join(to_dir, test_filename))
  else:
    raise ValueError(f"No file found at {test_path}")


def read_jsonl(file_path: str, key: str = "pairID"):
  """
  Reads JSONL file to recover mapping between one particular key field
  in the line and the result of the line as a JSON dict.
  If no key is provided, return a list of JSON dicts.
  """
  df = pd.read_json(file_path, lines=True)
  records = df.to_dict('records')
  logger.info(f"Read {len(records)} JSON records from {file_path}.")

  if key:
    assert key in df.columns
    return {record[key]: record for record in records}
  return records

def read_jsonl(path, key="pairID", 
               class_key='gold_label', filt_class="-"):
    '''handle duplicates, filter out '-' class, encode pairIDs'''
    import json
    data = {}
    f = open(path)
    for line in tqdm(f, desc="reading raw data"):
        line = line.strip()
        item = json.loads(line)
        # ignore filt_class datapoints.
        if item[class_key] == filt_class: 
            continue
        # ignore duplicate keys
        id = item[key]
        data[item[key]] = item
        # encode pairIds
        if key == "pairID":
            data[id][key] = convert_string_to_unique_number(id)

    f.close()
    data = list(data.values())
            
    return data

# Namespaces

## Namespaces for dataset structure info.
Keys = {}
Keys["MNLI"] = argparse.Namespace()
Keys["MNLI"].key = "pairID"
Keys["MNLI"].class_key = "gold_label"
Keys["MNLI"].label_map = {"neutral": 0, "entailment": 1, "contradiction": 2}
Keys["SNLI"] = argparse.Namespace()
Keys["SNLI"].key = "pairID"
Keys["SNLI"].class_key = "gold_label"
Keys["SNLI"].label_map = {"neutral": 0, "entailment": 1, "contradiction": 2}

## Namespace for tokenization config.
TokParams = argparse.Namespace()
### tokenization paramters.
TokParams.padding = "max_length" # max length strategy for padding (pad to fixed length)
TokParams.max_length = 100 # the max length for padding.
TokParams.add_special_tokens=True # add special tokens like [CLS], [SEP]
'''
**Longest first strategy as described in the Huggingface Docs:**

Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
'''
TokParams.truncation = True # same a `longest_first`

class GLUEDataset(Dataset):
    '''class to read jsonl data.'''
    def __init__(self, path: Union[str, Path], 
                 tokenizer, task_name="MNLI",
                 tok_params=TokParams):
        # keys for accesing attributes of dataset.
        self.task_name = task_name
        print("Running Task:", task_name)
        self.tok_params = tok_params
        print("Tokenization Parameters:")
        pprint_args(self.tok_params)
        self.tokenizer = tokenizer
        print("Dataset Structure:")
        self.keys = Keys.get(task_name)
        pprint_args(self.keys)
        if self.keys is None:
            exit("task is not supported!!")
        # caching paths
        subset = Path(path).stem
        self.cache_dir = os.path.join(CURRENT_DIR, f"../{task_name}_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{subset}.pkl")
        # print(self.cache_dir, self.cache_path)
        # load features from cache if it is cached.
        if os.path.exists(self.cache_path):
            self.proc_data = pkl.load(open(self.cache_path, "rb"))
        # cache features for quicker loading times in the future.
        else:
            data = read_jsonl(path, key=self.keys.key, 
                              class_key=self.keys.class_key)
            self.proc_data = []
            for item in tqdm(data, desc="encoding features"):
                self.proc_data.append(
                    self._preproc(item)
                )
            with open(self.cache_path, "wb") as f:
                pkl.dump(self.proc_data, f)
        # if file.endswith(".jsonl"):
        #     self.data = read_jsonl(file_path, key=key)
        # elif file.endswith(".tsv"):
        #     self.data = pd.read_csv(file_path, sep="\t")
        #     self.data = self.data.to_records()
    def __len__(self):
        return len(self.proc_data)
    
    def _preproc(self, record_item: dict):
        id = record_item[self.keys.key]
        s1 = record_item["sentence1"]
        s2 = record_item["sentence2"]
        tok_dict = self.tokenizer(
            s1, s2,
            **vars(self.tok_params)
        )
        iids = tok_dict["input_ids"]
        attn_mask = tok_dict["attention_mask"]
        # token type ids (useful only for BERT, ignored for roberta). For roberta a placeholder of all zeros is used.
        tok_typ_ids = tok_dict.get("token_type_ids", [0]*len(iids))
        label = self.keys.label_map.get(
            record_item[self.keys.class_key], 
            len(self.keys.label_map)+1
        )

        return {
            "id": id,
            "input_ids": iids,
            "attention_mask": attn_mask,
            "token_type_ids": tok_typ_ids,
            "label": label,
        }
    
    def __getitem__(self, index):
        tok_dict = self.proc_data[index]
        tensor_dict = {k: torch.as_tensor(v) for k,v in tok_dict.items()}
        
        return tensor_dict
    
    
if __name__ == "__main__":
    import time
    import transformers
    from transformers import RobertaTokenizer
    transformers.logging.set_verbosity_error()
    
    # print("HERE 1")
    tokenizer = RobertaTokenizer.from_pretrained("../roberta-base-tok") 
    # RobertaTokenizer.from_pretrained("roberta-base")
    # print("HERE 2")
    # tokenizer.save_pretrained("../roberta-base-tok")
    # print("HERE 3")
    s = time.time()
    trainset = GLUEDataset(
        path="./data/MNLI/original/multinli_1.0_train.jsonl",
        tokenizer=tokenizer, 
        task_name="MNLI",
    )
    print(f"dataset loaded in {time.time()-s}s")
    # print("HERE 4")
    for i in tqdm(range(len(trainset))):
        id, iids, attn_mask, tok_typ_ids, label = trainset[i]
        if id == 591262: break
    print(id, iids, attn_mask, tok_typ_ids, label)
    trainloader = DataLoader(trainset, batch_size=32, num_workers=4, shuffle=False)
    from pprint import pprint
    for batch in trainloader:
        pprint(batch)
        break
    # total iteration time.
    s = time.time()
    for batch in tqdm(trainloader):
        pass
    print(f"dataloader iteration took {time.time()-s}s")