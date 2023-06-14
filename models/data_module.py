import lightning.pytorch as pl
import pandas as pd
import numpy as np
import torch
import ujson
import spacy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorForTokenClassification,
)
from tqdm import tqdm
from functools import partial
from logger import setup_logger

# Setup
sci_nlp = spacy.load("en_core_sci_sm")  # scispacy
logger = setup_logger()


# Helper Functions
def get_sent_boundaries(sci_nlp, text):
    """
    Returns char indices for start and end of sentences from the full text.
    """

    # start with scispacy's sentence splitting
    sents = [sent.text for sent in sci_nlp(text).sents]

    sent_span_idxs = []

    ch_idx = 0

    for sent in sents:
        start_idx = ch_idx
        end_idx = ch_idx + (len(sent) - 1)

        # move to next char, skips ws
        try:
            if text[end_idx + 1] in [" ", "\n"]:
                ch_idx = end_idx + 2
            else:  # happens when sentence splitting fails
                ch_idx = end_idx + 1
        except IndexError:  # end of text
            ch_idx = end_idx

        sent_span_idxs.append((start_idx, end_idx))

    return sent_span_idxs


def get_abstract_split(abstract, tokenizer, max_len=500):
    """
    Split abstract into chunks along sentence boundaries if longer than max_len
    """
    text = abstract["text"]
    boundaries = get_sent_boundaries(sci_nlp, text)
    split_inds = [0]
    for i, x in enumerate(boundaries):
        # print(x, split_inds)
        encoded = tokenizer(text[split_inds[-1] : x[1] + 1])
        input_ids = encoded["input_ids"]
        if len(input_ids) > max_len:
            split_inds.append(x[0])

    return split_inds


def update_offsets(abstract, split_inds):
    """
    Update character offsets of annotations afte splitting abstract into chunks
    """
    chunks = []
    spans = abstract["spans"]
    num_splits = len(split_inds)
    for i in range(num_splits):
        chunk = {"chunk_id": i}
        chunk_spans = []
        split_offset = split_inds[i]
        if i + 1 >= num_splits:
            split_end = 9999999
        else:
            split_end = split_inds[i + 1]
        chunk["pmid"] = abstract["pmid"]
        # print("offsets", split_offset, split_end)
        chunk["text"] = abstract["text"][split_offset:split_end]
        # print(chunk["text"])
        for span in spans:
            if (span["start"] < split_offset) or (span["end"] > split_end):
                continue
            else:
                chunk_spans.append(
                    {
                        "start": span["start"] - split_offset,
                        "end": span["end"] - split_offset,
                        "tag": span["tag"],
                        "label": span["label"],
                        "text": span["text"],
                    }
                )

        chunk["spans"] = chunk_spans
        chunks.append(chunk)

    return chunks


class NERDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        for x in data:
            if "pmid" not in x:
                print(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["text"], torch.LongTensor(
            self.data[index]["token_labels"]
        )


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        hf_model: str,
        batch_size: int = 32,
        max_length=500,
        debug=False,
    ):
        logger.info(f"Loading Tokenizer: {hf_model}")
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.max_len = max_length
        self.debug = debug

    def setup(self):
        """
        Load data:
            Data will be in form
            {'text':text, (str)
             'spans':spans, (list of dict)
             'pmid':pmid, (int)
             'split':split (str)
            }

            spans will have form:
            {'start':int,
             'end':int,
             'label':int,
             'tag':str,
             'text':str
            }

        Once loaded, do the following:
            * Split abstracts that are too long into chunks
            * Update offsets after splitting abstracts
            * Get token-level labels from spans
            * Create train, validation, and test pytorch Datasets
        """
        # Load data
        logger.info("Loading data")
        self._load_data()

        # Make sure dataset has appropriate splits
        for x in ["train", "validation", "test"]:
            assert x in self.split_names

        # make mapping of tags to label_ids
        iter = 1
        label2id = {"O": 0}
        for tag in self.span_tags:
            label2id[f"B-{tag}"] = iter
            label2id[f"I-{tag}"] = iter + 1
            iter += 2
        self.label2id = label2id
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Add token labels to update mapping of token tags
        for split in self.split_names:
            updated_examples = []
            for ex in self.split_data[split]:
                ex["token_labels"] = [self.label2id[x] for x in ex["token_tags"]]
                updated_examples.append(ex)
                # print(ex)
                # print(self.label2id)

            self.split_data[split] = updated_examples

        self.train = NERDataset(self.split_data["train"])
        self.test = NERDataset(self.split_data["test"])
        self.validation = NERDataset(self.split_data["validation"])

        # Custom collate function
        self.collate_fn = partial(
            token_classification_collate_fn, tokenizer=self.tokenizer
        )

        logger.info("Data Setup Complete")

    def _load_data(self):
        """
        Load in data for NER
        """
        data = ujson.load(open(self.data_path))
        if self.debug:
            data = data[:20]
        split_data = defaultdict(list)
        span_tags = set([])
        split_names = set([])

        # Split abstracts into smaller chunks if necessary
        logger.info(
            "Splitting abstracts into chunks (if needed) and getting token-level labels"
        )
        for a in tqdm(data):
            split_inds = get_abstract_split(a, self.tokenizer, max_len=self.max_len)
            if len(split_inds) > 1:
                chunks = update_offsets(a, split_inds)

                # Make sure remapings were performed corectly
                for chunk in chunks:
                    text = chunk["text"]
                    for span in chunk["spans"]:
                        span_tags.add(span["tag"])
                        assert text[span["start"] : span["end"]] == span["text"]
            else:
                for span in a["spans"]:
                    span_tags.add(span["tag"])
                chunks = [a]

            # logger.info('Getting token-level tags for each chunk')
            for chunk in chunks:
                token_tags = self.get_token_labels_from_char_spans(chunk)
                chunk["token_tags"] = token_tags

            split_data[a["split"]].extend(chunks)
            split_names.add(a["split"])

        self.split_names = split_names
        self.split_data = split_data
        self.span_tags = span_tags

    def get_token_labels_from_char_spans(self, span_data: Dict[str, any]) -> List[str]:
        # Tokenize the input text and get the offset mapping
        tokenized = self.tokenizer(span_data["text"], return_offsets_mapping=True)

        # Create an empty list of tags, one for each token
        tags = ["O"] * len(tokenized["input_ids"])

        # Go through each span in the data
        for span in span_data["spans"]:
            span_started = False
            # Go through each offset mapping
            for i, (start, end) in enumerate(tokenized["offset_mapping"]):
                # Skip special tokens
                if start == end == 0:
                    continue

                # If the start of the token is inside the span, start the span
                if start >= span["start"] and start < span["end"]:
                    if not span_started:
                        tags[i] = f"B-{span['tag']}"
                        span_started = True
                    else:
                        tags[i] = f"I-{span['tag']}"

                # If the token is inside the span, continue the span
                elif start < span["start"] and end > span["start"]:
                    tags[i] = f"B-{span['tag']}"
                    span_started = True

        return tags

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=self.collate_fn
        )


def token_classification_collate_fn(batch, tokenizer):
    """
    Collate function for token classification
    """
    text, labels = zip(*batch)
    output_dict = tokenizer(text, padding="longest", return_tensors="pt")
    output_dict["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
    return output_dict
