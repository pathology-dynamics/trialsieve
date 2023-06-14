"""
Format annotations for model training
"""

import ujson
import pandas as pd
import numpy as np
from tqdm import tqdm


def find_consecutive_spans(arr, mask):
    """
    Inputs:
        arr: 1-d numpy array from which repeated values should be considered
        mask: mask on whether to include index in returned spans
    Returns:
        spans: List of tuples containing (start, end, val) of each span of  repeated integers
    """
    if len(arr) != len(mask):
        raise ValueError("Array and mask lengths do not match.")

    # initialization
    spans = []
    start, end, val = None, None, None

    for i, (x, m) in enumerate(zip(arr, mask)):
        if m:  # if the mask is True
            if val is None:  # start of a new span
                start, end, val = i, i + 1, x
            elif x == val:  # continuation of a span
                end = i + 1
            else:  # end of a span due to different value
                spans.append((start, end, val))
                start, end, val = i, i + 1, x
        else:  # if the mask is False, we end the span
            if val is not None:  # there was an open span
                spans.append((start, end, val))
                start, end, val = None, None, None

    # don't forget the last span
    if val is not None:
        spans.append((start, end, val))

    return spans


# Load in annotations
clean_annotations = pd.read_csv(
    "data/final_schema_data.csv", na_filter=False
).convert_dtypes()
abstracts = pd.read_csv("data/pmid_title_abstract.csv")
clean_annotations["annotation_length"] = clean_annotations["value"].map(
    lambda x: len(x.split())
)


# Get relevant mappings
class_counts = clean_annotations.tag.value_counts()
class2id = {x: i for i, x in enumerate(class_counts.index)}
id2class = {i: x for x, i in class2id.items()}
pmid2text = {x["pmid"]: x["text"] for x in abstracts.to_dict(orient="records")}


# Group by document
grouped = clean_annotations.groupby("pmid")
examples_for_modeling = []
total_spans = 0

# Find high-quality spans at character level
for pmid, group in tqdm(grouped):
    text = pmid2text[pmid]
    max_end = group.end.max()
    char_tag_counts = np.zeros((len(class2id), max_end)).astype(int)

    # Get splits of abstracts
    split = "train"
    rv = np.random.rand()
    if rv < 0.3:
        split = "validation"
    if rv < 0.15:
        split = "test"

    for row in group.to_dict(orient="records"):
        ind = class2id[row["tag"]]
        start = row["start"]
        end = row["end"]

        char_tag_counts[ind, start:end] += 1
        if row["correct"]:
            char_tag_counts[ind, start:end] += 1
        elif row["correct"] == False:
            char_tag_counts[ind, start:end] -= 1

    best_ind = char_tag_counts.argmax(axis=0)
    best_val = char_tag_counts.max(axis=0)

    # Get consecutive character spans with class agreement
    consecutive_spans = find_consecutive_spans(best_ind, best_val > 1)
    labeled_spans = [
        {
            "start": x[0],
            "end": x[1],
            "label": int(x[2]),
            "tag": id2class[x[2]],
            "text": text[x[0] : x[1]],
        }
        for x in consecutive_spans
    ]

    # Fix common errors in quantitative measurements
    filtered_spans = []
    for s in labeled_spans:
        if s["tag"] == "Quantitative Measurement":
            # print(type(s['label']))
            if s["text"].lower() in [
                "hazard ratio",
                "95% ci",
                "odds ratio",
                "risk ratio",
                "OR",
                "HR",
                "RR",
            ]:
                s["tag"] = "Type of Quant. Measure"
                s["label"] = class2id[s["tag"]]
                total_spans += 1
            elif (
                len(
                    s["text"]
                    .replace("of", "")
                    .replace("percent", "")
                    .strip("0123456789+-.%,/ ")
                    .strip()
                )
                > 0
            ):
                # print(s['text'])
                continue
            else:
                total_spans += 1
        else:
            total_spans += 1
        filtered_spans.append(s)

    examples_for_modeling.append(
        {"text": text, "spans": filtered_spans, "pmid": pmid, "split": split}
    )


# Write to file
ujson.dump(examples_for_modeling, open("data/preprocessed_for_modeling.json", "w"))
