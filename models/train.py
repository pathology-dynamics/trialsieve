"""
Train and evaluate four different NER models on TrialSieve
"""

import numpy as np
import evaluate
import warnings
import ujson
import os

# Ignore all warnings
warnings.filterwarnings("ignore")

from data_module import NERDataset, NERDataModule, token_classification_collate_fn

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from functools import partial
from logger import setup_logger
from transformers.integrations import WandbCallback

# Load package to evaluate models
seqeval = evaluate.load("seqeval")
logger = setup_logger()

# Models to evaluate
models = [
    "michiyasunaga/BioLinkBERT-base",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
]

# Set up logdir
if not os.path.isdir("logs"):
    os.mkdir("logs")


def compute_metrics(p, id2label, mode="eval"):
    """
    Compute precision, recall, and f1 for each class and overall
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # print(results)

    class_specific_f1 = {
        k: v["f1"] for k, v in results.items() if not k.startswith("overall")
    }

    if mode == "eval":
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "class_specific_f1": class_specific_f1,
            "detailed_results": results,
        }
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "class_specific_results": results,
        }


for hf_model in models:
    # Set up dataloaders
    data_module = NERDataModule(
        "data/preprocessed_for_modeling.json",
        hf_model=hf_model,
        debug=False,
    )
    data_module.setup()
    train = data_module.train
    validation = data_module.validation
    test = data_module.test
    tokenizer = data_module.tokenizer

    # Map each numeric label to its human-readable label
    id2label = data_module.id2label
    label2id = data_module.label2id
    num_labels = len(id2label)

    # Data collator
    collate_fn = partial(token_classification_collate_fn, tokenizer=tokenizer)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        hf_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # Train model
    training_args = TrainingArguments(
        output_dir="outputs",
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,
        weight_decay=0.01,
        warmup_ratio=0.2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="validation_f1",
        logging_strategy="epoch",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset={"validation": validation},
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, id2label=id2label),
    )

    trainer.train()

    # Predict output and save metrics
    output = trainer.predict(test)
    logger.info(output.metrics)
    model_name = hf_model.split("/")[-1]
    with open(f"logs/{model_name}.json", "w") as f:
        f.write(ujson.dumps(output.metrics, indent=2))
