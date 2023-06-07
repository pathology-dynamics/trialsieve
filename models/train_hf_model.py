# %load_ext autoreload
# %autoreload 2


import numpy as np
import evaluate
import warnings
import pickle
import os
import ujson

# Ignore all warnings
warnings.filterwarnings("ignore")

from data_module import NERDataset, NERDataModule, token_classification_collate_fn

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from functools import partial
from logger import setup_logger
from transformers.integrations import WandbCallback

seqeval = evaluate.load("seqeval")
logger = setup_logger()

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
models = [
    "michiyasunaga/BioLinkBERT-base",
    "dmis-lab/biobert-base-cased-v1.2",
    # # "kamalkraj/BioELECTRA-PICO",
    "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
]

# wandb = WandbCallback().setup()


def compute_metrics(p, id2label, mode="eval"):
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


# class CustomTrainer(Trainer):
#     def log_metrics(split, metrics)

# hf_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

for hf_model in models:
    # data_module_path = "data_modules/data_module_v1.pickle"
    # if os.path.isfile(data_module_path):
    #     data_module = pickle.load(open(data_module_path, "rb"))
    # else:
    data_module = NERDataModule(
        "../data/preprocessed_for_modeling.json",
        hf_model=hf_model,
        debug=False,
    )
    data_module.setup()
    # pickle.dump(data_module, open("data_module_path", "wb"))
    # train_loader = data_module.train_dataloader()
    train = data_module.train
    validation = data_module.validation
    test = data_module.test
    tokenizer = data_module.tokenizer

    id2label = data_module.id2label
    label2id = data_module.label2id
    num_labels = len(id2label)
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    collate_fn = partial(token_classification_collate_fn, tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        hf_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

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
        # jit_mode_eval=True,
        logging_strategy="epoch",
        report_to="wandb"
        # device=0
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset={"validation": validation},
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, id2label=id2label),
        # callbacks=[wandb],
    )

    trainer.train()

    # Predict output and save metrics
    output = trainer.predict(test)
    logger.info(output.metrics)
    model_name = hf_model.split("/")[-1]
    # with open(f"../test_preds/{model_name}.pickle", "wb") as f:
    #     pickle.dump(output, f)
    with open(f"../logs/{model_name}.json", "w") as f:
        f.write(ujson.dumps(output.metrics, indent=2))
