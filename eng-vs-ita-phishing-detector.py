# =========================
# Kaggle Notebook: Fine-tune on fine-tuning.csv + test on test.csv (two separate files)
# - Loads fine-tuning.csv as training data
# - Fine-tunes XLM-R
# - Loads test.csv as external test data
# - Outputs predictions + (if labels exist) accuracy/precision/recall/F1
# =========================

# If Kaggle Internet is OFF and you already have these installed, you can delete this line.
!pip -q install transformers datasets accelerate scikit-learn

# ---- Force single GPU to avoid multi-GPU DataParallel replica issues on Kaggle T4 x2 ----
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # force only one GPU 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # use both GPU 

import inspect
import logging
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# CONFIG: CHOOSING MODEL AND SETTING THE PATHS
# -------------------------
MODEL_NAME = "xlm-roberta-base"
MODEL_NAME = "distilbert-base-multilingual-cased"

# setting Kaggle paths to files (training and test), valid only for Kaggle, if running on local, update the paths:
TRAIN_CSV = "/kaggle/input/datasets/bojanpoletan/fine-tuning-en/fine-tuning-eng.csv"    # training set  in english
#TRAIN_CSV  = "/kaggle/input/datasets/bojanpoletan/trining-ita/training-ita.csv"        # training set in italian
TEST_CSV  = "/kaggle/input/datasets/bojanpoletan/test-dataset/phishing-test-finale_cleaned.csv"    # test set in italian

# Column names expected 
FROM_COL = "From"
SUBJECT_COL = "Subject"
BODY_COL = "Body"
LABEL_COL_TRAIN = "label"   # phishing emails are labeled as "1" in the dataset
LABEL_COL_TEST = "label"    # phishing emails are labeled as "1" in the dataset

# Outputs - valid only for Kaggle, if running on local, update the paths
OUTPUT_DIR = "/kaggle/working/phishing_model"
FINAL_DIR  = "/kaggle/working/phishing_model/final_model"
PRED_TEST_CSV = "/kaggle/working/test_predictions.csv"

SEED = 42
MAX_LENGTH = 512

def _training_args(**kwargs):
    # only kwargs supported by this transformers version.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filtered)

# merge the dataset columns into one row
def combine_text_from_row(frm, subj, body):
    frm = "" if frm is None else str(frm)
    subj = "" if subj is None else str(subj)
    body = "" if body is None else str(body)
    return f"From: {frm}\nSubject: {subj}\nBody: {body}"

# print metrix at the end of the execution
def print_metrics(y_true, y_pred, title="METRICS"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n--- {title} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("--------------------\n")


def main():
    # -------------------------
    # ENVIROMENT CHECK
    # -------------------------
    print("===== ENV CHECK =====")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("GPU:", torch.cuda.get_device_name(0))
    print("=====================")

    print(f"\nUsing model: {MODEL_NAME}")

    # =========================
    # 1) LOAD TRAINING FILE (more details in "CONFIG" section)
    # =========================
    print(f"\nLoading training CSV: {TRAIN_CSV}")
    train_raw = load_dataset("csv", data_files=TRAIN_CSV)["train"]

    required_train = [FROM_COL, SUBJECT_COL, BODY_COL, LABEL_COL_TRAIN]
    missing_train = [c for c in required_train if c not in train_raw.column_names]
    if missing_train:
        raise ValueError(f"fine-tuning.csv missing columns: {missing_train}. Found: {train_raw.column_names}")

    # make sure the file labels are 0s or 1s
    train_raw = train_raw.map(lambda x: {LABEL_COL_TRAIN: int(x[LABEL_COL_TRAIN])})

    # (optional) split from training data for in-training evaluation
    # If the environment supports stratify_by_column, it will keep class ratios stable.
    try:
        train_split = train_raw.train_test_split(test_size=0.1, seed=SEED, stratify_by_column=LABEL_COL_TRAIN)
        print("Created internal validation split (10%) with stratification.")
    except Exception as e:
        print(f"Stratified internal split not available ({e}). Using random internal split.")
        train_split = train_raw.train_test_split(test_size=0.1, seed=SEED)

    # Build "text"
    def _combine_train(ex):
        return {"text": combine_text_from_row(ex[FROM_COL], ex[SUBJECT_COL], ex[BODY_COL])}

    train_split = train_split.map(_combine_train)

    # Rename label -> labels for Trainer
    if LABEL_COL_TRAIN != "labels":
        train_split = train_split.rename_column(LABEL_COL_TRAIN, "labels")

    # Remove original cols from model input
    train_split = train_split.remove_columns([FROM_COL, SUBJECT_COL, BODY_COL])

    # Tokenizer + tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LENGTH)

    tokenized_train = train_split.map(_tok, batched=True, remove_columns=["text"])
    tokenized_train.set_format("torch")

    if "input_ids" not in tokenized_train["train"].column_names:
        raise ValueError("Tokenization failed: 'input_ids' missing in tokenized training set.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # TrainingArguments API differences (evaluation_strategy vs eval_strategy, set the one that is actually used in this version)
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in ta_sig else ("eval_strategy" if "eval_strategy" in ta_sig else None)

    args_kwargs = dict(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
        disable_tqdm=True,
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        save_total_limit=2,
    )
    if eval_key is not None:
        args_kwargs[eval_key] = "epoch"
        print(f"Using TrainingArguments.{eval_key}='epoch' for internal validation.")
    else:
        print("No eval_strategy/evaluation_strategy supported; training without per-epoch eval.")

    training_args = _training_args(**args_kwargs)

    # metrics during training (on internal validation split)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_train["test"] if eval_key is not None else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # mave fine-tuned model
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"Saved fine-tuned model to: {FINAL_DIR}")

    # =========================
    # 2) LOAD TEST FILE AND RUN INFERENCE
    # =========================
    print(f"\nLoading test CSV: {TEST_CSV}")
    test_raw = load_dataset("csv", data_files=TEST_CSV)["train"]

    required_test = [FROM_COL, SUBJECT_COL, BODY_COL]
    missing_test = [c for c in required_test if c not in test_raw.column_names]
    if missing_test:
        raise ValueError(f"test dataset is missing columns: {missing_test}. Found: {test_raw.column_names}")

    # Keep original columns for output CSV
    test_df = test_raw.to_pandas()

    # Build text for test set
    def _combine_test(ex):
        return {"text": combine_text_from_row(ex[FROM_COL], ex[SUBJECT_COL], ex[BODY_COL])}

    test_raw = test_raw.map(_combine_test)
    # Remove original cols for model input
    test_model_ds = test_raw.remove_columns([FROM_COL, SUBJECT_COL, BODY_COL])

    # If test labels exist and you want metrics, normalize + rename to labels
    has_test_labels = (LABEL_COL_TEST is not None) and (LABEL_COL_TEST in test_model_ds.column_names or LABEL_COL_TEST in test_df.columns)
    if has_test_labels and LABEL_COL_TEST in test_model_ds.column_names:
        test_model_ds = test_model_ds.map(lambda x: {LABEL_COL_TEST: int(x[LABEL_COL_TEST])})
        if LABEL_COL_TEST != "labels":
            test_model_ds = test_model_ds.rename_column(LABEL_COL_TEST, "labels")

    # Tokenize test
    tokenized_test = test_model_ds.map(_tok, batched=True, remove_columns=["text"])
    tokenized_test.set_format("torch")

    if "input_ids" not in tokenized_test.column_names:
        raise ValueError("Tokenization failed: 'input_ids' missing in tokenized test set.")

    # Predict
    print("\nPredicting on test dataset ...")
    pred_out = trainer.predict(tokenized_test)
    preds = np.argmax(pred_out.predictions, axis=-1)

    # Save predictions
    test_df["predicted_label"] = preds
    test_df.to_csv(PRED_TEST_CSV, index=False)
    print(f"Saved test predictions CSV: {PRED_TEST_CSV}")
    print(test_df[[FROM_COL, SUBJECT_COL, "predicted_label"]].head(10))

    # Metrics (if test has labels)
    if has_test_labels and LABEL_COL_TEST in test_df.columns:
        y_true = test_df[LABEL_COL_TEST].astype(int).to_numpy()
        y_pred = test_df["predicted_label"].astype(int).to_numpy()
        print_metrics(y_true, y_pred, title="TEST.CSV METRICS")
    else:
        print("\ntest dataset has no label column (or LABEL_COL_TEST=None), so metrics were not computed.")

except_block = None


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM: reduce per_device_train_batch_size (4→2) or max_length (512→256).")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Check TRAIN_CSV / TEST_CSV paths under /kaggle/input/...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
