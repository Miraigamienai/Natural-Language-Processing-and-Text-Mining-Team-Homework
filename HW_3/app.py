import os
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)
def get_base_dir():
    try: base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: base_dir = os.getcwd()
    return base_dir

# CONFIG
BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "BTC", "CONLL-format", "data")
MODEL_NAME = "bert-base-cased"

BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

# LABELS
label_list = [
    "O",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
    "B-PER",
    "I-PER",
]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# READ CONLL
def read_conll(path):

    sentences = []
    labels = []

    words = []
    tags = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # sentence end
            if line == "":
                if len(words) > 0:
                    sentences.append(words)
                    labels.append(tags)
                words = []
                tags = []
            else:
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                word, tag = parts
                words.append(word)
                tags.append(tag)
    return sentences, labels

# LOAD ALL BTC FILES
all_sentences = []
all_labels = []
files = [
    "a.conll",
    "b.conll",
    "e.conll",
    "f.conll",
    "g.conll",
    "h.conll"
]

for file in files:
    path = os.path.join(DATA_DIR, file)
    sents, labs = read_conll(path)
    all_sentences.extend(sents)
    all_labels.extend(labs)

print("Total sentences:", len(all_sentences))

# TRAIN / TEST SPLIT
from sklearn.model_selection import train_test_split
train_tokens, test_tokens, train_labels, test_labels = train_test_split(
    all_sentences,
    all_labels,
    test_size=0.2,
    random_state=42
)
train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_tokens,
    train_labels,
    test_size=0.1,
    random_state=42
)

# DATASET
dataset = DatasetDict({
    "train": Dataset.from_dict({
        "tokens": train_tokens,
        "ner_tags": train_labels
    }),

    "validation": Dataset.from_dict({
        "tokens": val_tokens,
        "ner_tags": val_labels
    }),

    "test": Dataset.from_dict({
        "tokens": test_tokens,
        "ner_tags": test_labels
    }),
})

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TOKENIZE + ALIGN LABELS
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # special tokens
            if word_idx is None:
                label_ids.append(-100)
            # first subword
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # remaining subwords
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True
)

# MODEL
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# METRICS
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        current_preds = []
        current_labels = []
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                current_preds.append(id2label[p_])
                current_labels.append(id2label[l_])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    print("\nClassification Report:\n")
    print(classification_report(true_labels, true_predictions))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LR,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    # logging_dir="./logs",

    load_best_model_at_end=True,
)

# TRAINER
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# TRAIN
trainer.train()

# TEST
metrics = trainer.evaluate(tokenized_dataset["test"])

print("\n===== FINAL RESULT =====")

print(f"Precision : {metrics['eval_precision']:.4f}")
print(f"Recall    : {metrics['eval_recall']:.4f}")
print(f"F-measure : {metrics['eval_f1']:.4f}")
print(f"Accuracy  : {metrics['eval_accuracy']:.4f}")

# SAVE MODEL
trainer.save_model(os.path.join(BASE_DIR, "btc_ner_model"))

print("\nModel saved.")