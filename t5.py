from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
from transformers import AutoTokenizer

billsum = load_dataset("billsum", split="ca_test")

billsum = billsum.train_test_split(test_size=0.2)


checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


training_args = Seq2SeqTrainingArguments(
    output_dir="model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    #fp16=True,
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Print debug information
# print(f"Output directory: {training_args.output_dir}")
# print(f"Number of training epochs: {training_args.num_train_epochs}")

trainer.train()

# text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
#
#
# summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
# summarizer(text)
#
#
#
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids
#
#
# model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
# outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
#
# tokenizer.decode(outputs[0], skip_special_tokens=True)