import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
from bert_score import score as bert_score
import torch
torch.cuda.empty_cache()
model_name = 'facebook/bart-base'

# Load and preprocess dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    dialogues = df.groupby('dialogue_id')
    
    inputs = []
    outputs = []
    
    for dialogue_id, group in dialogues:
        turns = group.sort_values('turn_id')
        for _, row in turns.iterrows():
            if row['type'] == 'CN':
                # Get all previous turns in the dialogue
                context = turns[turns['turn_id'] < row['turn_id']]
                input_text = "\n".join(context['text'].tolist())
                inputs.append(input_text)
                outputs.append(row['text'])
    
    return pd.DataFrame({'input': inputs, 'output': outputs})

# Process data
data = preprocess_data('DIALOCONAN.csv')
train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Initialize model and tokenizer
# model_name = 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['output']]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=1024, 
        truncation=True, 
        padding='max_length'
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length'
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output']
)

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['input', 'output']
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Evaluation function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(
        predictions=decoded_preds, 
        references=[[ref] for ref in decoded_labels]
    )
    
    # Compute ROUGE
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    
    # Compute BERTScore
    P, R, F1 = bert_score(
        decoded_preds,
        decoded_labels,
        lang="en",
        rescale_with_baseline=True
    )
    
    return {
        'bleu': bleu_results['bleu'],
        'rougeL': rouge_results['rougeL'],
        'bert_score_f1': np.mean(F1.numpy())
    }

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    generation_max_length=128,
    generation_num_beams=4,
)

# When initializing the trainer, pass compute_metrics:
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()
print(results)

# Generate predictions
def generate_counterspeech(input_text):
    inputs = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)