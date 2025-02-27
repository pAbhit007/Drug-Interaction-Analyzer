# model_training.py
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# Create models directory
os.makedirs("models", exist_ok=True)

def prepare_dataset():
    """Prepare dataset for training"""
    # Load data
    train_df = pd.read_csv("data/train_data.csv")
    val_df = pd.read_csv("data/val_data.csv")
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer, max_input_length=512, max_output_length=512):
    """Tokenize inputs and outputs for Seq2Seq model"""
    inputs = examples["input"]
    outputs = examples["output"]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    labels = tokenizer(outputs, max_length=max_output_length, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_medical_model():
    """Fine-tune a medical model for drug interactions"""
    print("Loading medical model and tokenizer...")
    
    # We'll use Flan-T5 as our base model (you could also use a medical specific model)
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_dataset()
    
    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # Set up LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,  # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]  # attention query and value matrices
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="models/drug-interaction-model",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )
    
    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    
    # Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting model fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained("models/drug-interaction-model")
    tokenizer.save_pretrained("models/drug-interaction-model-tokenizer")
    print("Model fine-tuning complete!")
    
    return "models/drug-interaction-model"

if __name__ == "__main__":
    fine_tune_medical_model()