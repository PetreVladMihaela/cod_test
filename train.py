import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from create_offer import offer


texts = []
for txtFile in os.listdir("oferte"):
    with open(os.path.join("oferte", txtFile), "r") as f:
        text = f.read()
        texts.append(text)
dataset = Dataset.from_dict({"text": texts})
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.15, seed=42)


# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# Fine-tuned model
new_model = "llm-offers"


#4-bit quantization via QLoRA with NF4 type configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=True
)


# load model using 4-bit precision
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable() # reduce memory usage during fine-tuning


# load the tokenizer from Hugginface and set padding_side to “right” to fix the issue with fp16
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_eos_token = True


#Parameter-Efficient Fine-Tuning (PEFT) works by only updating a small subset of the model's most influential parameters
peft_config = LoraConfig(
    r=16,# A higher rank will allow for more expressivity, but there is a compute tradeoff
    lora_alpha=16, # a higher value for alpha assigns more weight to the LoRA activations
    lora_dropout=0.05,
    #target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj', 'lm_head'],
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=2,
    save_steps=2,
    learning_rate=5e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.5,
    #warmup_steps=10,
    group_by_length=True,
    lr_scheduler_type="linear",
)

# Supervised fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

# save the model adopter and tokenizers
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)


# test the fine-tuned model using transformers text generation pipeline
for i in range(3):
    prompt = dataset["test"]["text"][i].partition("[/INST]")[0].lstrip("<s>[INST]").strip()
    print("Request")
    print(prompt)
    print("Offer")
    print(offer(prompt))
    print()
