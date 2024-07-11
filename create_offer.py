from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import argparse


def offer(prompt: str):
    # First load the tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_reload = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, "float16"),
        device_map="auto",
        trust_remote_code=True,
    )

    # Merge adapter with base model
    model = PeftModel.from_pretrained(model_reload, "llm-offers")
    model = model.merge_and_unload()

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    result = pipe(f"<s>[INST] {prompt} [/INST]", max_new_tokens=5000, do_sample=True, temperature=0.5, top_p=0.5, top_k=10)
    return result[0]['generated_text']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt')
    args = vars(parser.parse_args())
    print(offer(args["prompt"]))
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    result = pipe(f"<s>[INST] {prompt} [/INST]", max_new_tokens=5000, do_sample=True, temperature=0.5, top_p=0.5, top_k=10)
    return result[0]['generated_text']
