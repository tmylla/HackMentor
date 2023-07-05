import os

import transformers
import torch
from peft import PeftModel
import fire


def load_model(base_model, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+base_model+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(base_model)
    model = transformers.LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    generator = model.generate

def load_lora_model(base_model, lora_model, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+base_model+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(base_model)
    model = transformers.LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()

    print("Loading "+lora_model+"...")
    model = PeftModel.from_pretrained(
        model,
        lora_model,
        torch_dtype=torch.float16
    )
    generator = model.generate


def go(act_as):
    invitation = "HackMentor: "
    human_invitation = "User: "

    # input
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)
    fulltext = act_as + "\n\n".join(history) + "\n\n" + invitation

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
        # no_lora-model
        generated_ids = generator(
            input_ids = gen_in,
            max_new_tokens=2048,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1,
            temperature=0.6,
            top_k = 50,
            top_p = 1.0,
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt
    response = response.split(human_invitation)[0]
    response.strip()

    print(invitation + response)
    print("")

    history.append(invitation + response)



def chat(
        # model/data params
        base_model: str = "",
        use_lora: bool = True,
        lora_model: str = "",
):
    print(
        f"Chat model load with params:\n"
        f"base_model: {base_model}\n"
        f"use_lora: {use_lora}\n"
        f"lora_model: {lora_model}\n"
    )

    global model, tokenizer, generator, history
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    if use_lora:
        if load_model == "":
            raise ValueError("Please enter lora_model location, or set use_lora to False!")
        load_lora_model(base_model, lora_model)
    else:
        load_model(base_model)
    
    First_chat = "HackMentor: I am HackMentor, what cybersecurity questions do you have?\n"
    print(First_chat)
    history = []
    history.append(First_chat)

    act_as = "You are an intelligent assistant in the field of cyber security\
        that mainly helps with resolving cybersecurity claims. \n\n"
    while True:
        go(act_as)

if __name__ == "__main__":
    fire.Fire(chat)