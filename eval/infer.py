import sys
import json

import fire
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

if torch.cuda.is_available():
    device = "cuda"


def load_instruction(instruct_dir):
    input_data = []
    with open(instruct_dir, "r", encoding="utf-8") as f:
        iios = json.load(f)
        for iio in iios:
            _dict = {}
            _dict["instruction"] = iio["instruction"]
            for ins in iio["instances"]:
                _dict["input"] = ins["input"]
                _dict["output"] = ins["output"]
                input_data.append(_dict)

    return input_data


def evaluate(
    instruction,
    input,
    temperature=0.8,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=1024,
    **kwargs,
):
    template = {
        "description": "A shorter template to experiment with.",
        "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
        "response_split": "### Response:"
    }
    if input.strip() != "":
        prompt = template["prompt_input"].format(
            instruction=instruction, input=input
        )
    else:
        prompt = template["prompt_no_input"].format(
            instruction=instruction
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            # return_dict_in_generate=True,
            # output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    # s = generation_output.sequences[0]
    # output = tokenizer.decode(s)

    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return output.split(template["response_split"])[1].strip()


def main(
    load_8bit: bool = False,
    base_model: str = "",
    use_lora: bool = False,
    lora_weights: str = "",
    instruct_dir: str = "",
    result_file: str = "",
):
    print(f"Base_model: {base_model}\nLora_weights: {lora_weights}")

    global tokenizer, model
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if instruct_dir != "no_input":
        input_data = load_instruction(instruct_dir)
        for idx, data in enumerate(tqdm(input_data)):
            instruction = data["instruction"]
            input = data["input"]
            output = data["output"]
            model_output = evaluate(instruction, input)

            result_ = {
                "instruction": instruction,
                "input": input,
                "golden_output": output,
                "model_output": model_output
            }
            with open(result_file, 'a', encoding='utf-8') as f:
                json.dump(result_, f)
                f.write("\n")

            print(f"###{idx}. instruction###")
            print(instruction)
            print(f"###{idx}. input###")
            print(input)
            print(f"###{idx}. golden output###")
            print(output)
            print(f"###{idx}. model output###")
            print(model_output, '\n')
    else:
        for idx, instruction in enumerate(tqdm([
            "Symmetric Vs Asymmetric encryption.",
            "What is the difference between Threat, Vulnerability, and Risk?",
            "I'm struggling to choose a reliable network firewall for our company. We've tested several products, but we're not sure if they're effective enough to protect us from cyber threats. What testing methods and skills should we use to ensure we choose the right product?",
            "How can a deep learning system be compromised?",
        ])):
            print(f"{idx}. Instruction:\n", instruction)
            print(f"{idx}. Response:\n", evaluate(instruction, input=''))
            print()


if __name__ == "__main__":
    fire.Fire(main)
