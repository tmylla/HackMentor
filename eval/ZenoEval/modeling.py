"""Chatbots using API-based services."""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
from collections.abc import Iterable
from typing import Literal

from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn
from zeno_build.models.chat_generate import generate_from_chat_prompt
from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt

import config as chatbot_config

import transformers
from peft import PeftModel
import torch
import re
import tqdm


def _generate_from_huggingface_lora(
    # full_contexts,
    # prompt_template,
    # model_config,
    # temperature: float,
    # max_tokens: int,
    # top_p: float,
    # context_length: int,
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
) -> list[str]:
    # Load model
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls = (
        model_config.model_cls
        if model_config.model_cls is not None
        else transformers.AutoModelForCausalLM
    )
    tokenizer_cls = (
        model_config.tokenizer_cls
        if model_config.tokenizer_cls is not None
        else transformers.AutoTokenizer
    )

    tokenizer: transformers.PreTrainedTokenizer = tokenizer_cls.from_pretrained(
        model_config.model
    )
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model: transformers.PreTrainedModel = model_cls.from_pretrained(
        model_config.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 自定义导入lora模型
    if model_config.lora_model is not None:
        print(f"using lora {model_config.lora_model }")
        model = PeftModel.from_pretrained(
            model,
            model_config.lora_model,
            torch_dtype=torch.float16,
        )

    model.eval()

    gen_config = transformers.GenerationConfig(
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Create the prompts
    filled_prompts: list[str] = [
        prompt_template.to_text_prompt(
            full_context=full_context.limit_length(context_length),
            name_replacements=model_config.name_replacements,
        )
        for full_context in full_contexts
    ]
    # Process in batches
    results = []
    batch_size = 8
    for i in tqdm.trange(0, len(filled_prompts), batch_size):
        batch_prompts = filled_prompts[i : i + batch_size]
        encoded_prompts = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(torch_device)
        with torch.no_grad():
            outputs = model.generate(**encoded_prompts, generation_config=gen_config)
        outputs = outputs[:, encoded_prompts["input_ids"].shape[-1] :]
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    # Post-processing to get only the system utterance
    results = [re.split("\n\n", x)[0].strip() for x in results]
    return results


def generate_from_chat_prompt_lora(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 50,
) -> list[str]:
    """Generate from a list of chat-style prompts.

    Args:
        variables: The variables to be replaced in the prompt template.
        prompt_template: The template for the prompt.
        api_based_model_config: The API-based model configuration.
        temperature: The temperature to use.
        max_tokens: The maximum number of tokens to generate.
        top_p: The top p value to use.
        context_length: The length of the context to use.
        requests_per_minute: Limit on the number of OpenAI requests per minute

    Returns:
        The generated text.
    """
    print(
        f"Generating with {prompt_template=}, {model_config.model=}, "
        f"{temperature=}, {max_tokens=}, {top_p=}, {context_length=}..."
    )
    if model_config.provider == "huggingface":
        return _generate_from_huggingface_lora(
            full_contexts,
            prompt_template,
            model_config,
            temperature,
            max_tokens,
            top_p,
            context_length,
        )
    else:
        raise ValueError("model_config.provider should be 'huggingface'!!!")

def make_predictions(
    data: list[ChatMessages],
    prompt_preset: str,
    model_preset: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_p: float = 1,
    context_length: int = -1,
    cache_root: str | None = None,
) -> list[str]:
    """Make predictions over a particular dataset.

    Args:
        data: The test dataset containing all messages up to last user one.
        prompt_preset: The prompt to use for the API call.
        model_preset: The model to use for the API call.
        temperature: The temperature to use for sampling.
        max_tokens: The maximum number of tokens to generate.
        top_p: The value to use for top-p sampling.
        context_length: The maximum length of the context to use. If 0,
            use the full context.
        cache_root: The location of the cache directory if any

    Returns:
        The predictions in string format.
    """
    # Load from cache if existing
    cache_path: str | None = None
    if cache_root is not None:
        cache_filename = f"{model_preset}-{temperature}-{context_length}.json"
        cache_path = os.path.join(cache_root, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)

    # Make predictions
    hackmentor = [
        "llama-7b",
        "vicuna-7b",
        "llama-13b",
        "vicuna-13b",
        "llama-7b-lora-iio",
        "vicuna-7b-lora-iio",
        "llama-13b-lora-iio",
        "vicuna-13b-lora-iio",
        "llama-7b-lora-turn",
        "llama-13b-lora-turn",
    ]

    if model_preset in hackmentor:  # our finetuned cybersecurity LLMs
        predictions: list[str] = generate_from_chat_prompt_lora(
        data,
        chatbot_config.prompt_messages[prompt_preset],
        chatbot_config.model_configs[model_preset],
        temperature,
        max_tokens,
        top_p,
        context_length,
    )
    else:
        predictions: list[str] = generate_from_chat_prompt(
            data,
            chatbot_config.prompt_messages[prompt_preset],
            chatbot_config.model_configs[model_preset],
            temperature,
            max_tokens,
            top_p,
            context_length,
        )

    # Dump the cache and return
    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(predictions, f)
    return predictions
