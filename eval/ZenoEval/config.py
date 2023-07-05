"""Various configuration options for the chatbot task.

This file is intended to be modified. You can go in and change any
of the variables to run different experiments.
"""

from __future__ import annotations

from typing import Any

import transformers

from zeno_build.evaluation.text_features.exact_match import avg_exact_match, exact_match
from zeno_build.evaluation.text_features.length import (
    chat_context_length,
    input_length,
    label_length,
    output_length,
)
from zeno_build.evaluation.text_metrics.critique import (
    avg_bert_score,
    avg_chrf,
    avg_length_ratio,
    bert_score,
    chrf,
    length_ratio,
    avg_coherence,
    coherence,
    avg_naturalness,
    naturalness,
    avg_understandability,
    understandability
)
from zeno_build.experiments import search_space
from zeno_build.models.lm_config import LMConfig
from zeno_build.prompts.chat_prompt import ChatMessages, ChatTurn

# Define the space of hyperparameters to search over.
space = {
    "model_preset": search_space.Categorical(
        [
            "text-davinci-003",
            "gpt-3.5-turbo",
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
    ),
    "prompt_preset": search_space.Categorical(
        [
            "security",
        ]
    ),
    "temperature": search_space.Discrete([0.3, 0.6, 0.9]),
    "context_length": search_space.Discrete([1, 2, 3, 4]),
}

# Any constants that are not searched over
constants: dict[str, Any] = {
    "test_dataset": "load_custom_dataset",
    "data_column": "turns",
    "data_format": "custom",
    "test_split": None,
    "test_examples": None,
    "max_tokens": 1024,
    "top_p": 1.0,
}

# The number of trials to run
num_trials = 150

# The details of each model
model_configs = {
    "text-davinci-003": LMConfig(provider="openai", model="text-davinci-003"),
    "gpt-3.5-turbo": LMConfig(provider="openai_chat", model="gpt-3.5-turbo"),
    "llama-7b": LMConfig(
        provider="huggingface",
        model="models/llama-7b",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "llama-13b": LMConfig(
        provider="huggingface",
        model="models/llama-13b",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "vicuna-7b": LMConfig(
        provider="huggingface",
        model="models/vicuna-7b",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "vicuna-13b": LMConfig(
        provider="huggingface",
        model="models/vicuna-13b",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "llama-7b-lora-iio": LMConfig(
        provider="huggingface",
        model="models/llama-7b",
        lora_model="lora_models/llama-7b-lora-iio",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "llama-13b-lora-iio": LMConfig(
        provider="huggingface",
        model="models/llama-13b",
        lora_model="lora_models/llama-13b-lora-iio",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "vicuna-7b-lora-iio": LMConfig(
        provider="huggingface",
        model="models/vicuna-7b",
        lora_model="lora_models/vicuna-7b-lora-iio",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "vicuna-13b-lora-iio": LMConfig(
        provider="huggingface",
        model="models/vicuna-13b",
        lora_model="lora_models/vicuan-13b-lora-iio",
        name_replacements={
            "system": "ASSISTANT",
            "assistant": "ASSISTANT",
            "user": "HUMAN",
        },
    ),
    "llama-7b-lora-turn": LMConfig(
        provider="huggingface",
        model="models/llama-7b",
        lora_model="lora_models/llama-7b-lora-turn",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
    "llama-13b-lora-turn": LMConfig(
        provider="huggingface",
        model="models/llama-13b",
        lora_model="lora_models/llama-13b-lora-turn",
        tokenizer_cls=transformers.LlamaTokenizer,
    ),
}

# The details of the prompts
prompt_messages: dict[str, ChatMessages] = {
    "standard": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are a chatbot tasked with making small-talk with "
                "people.",
            ),
        ]
    ),
    "security": ChatMessages(
        messages=[
            ChatTurn(
                role="system",
                content="You are an intelligent assistant in the field of cyber security "
                "that mainly helps with resolving cybersecurity claims."
            ),
        ]
    ),
}

# The functions to use to calculate scores for the hyperparameter sweep
sweep_distill_functions = [chrf]
sweep_metric_function = avg_chrf

# The functions used for Zeno visualization
zeno_distill_and_metric_functions = [
    output_length,
    input_length,
    label_length,
    avg_chrf,
    chrf,
    avg_length_ratio,
    length_ratio,
    avg_bert_score,
    bert_score,
    avg_coherence,
    coherence,
    avg_naturalness,
    naturalness,
    avg_understandability,
    understandability
]
