#!/bin/sh

# Base-models
LLAMA_7B_MODEL="models/pretrained/llama-7b"
LLAMA_13B_MODEL="models/pretrained/llama-13b"
VICUNA_7B_MODEL="models/pretrained/vicuna-7b"
VICUNA_13B_MODEL="models/pretrained/vicuna-13b"

# Test-instructions
INSTRUCT_DIR="eval/eval_data/eval_seed_data.json"
# INSTRUCT_DIR="no_input" # 测试用例

# llama
llama_7b_cmd="python eval/infer.py \
    --base_model ${LLAMA_7B_MODEL} \
    --use_lora False \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-7b.jsonl"

llama_13b_cmd="python eval/infer.py \
    --base_model ${LLAMA_13B_MODEL} \
    --use_lora False \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-13b.jsonl"

# Vicuna
vicuna_7b_cmd="python eval/infer.py \
    --base_model ${VICUNA_7B_MODEL} \
    --use_lora False \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/vicuna-7b.jsonl"

vicuna_13b_cmd="python eval/infer.py \
    --base_model ${VICUNA_13B_MODEL} \
    --use_lora False \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/vicuna-13b.jsonl"

echo "llama_7b"
eval $llama_7b_cmd >> eval/infer_results/output.txt
echo "llama_13b"
eval $llama_13b_cmd >> eval/infer_results/output.txt
echo "vicuna_7b"
eval $vicuna_7b_cmd >> eval/infer_results/output.txt
echo "vicuna_13b"
eval $vicuna_13b_cmd >> eval/infer_results/output.txt


##############LORA-IIO###############
# Lora-weights-iio
LLAMA_7B_LORA_IIO="models/lora_models/llama-7b-lora-iio"
LLAMA_13B_LORA_IIO="models/lora_models/llama-13b-lora-iio"
VICUNA_7B_LORA_IIO="models/lora_models/vicuna-7b-lora-iio"
VICUNA_13B_LORA_IIO="models/lora_models/vicuna-13b-lora-iio"

# llama-sec-iio
llama_7b_sec_iio_cmd="python eval/infer.py \
    --base_model ${LLAMA_7B_MODEL} \
    --use_lora True \
    --lora_weights ${LLAMA_7B_LORA_IIO} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-7b-lora-iio.jsonl"

llama_13b_sec_iio_cmd="python eval/infer.py \
    --base_model ${LLAMA_13B_MODEL} \
    --use_lora True \
    --lora_weights ${LLAMA_13B_LORA_IIO} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-13b-lora-iio.jsonl"

# vicuna-sec-iio
vicuna_7b_sec_iio_cmd="python eval/infer.py \
    --base_model ${VICUNA_7B_MODEL} \
    --use_lora True \
    --lora_weights ${VICUNA_7B_LORA_IIO} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/vicuna-7b-lora-iio.jsonl"

vicuna_13b_sec_iio_cmd="python eval/infer.py \
    --base_model ${VICUNA_13B_MODEL} \
    --use_lora True \
    --lora_weights ${VICUNA_13B_LORA_IIO} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/vicuan-13b-lora-iio.jsonl"

echo "llama-7b-sec-iio"
eval $llama_7b_sec_iio_cmd >> eval/infer_results/output.txt
echo "llama_13b-sec-iio"
eval $llama_13b_sec_iio_cmd >> eval/infer_results/output.txt
echo "vicuna_7b-sec-iio"
eval $vicuna_7b_sec_iio_cmd >> eval/infer_results/output.txt
echo "vicuna_13b-sec-iio"
eval $vicuna_13b_sec_iio_cmd >> eval/infer_results/output.txt


##############LORA-TURN###############
# Lora-weights-turn
LLAMA_7B_LORA_TURN="models/lora_models/llama-7b-lora-turn"
LLAMA_13B_LORA_TURN="models/lora_models/llama-13b-lora-turn"

# llama-sec-turn
llama_7b_sec_turn_cmd="python eval/infer.py \
    --base_model ${LLAMA_7B_MODEL} \
    --use_lora True \
    --lora_weights ${LLAMA_7B_LORA_TURN} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-7b-lora-turn.jsonl"

llama_13b_sec_turn_cmd="python eval/infer.py \
    --base_model ${LLAMA_13B_MODEL} \
    --use_lora True \
    --lora_weights ${LLAMA_13B_LORA_TURN} \
    --instruct_dir ${INSTRUCT_DIR} \
    --result_file eval/infer_results/llama-13b-lora-turn.jsonl"

echo "llama-7b-sec-turn"
eval $llama_7b_sec_turn_cmd >> eval/infer_results/output.txt
echo "llama_13b-sec-turn"
eval $llama_13b_sec_turn_cmd >> eval/infer_results/output.txt