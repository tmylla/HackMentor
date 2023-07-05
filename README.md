**[中文](README-ZH.md)** | [English](README.md)

# HackMentor: Fine-Tuning Large Language Models for Cybersecurity

![HackMentor Logo](assets\HackMentor.png)

HackMentor is a cybersecurity LLMs (Large Language Models) focused on domain-specific data fine-tuning. This project consists of three main parts: data construction, model training, and model evaluation.

**(Please note that the complete code will be published after the acceptance of the paper. Thank you for your patience.)**

## Features

- Data construction: Methods and tools for creating domain-specific datasets (instructions & conversations) for fine-tuning LLMs.
- Model training: Techniques and processes for training LLMs on the constructed datasets.
- Model evaluation: Metrics and evaluation methodologies to assess the performance of the fine-tuned models.

## Lora-weights Model

We are currently releasing the Lora-weights model, which is available for download. Please note that the complete code will be released after the acceptance of our paper. Thank you for your patience.

### Download Lora-weights

All HackMentor weights are accessible [here]((https://drive.google.com/drive/folders/1_woz0dsFKq8QYU-X3q0PGoyGoDn-a20t?usp=drive_link)), and the specified Lora-weights can be accessed in the table below:

| HackMentor | Lora-weights |
|-----|-----|
| llama-7b-lora-iio | [download](https://drive.google.com/drive/folders/13xbcQMizfWBtLAJ7OeyRco9QbVn0ZdcM?usp=drive_link) |
| **llama-13b-lora-iio** | [download](https://drive.google.com/drive/folders/17i3A1uuCkpPUJo3DGLvxMVZwqjviqcoE?usp=drive_link) |
| vicuna-7b-lora-iio | [download](https://drive.google.com/drive/folders/1lOEn7QH153QqZ10sFYKdB9afCSkIinMK?usp=drive_link) |
| vicuna-13b-lora-iio | [download](https://drive.google.com/drive/folders/1SF51j4KDyGM356vLx-KuKIni7xYWNbTf?usp=drive_link) |
| llama-7b-lora-turn | [download](https://drive.google.com/drive/folders/1e-Hb3YHlo25y6CL-RhRnrQLTurhgF1Af?usp=drive_link) |
| llama-13b-lora-turn | [download](https://drive.google.com/drive/folders/1lElL6WH1MUWTQZge5utMnmH7aUNIiheK?usp=drive_link) |
|  |  |

*Notes:*
1. The naming convention for HackMentor is as follows: {base model}-{model size}-{fine-tuning method}-{fine-tuning data}. Here, the base model can be Llama/Vicuna, the model size can be 7b/13b, the fine-tuning method can be Lora (with plans to include full-parameters fine-tuning), and the fine-tuning data can be iio or turn, where iio represents Instruction, Input, Output data, and turn represents conversation data.
2. In our testing, the best-performing model was **Llama-13b-Lora-iio**, for reference.


## Local Deployment and Usage

To deploy and utilize the Lora-weights model locally, follow the steps below:

1. Download the base models Llama/Vicuna and the Lora-weights model provided by this project, and place them in the `models` directory.:


2. Download `chat.py`, configure the environment, and ensure the following dependencies are installed for running the Python file:

    ```python
    bitsandbytes==0.39.0
    fire==0.5.0
    peft @ git+https://github.com/huggingface/peft.git@3714aa2fff158fdfa637b2b65952580801d890b2
    torch==2.0.1
    transformers==4.28.1
    ```

3. Switch to the corresponding directory and run the scripts shown in the table below according to the requirements:

    | base-model | lora-weights | domain | execute-command |
    | --- | --- | --- | --- |
    | llama-7b | - | general | `python chat.py --base_model models/pretrained/llama-7b --use_lora False` |
    | vicuna-13b | - | general | `python chat.py --base_model models/pretrained/vicuna-13b --use_lora False` |
    | **llama-13b** | **llama-13b-lora-iio** | **security** | `python chat.py --base_model models/pretrained/llama-13b --lora_model models/lora_models/llama-13b-lora-iio` |
    | vicuna-7b | vicuna-7b-lora-iio | security | `python chat.py --base_model models/pretrained/vicuna-7b  --lora_model models/lora_models/vicuna-7b-lora-iio` |
    | llama-7b | llama-7b-lora-turn | security | `python chat.py --base_model models/pretrained/llama-7b --lora_model models/lora_models/llama-7b-lora-turn` |
    | ...  | ... | ... | ... |
    |  |  |  |  |

Please note that the above code examples are for illustrative purposes only and you may need to make appropriate adjustments based on your specific situation.


## Contribution

We welcome contributions to the HackMentor project. If you find any issues or have any improvement suggestions, please submit an issue or send a pull request. Your contributions will help make this project better.

## More

Coming soon ...
