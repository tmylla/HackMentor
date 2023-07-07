# HackMentor - 网络安全LLMs领域数据微调

![HackMentor Logo](assets/HackMentor.png)

欢迎来到 HackMentor，一个专注于领域数据微调的网络安全LLMs项目。本项目旨在提供高效的预训练语言模型（LLMs），以帮助解决网络安全领域的挑战。

## 项目概述

HackMentor项目主要包括以下三部分：

1. **数据构建**：我们致力于收集并准备网络安全领域的数据，用于微调预训练语言模型。高质量的训练数据对于模型的成功微调至关重要。

2. **模型训练**：我们利用预训练的语言模型作为基础，在领域数据上进行微调。这样的微调将使模型更加适应网络安全领域的任务。

3. **模型评估**：我们对微调后的模型进行评估，以确保其在网络安全任务上的性能优越。

## Lora-weights 模型

我们目前正在发布可供下载的Lora-weights模型。请注意，在我们的论文被接受后，将发布完整的代码。感谢您的耐心等待。

### Lora-weights 下载

所有的HackMentor权重可访问[链接](https://drive.google.com/drive/folders/1_woz0dsFKq8QYU-X3q0PGoyGoDn-a20t?usp=drive_link)，指定Lora-weights可访问下表：

| HackMentor | Lora-weights |
|-----|-----|
| llama-7b-lora-iio | [download](https://drive.google.com/drive/folders/13xbcQMizfWBtLAJ7OeyRco9QbVn0ZdcM?usp=drive_link) |
| **llama-13b-lora-iio** | [download](https://drive.google.com/drive/folders/17i3A1uuCkpPUJo3DGLvxMVZwqjviqcoE?usp=drive_link) |
| vicuna-7b-lora-iio | [download](https://drive.google.com/drive/folders/1lOEn7QH153QqZ10sFYKdB9afCSkIinMK?usp=drive_link) |
| vicuna-13b-lora-iio | [download](https://drive.google.com/drive/folders/1SF51j4KDyGM356vLx-KuKIni7xYWNbTf?usp=drive_link) |
| llama-7b-lora-turn | [download](https://drive.google.com/drive/folders/1e-Hb3YHlo25y6CL-RhRnrQLTurhgF1Af?usp=drive_link) |
| llama-13b-lora-turn | [download](https://drive.google.com/drive/folders/1lElL6WH1MUWTQZge5utMnmH7aUNIiheK?usp=drive_link) |


*Notes*：
1. HackMentor命名规则为：{基座模型}-{模型尺寸}-{微调方式}-{微调数据}，其中，基座模型可取Llama/Vicuna，模型尺寸有7b/13b，微调方式有Lora（将来拟加入全量微调），微调数据有iio和turn，iio表示指令数据Instruction、Input、Output，turn表示对话数据。
2. 在我们的测试中，表现最好的模型为**llama-13b-lora-iio**，供参考。


### 本地部署和使用 Lora-weights 的步骤

如果您想在本地部署并应用 Lora-weights 模型，按照以下步骤进行操作：

1. 下载基座模型 Llama/Vicuna 和本项目提供的 Lora-weights 模型，放置到 `models` 目录下。

2. 下载 `chat.py`，配置环境，该 py 文件运行依赖包如下：

    ```python
    bitsandbytes==0.39.0
    fire==0.5.0
    peft @ git+https://github.com/huggingface/peft.git@3714aa2fff158fdfa637b2b65952580801d890b2
    torch==2.0.1
    transformers==4.28.1
    ```

3. 切换到相应目录，根据需求运行下表所示脚本。

    | base-model | lora-weights | domain | execute-command |
    | --- | --- | --- | --- |
    | llama-7b | - | general | `python chat.py --base_model models/pretrained/llama-7b --use_lora False` |
    | vicuna-13b | - | general | `python chat.py --base_model models/pretrained/vicuna-13b --use_lora False` |
    | **llama-13b** | **llama-13b-lora-iio** | **security** | **`python chat.py --base_model models/pretrained/llama-13b --lora_model models/lora_models/llama-13b-lora-iio`** |
    | vicuna-7b | vicuna-7b-lora-iio | security | `python chat.py --base_model models/pretrained/vicuna-7b  --lora_model models/lora_models/vicuna-7b-lora-iio` |
    | llama-7b | llama-7b-lora-turn | security | `python chat.py --base_model models/pretrained/llama-7b --lora_model models/lora_models/llama-7b-lora-turn` |
    | ...  | ... | ... | ... |


请注意，以上代码示例仅用于说明目的，您可能需要根据自己的实际情况进行适当调整。

## 参与贡献

我们非常欢迎对于 HackMentor 项目的贡献。如果您发现任何问题或有改进建议，请提交 issue 或发送 pull 请求。您的贡献将会使这个项目变得更好。


## 更多代码与数据

感谢您的耐心等待...