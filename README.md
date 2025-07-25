# LLAMA, PHI-3, and GEMMA-2 Tokenizer Comparison

This notebook demonstrates how to load, configure, and run inference with multiple popular instruction-tuned LLMs — including **LLaMA 3.1**, **Phi-3**, and **Gemma-2** — using Hugging Face Transformers and low-bit quantization for efficient inference on consumer GPUs or Google Colab.

## Models Used

- [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)

## Features

- Runs on **Google Colab (T4 GPU)**
- Loads and runs models via Hugging Face `transformers` + `accelerate`
- Uses `bitsandbytes` for **4-bit quantized inference**
- Tokenization and message formatting for chat-style models
- Safe for low-resource environments

## Installation

Required libraries:

```python
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q transformers==4.48.3 bitsandbytes==0.46.0 accelerate==1.3.0
```

## Usage

Authenticate with Hugging Face (e.g. using `google.colab.userdata`):

```python
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get('your_hf_token')
login(hf_token)
```

Set up models and prompt:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch, gc

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # or any of the others
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]
```

Run inference:

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer)
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
_ = model.generate(input_ids=input_ids, streamer=streamer, max_new_tokens=200)
```

## Notes

- Warnings from pip about package incompatibilities (e.g., `fsspec`) can be ignored.
- If CUDA or `bitsandbytes` errors occur in Colab, restart the runtime and rerun setup cells.

## License

This project uses models from Hugging Face under their respective licenses.

---
**Last Updated:** July 2025
