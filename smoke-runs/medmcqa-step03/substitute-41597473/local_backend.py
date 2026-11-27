import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id):
    override = os.environ.get(f"SKILL_MOE_MODEL_{model_id.rsplit('/', 1)[-1]}")
    if override is None and model_id.startswith("Qwen/"):
        override = os.environ.get("SKILL_MOE_MODEL_Qwen")
    model_id = override or model_id
    device = get_device()
    dtype = torch.float16 if device.type in {"mps", "cuda"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_texts(model, tokenizer, prompts, agent_name, temperature=0.7):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    max_tokens = int(
        os.environ.get(
            "SKILL_MOE_MAX_NEW_TOKENS",
            "32768" if "R1" in agent_name else "4096",
        )
    )
    do_sample = temperature > 0
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    outputs = []
    for index, sequence in enumerate(generated):
        input_length = int(encoded["attention_mask"][index].sum().item())
        outputs.append(tokenizer.decode(sequence[input_length:], skip_special_tokens=True))
    return outputs
