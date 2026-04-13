#!/usr/bin/env python3
# diagnose_vllm_repeat.py

import argparse
import json
import requests
from transformers import AutoTokenizer

def print_block(title, content):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)
    print(content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="本地模型目录")
    ap.add_argument("--model-name", required=True, help="vLLM服务里的模型名")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM地址")
    ap.add_argument("--prompt", default="介绍一下你自己，50字以内。")
    ap.add_argument("--max-tokens", type=int, default=80, help="排查时建议先设小，避免跨轮继续生成")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tpl = tok.chat_template
    if isinstance(tpl, dict):
        tpl = tpl.get("default", "")

    print_block("Tokenizer Meta", json.dumps({
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "bos_token": tok.bos_token,
        "bos_token_id": tok.bos_token_id,
        "has_chat_template": bool(tpl),
    }, ensure_ascii=False, indent=2))

    messages = [{"role": "user", "content": args.prompt}]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print_block("Rendered Prompt Tail", rendered[-800:])

    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Stop markers observed in Gemma turn/channel style templates.
    template_stop = [
        "<turn|>",
        "<|turn>user",
        "<|turn>model",
        "<|channel>thought",
        "<|channel>final",
        "<channel|>",
    ]
    # Keep some common fallbacks for other templates.
    legacy_stop = ["<end_of_turn>", "<eot_id>", "</s>", "<|eot_id|>"]
    stop = template_stop + legacy_stop

    payload_no_stop = {
        "model": args.model_name,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens
    }

    payload_with_stop = {
        "model": args.model_name,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "stop": stop
    }

    for tag, payload in [("no_stop", payload_no_stop), ("with_stop", payload_with_stop)]:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage", {})
        print_block(
            f"Result: {tag}",
            json.dumps({
                "finish_reason": finish_reason,
                "usage": usage,
                "text": text
            }, ensure_ascii=False, indent=2)
        )

if __name__ == "__main__":
    main()
