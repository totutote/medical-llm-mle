"""Qwenの特殊トークンを確認するスクリプト"""
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.secrets', override=True)

hf_token = os.getenv("HF_TOKEN")
base_model = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B")

print(f"モデル: {base_model}\n")

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    token=hf_token
)

print("=== Qwenの特殊トークン ===")
print(f"EOS token: '{tokenizer.eos_token}'")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"BOS token: '{tokenizer.bos_token}'")
print(f"PAD token: '{tokenizer.pad_token}'")
print(f"UNK token: '{tokenizer.unk_token}'")

print(f"\n=== 全特殊トークンマップ ===")
for key, value in tokenizer.special_tokens_map.items():
    print(f"{key}: '{value}'")

print(f"\n=== テスト: EOSトークンをテキストに追加 ===")
test_text = "これはテスト文章です。"
test_with_eos = test_text + tokenizer.eos_token
print(f"元のテキスト: {test_text}")
print(f"EOS追加後: {test_with_eos}")

# トークン化して確認
tokens = tokenizer.encode(test_with_eos)
decoded = tokenizer.decode(tokens)
print(f"\nトークン化: {tokens}")
print(f"デコード: {decoded}")
