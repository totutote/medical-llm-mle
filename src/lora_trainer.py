"""
Qwen 0.6B LoRAファインチューニングスクリプト

PEFT (Parameter-Efficient Fine-Tuning) を使用したLoRA学習
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import numpy as np
from dotenv import load_dotenv

# .envと.env.secretsの両方を読み込み
load_dotenv(override=True)  # .env（常に上書き）
load_dotenv('.env.secrets', override=True)  # .env.secrets (優先)


@dataclass
class LoRATrainingConfig:
    """LoRA学習の設定"""
    
    # モデル設定
    base_model: str = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    
    # LoRA設定
    lora_r: int = int(os.getenv("LORA_R", "16"))  # 8→16に増加（4倍のパラメータ）
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))  # r×2を維持
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))  # 過学習を抑制するため減少
    target_modules: list = None
    
    # 学習設定
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "100"))
    num_epochs_followup: int = int(os.getenv("NUM_EPOCHS_FOLLOWUP", "30"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    warmup_steps: int = 100
    logging_steps: int = 1
    save_steps: int = 100
    
    # 出力設定
    output_dir: str = "checkpoints"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Qwen3 (Qwen2.5ベース) のターゲットモジュール
            # Attentionのみの場合: ["q_proj", "k_proj", "v_proj", "o_proj"]
            # より多くのパラメータを学習する場合はMLPも含める
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"  # MLP (SwiGLU FFN)
            ]


class MedicalLoRATrainer:
    """医療LLMのLoRAトレーナー"""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model(self, previous_checkpoint: str = None):
        """モデルとトークナイザーのセットアップ
        
        Args:
            previous_checkpoint: 前回のマージ済みモデルパス（継続学習の場合）
        """
        if previous_checkpoint:
            print(f"\n🔧 前回のマージ済みモデルから継続学習: {previous_checkpoint}")
        else:
            print(f"\n🔧 ベースモデルから新規学習: {self.config.base_model}")
        
        # Hugging Face Tokenの設定（環境変数から）
        hf_token = os.getenv("HF_TOKEN", None)
        
        # トークナイザーの読み込み（常にベースモデルから）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
            token=hf_token
        )
        
        # パディングトークンの設定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの読み込み
        if previous_checkpoint:
            # 前回のマージ済みモデルを新しいベースとして読み込む
            print("  📥 前回のマージ済みモデルを読み込み中...")
            self.model = AutoModelForCausalLM.from_pretrained(
                previous_checkpoint,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            print("✅ マージ済みモデル読み込み完了")
            print("   💡 このモデルに新しいLoRAレイヤーを追加して学習します")
        else:
            # ベースモデルから新規学習
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            print("✅ ベースモデル読み込み完了")
        
        # 勾配チェックポイントを有効化（メモリ節約）
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
    def setup_lora(self):
        """LoRA設定の適用"""
        print("\n🔧 LoRA設定を適用中...")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 現在のモデル（ベースまたはマージ済み）に新しいLoRAレイヤーを追加
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        print("✅ LoRA設定完了")
        
    def prepare_dataset(self, data_path: str):
        """データセットの準備"""
        print(f"\n📊 データセット準備: {data_path}")
        
        # JSONLファイルの読み込み
        dataset = load_dataset("json", data_files=data_path, split="train")
        
        def format_instruction(example):
            """命令形式のプロンプトを作成"""
            prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{example['instruction']}

### 応答:
{example['output']}"""
            return {"text": prompt}
        
        # データセットをフォーマット
        formatted_dataset = dataset.map(format_instruction)
        
        def tokenize_function(examples):
            """トークン化"""
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
        )
        
        print(f"✅ データセット準備完了: {len(tokenized_dataset)} サンプル")
        return tokenized_dataset
    
    def train(self, train_dataset, iteration: int = 1, previous_checkpoint: str = None):
        """LoRAファインチューニングの実行
        
        前回のチェックポイントがある場合は継続学習を行います。
        これにより、正答していた問題を忘れず、間違えた問題だけを追加学習できます。
        
        Args:
            train_dataset: 学習データセット
            iteration: 現在のiteration番号
            previous_checkpoint: 前回のチェックポイントパス（継続学習の場合）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 絶対パスに変換
        base_output_dir = os.path.abspath(self.config.output_dir)
        output_dir = os.path.join(
            base_output_dir,
            f"iteration_{iteration}_{timestamp}"
        )
        
        # 出力ディレクトリを事前に作成（親ディレクトリも含めて）
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🚀 学習開始 (Iteration {iteration})")
        print(f"   出力ディレクトリ: {output_dir}")
        if previous_checkpoint:
            print(f"   前回のチェックポイントから継続学習: {previous_checkpoint}")
        else:
            print("   ベースモデルから新規に学習します")
        
        # iterationに応じてエポック数を決定（iteration 0のみ100エポック）
        epochs = self.config.num_epochs if iteration == 0 else self.config.num_epochs_followup
        print(f"   エポック数: {epochs}")
        
        # 学習引数の設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs",
        )
        
        # データコレクターの設定
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # トレーナーの初期化
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # 学習の実行
        print("\n⏳ 学習中...")
        trainer.train()
        
        # LoRAアダプターを保存（デバッグ・分析用）
        lora_adapter_dir = os.path.join(output_dir, "lora_adapter")
        trainer.save_model(lora_adapter_dir)
        print(f"\n💾 LoRAアダプター保存: {lora_adapter_dir}")
        
        # LoRAをベースモデルにマージして保存（次回の継続学習・評価用）
        print("\n🔀 LoRAをマージして統合モデルとして保存中...")
        merged_model = self.peft_model.merge_and_unload()
        
        final_output_dir = os.path.join(output_dir, "final")
        merged_model.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        print(f"✅ マージ済みモデル保存: {final_output_dir}")
        print("   💡 このモデルは次回のiteration or 評価で直接読み込めます（LoRA不要）")
        
        # 学習統計の保存
        stats = {
            "iteration": iteration,
            "timestamp": timestamp,
            "config": {
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "num_epochs": epochs,
                "batch_size": self.config.batch_size,
            },
            "output_dir": final_output_dir,
            "lora_adapter_dir": lora_adapter_dir,
        }
        
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n✅ 学習完了!")
        print(f"   マージ済みモデル: {final_output_dir}")
        print(f"   LoRAアダプター: {lora_adapter_dir}")
        
        return final_output_dir
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """モデルからの応答生成（推論）"""
        formatted_prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{prompt}

### 応答:
"""
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプト部分を除去
        if "### 応答:" in response:
            response = response.split("### 応答:")[-1].strip()
        
        return response


def main():
    """テスト実行"""
    print("=" * 70)
    print("Qwen 0.6B LoRA ファインチューニング")
    print("=" * 70)
    
    # 設定の初期化
    config = LoRATrainingConfig()
    
    # 設定内容の表示
    print("\n📋 学習設定:")
    print(f"   エポック数: {config.num_epochs}")
    print(f"   バッチサイズ: {config.batch_size}")
    print(f"   学習率: {config.learning_rate}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   LoRA alpha: {config.lora_alpha}")
    
    # トレーナーの初期化
    trainer = MedicalLoRATrainer(config)
    trainer.setup_model()
    trainer.setup_lora()
    
    # データセットの準備
    train_dataset = trainer.prepare_dataset("data/training_data.jsonl")
    
    # 学習の実行
    checkpoint_path = trainer.train(train_dataset, iteration=1)
    
    # テスト推論
    print("\n" + "=" * 70)
    print("テスト推論")
    print("=" * 70)
    
    test_prompt = "以下の症状を持つ成人患者に対して、一般的に用いられる薬剤候補を複数挙げてください。\n\n症状: 頭痛と発熱"
    response = trainer.generate_response(test_prompt)
    
    print(f"\nプロンプト: {test_prompt}")
    print(f"\n応答:\n{response}")


if __name__ == "__main__":
    main()
