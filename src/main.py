"""
医療LLM LoRA反復学習・評価ループのオーケストレーター

ベースライン評価 → LoRAファインチューニング → 評価 のサイクルを管理
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from data_generator import create_training_data, create_evaluation_data, load_evaluation_data
from lora_trainer import MedicalLoRATrainer, LoRATrainingConfig
from evaluator import MedicalLLMEvaluator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class IterativeTrainingLoop:
    """反復学習ループの管理"""
    
    def __init__(
        self,
        base_model_name: str,
        num_iterations: int = 2,
        output_dir: str = "results",
        data_dir: str = "data",
        num_samples_per_question: int = 3,  # N回サンプリング（仕様書準拠）
        enable_baseline: bool = False,  # ベースライン評価を有効化（デフォルト: 無効）
        rehearsal_ratio: float = 1.0  # 成功例リハーサルデータの比率（誤答数に対する倍率）
    ):
        self.base_model_name = base_model_name
        self.num_iterations = num_iterations
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.num_samples_per_question = num_samples_per_question
        self.enable_baseline = enable_baseline
        self.rehearsal_ratio = rehearsal_ratio
        
        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験メタデータ
        self.experiment_id = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
        self.iteration_results = []
        
    def setup_data(self, num_train_samples: int = None, num_eval_samples: int = 15):
        """データセットのセットアップ（仕様書準拠: 15件）"""
        print("\n" + "=" * 70)
        print("📊 データセット生成")
        print("=" * 70)
        
        # トレーニングデータ生成（Noneの場合は全データを使用）
        train_path = create_training_data(
            output_dir=str(self.data_dir),
            num_samples=num_train_samples
        )
        
        # 評価データ生成
        eval_path = create_evaluation_data(
            output_dir=str(self.data_dir),
            num_samples=num_eval_samples
        )
        
        return train_path, eval_path
    
    def evaluate_model(
        self,
        model_name_or_path: str,
        eval_data: List[Dict],
        iteration: int,
        is_baseline: bool = False
    ) -> Dict:
        """モデルの評価を実行"""
        
        iteration_name = "baseline" if is_baseline else f"iteration_{iteration}"
        print("\n" + "=" * 70)
        print(f"📈 評価: {iteration_name}")
        print("=" * 70)
        
        # モデルとトークナイザーの読み込み
        print(f"\n🔧 モデル読み込み: {model_name_or_path}")
        
        # Hugging Face Tokenの設定
        hf_token = os.getenv("HF_TOKEN", None)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # マージ済みモデルとして直接読み込む（LoRA不要）
        # ベースラインもiteration後のモデルも同じ方法で読み込める
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        
        # 推論関数の定義
        def generate_fn(prompt: str) -> str:
            formatted_prompt = f"""以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

### 指示:
{prompt}

### 応答:
"""
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,  # 256の2/3に短縮して冗長な出力を防ぐ
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,  # 繰り返しを防ぐ
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "### 応答:" in response:
                response = response.split("### 応答:")[-1].strip()
            
            return response
        
        # 評価実行（N回サンプリング）
        evaluator = MedicalLLMEvaluator()
        
        all_results = []
        print(f"\n📊 各質問に対して{self.num_samples_per_question}回サンプリングを実行")
        
        for question_idx, test_case in enumerate(eval_data, 1):
            print(f"\n質問 {question_idx}/{len(eval_data)}: {test_case['symptom'][:50]}...")
            
            for sample_idx in range(self.num_samples_per_question):
                symptom = test_case["symptom"]
                expected = test_case["expected_medications"]
                
                # 最初と最後のサンプルは詳細表示
                show_details = (sample_idx == 0 or sample_idx == self.num_samples_per_question - 1)
                
                # モデルから応答を生成(軽量モデル向けテンプレート形式)
                if show_details:
                    print(f"\n  [{sample_idx + 1}/{self.num_samples_per_question}] モデル推論中...")
                else:
                    print(f"  [{sample_idx + 1}/{self.num_samples_per_question}] モデル推論中...", end="", flush=True)
                from data_generator import format_instruction
                prompt = format_instruction(symptom)
                
                # モデル推論時間の計測
                import time
                inference_start_time = time.time()
                model_response = generate_fn(prompt)
                inference_elapsed_time = time.time() - inference_start_time
                
                if show_details:
                    print(f"  ⏱️  モデル推論時間: {inference_elapsed_time:.2f}秒")
                else:
                    print(f" ({inference_elapsed_time:.2f}秒)", end="", flush=True)
                
                if show_details:
                    print(f"  📝 モデル出力:\n{'-' * 60}")
                    print(f"{model_response}")
                    print(f"{'-' * 60}")
                
                # 評価を実行
                if show_details:
                    print(f"  🤖 ChatGPT評価中...")
                else:
                    print(" → ChatGPT評価中...", end="", flush=True)
                
                evaluation = evaluator.evaluate_single(symptom, model_response, expected)
                all_results.append(evaluation)
                
                if show_details:
                    # JSON形式で評価結果を表示
                    from dataclasses import asdict
                    eval_dict = asdict(evaluation)
                    print(f"  ✅ 評価結果 (JSON):")
                    print(json.dumps(eval_dict, ensure_ascii=False, indent=2))
                else:
                    print(f" ✓ (ラベル: {evaluation.overall_label})")
                
                if (sample_idx + 1) % 10 == 0:
                    print(f"  📊 進捗: {sample_idx + 1}/{self.num_samples_per_question} 完了")
        
        results = all_results
        
        # メトリクス計算
        metrics = evaluator.calculate_metrics(results)
        
        # 結果保存
        result_path = self.output_dir / f"{iteration_name}_evaluation.json"
        evaluator.save_results(results, str(result_path))
        
        # サマリー表示
        evaluator.print_summary(results)
        
        # メモリ解放
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return {
            "iteration": iteration,
            "is_baseline": is_baseline,
            "model_path": model_name_or_path,
            "metrics": metrics,
            "result_path": str(result_path),
            "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat()
        }
    
    def extract_incorrect_cases(self, results: List) -> List[Dict]:
        """間違えたケースを抽出して教師データに変換（SAMPLE_MEDICAL_DATAから正しい答えを取得）"""
        incorrect_samples = []
        
        for result in results:
            # overall_labelがincorrect/unsafe/partially_correctを抽出
            if result.overall_label in ["incorrect", "unsafe", "partially_correct"]:
                # 症状を抽出
                symptom = result.question
                
                # SAMPLE_MEDICAL_DATAから正しい答えを取得
                from data_generator import find_correct_medication_from_sample_data, format_instruction
                correct_medications = find_correct_medication_from_sample_data(symptom)
                
                if correct_medications:
                    instruction = format_instruction(symptom)
                    
                    # 元データのフォーマットに合わせる: "薬剤名 - 理由\n2. 薬剤名 - 理由\n3. 薬剤名 - 理由"
                    formatted_output = self._format_medications_from_json(correct_medications)
                    
                    incorrect_samples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": formatted_output,
                        "symptom": symptom  # マッチング用
                    })
                else:
                    print(f"  ⚠️  SAMPLE_MEDICAL_DATAから正しい答えが見つかりません: {symptom[:50]}...")
        
        print(f"  抽出された誤答・不安全ケース: {len(incorrect_samples)}件")
        return incorrect_samples
    
    def _format_medications_from_json(self, medications: List[Dict]) -> str:
        """JSON配列形式の薬剤リストをテキスト形式に整形
        
        Args:
            medications: [{"name": "薬剤名", "reason": "理由"}, ...] 形式のリスト
            
        Returns:
            "薬剤名1 - 理由1\n2. 薬剤名2 - 理由2\n3. 薬剤名3 - 理由3" 形式の文字列
        """
        if not medications or len(medications) == 0:
            return ""
        
        # 最大3つまで
        meds = medications[:3]
        lines = []
        
        for i, med in enumerate(meds, start=1):
            name = med.get("name", "")
            reason = med.get("reason", "")
            
            if i == 1:
                # 最初は "1. " なし（元データのフォーマットに合わせる）
                lines.append(f"{name} - {reason}")
            else:
                lines.append(f"{i}. {name} - {reason}")
        
        return "\n".join(lines)
    
    def extract_correct_cases(self, results: List, num_samples: int) -> List[Dict]:
        """正答したケースからランダムにサンプリング（忘却防止用リハーサルデータ）"""
        correct_samples = []
        
        for result in results:
            # overall_labelがcorrectのもののみ抽出
            if result.overall_label == "correct":
                symptom = result.question
                
                # SAMPLE_MEDICAL_DATAから正しい答えを取得
                from data_generator import find_correct_medication_from_sample_data, format_instruction
                correct_medications = find_correct_medication_from_sample_data(symptom)
                
                if correct_medications:
                    instruction = format_instruction(symptom)
                    formatted_output = self._format_medications_from_json(correct_medications)
                    
                    correct_samples.append({
                        "instruction": instruction,
                        "input": "",
                        "output": formatted_output,
                        "symptom": symptom
                    })
        
        # ランダムにサンプリング
        import random
        if len(correct_samples) > num_samples:
            sampled = random.sample(correct_samples, num_samples)
        else:
            sampled = correct_samples
        
        print(f"  正答ケースから{len(sampled)}件をリハーサルデータとして抽出（全{len(correct_samples)}件中）")
        return sampled
    
    def create_dynamic_training_data(self, samples: List[Dict], iteration: int) -> str:
        """動的に抽出したサンプルから教師データファイルを作成"""
        output_path = self.data_dir / f"training_data_iteration_{iteration}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"  動的教師データ保存: {output_path} ({len(samples)}件)")
        return str(output_path)
    
    def create_curriculum_training_data(
        self, 
        prev_results: List,
        iteration: int
    ) -> str:
        """カリキュラム学習用のトレーニングデータを作成
        
        誤答修正データ(A) + 成功例リハーサルデータ(B)を組み合わせます。
        SAMPLE_MEDICAL_DATAから正しい処方のみを取得します。
        
        Args:
            prev_results: 前回の評価結果リスト
            iteration: 現在のiteration番号
        
        Returns:
            作成されたトレーニングデータのパス
        """
        # (A) 誤答修正データ: 間違えたケースをSAMPLE_MEDICAL_DATAの正しい答えで学習
        incorrect_samples = self.extract_incorrect_cases(prev_results)
        
        # (B) 成功例リハーサルデータ: 正答したケースから一部をサンプリング（忘却防止）
        num_rehearsal = int(len(incorrect_samples) * self.rehearsal_ratio)
        rehearsal_samples = self.extract_correct_cases(prev_results, num_rehearsal)
        
        # A + B を結合
        
        # symptomフィールドを削除してクリーンなデータにする
        clean_incorrect = [{
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"]
        } for s in incorrect_samples]
        
        clean_rehearsal = [{
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"]
        } for s in rehearsal_samples]
        
        # データセットとして結合
        final_samples = clean_incorrect + clean_rehearsal
        
        # JSONLファイルとして保存
        output_path = self.data_dir / f"training_data_iteration_{iteration}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in final_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        print(f"\n  📊 カリキュラム学習データ作成完了:")
        print(f"    - 誤答修正データ (A): {len(clean_incorrect)}件")
        print(f"    - 成功例リハーサルデータ (B): {len(clean_rehearsal)}件")
        print(f"    - 合計学習データ: {len(final_samples)}件")
        print(f"    - 保存先: {output_path}")
        
        return str(output_path)
    
    def run(self):
        """反復学習ループの実行"""
        
        print("\n" + "=" * 70)
        print("🚀 医療LLM LoRA反復学習・評価ループ")
        print("=" * 70)
        print(f"実験ID: {self.experiment_id}")
        print(f"ベースモデル: {self.base_model_name}")
        print(f"反復回数: {self.num_iterations}")
        print(f"ベースライン評価: {'有効' if self.enable_baseline else '無効（初回学習から開始）'}")
        print(f"出力先: {self.output_dir}")
        print("=" * 70)
        
        # データセットの準備
        train_path, eval_path = self.setup_data()
        eval_data = load_evaluation_data(eval_path)
        
        # ステップ1: ベースライン評価（オプション）
        if self.enable_baseline:
            print("\n" + "=" * 70)
            print("📊 ベースライン評価を実行します")
            print("=" * 70)
            baseline_result = self.evaluate_model(
                model_name_or_path=self.base_model_name,
                eval_data=eval_data,
                iteration=0,
                is_baseline=True
            )
            self.iteration_results.append(baseline_result)
        else:
            print("\n" + "=" * 70)
            print("⚡ ベースライン評価をスキップし、初回学習から開始します")
            print("   理由: Qwen 0.6Bは医療知識がほぼゼロで、評価コストが無駄になるため")
            print("=" * 70)
        
        # ステップ2: 初回必須学習（Iteration 0）
        print("\n" + "=" * 70)
        print("🎯 初回学習（Iteration 0）: training_data.jsonlで事前学習")
        print("=" * 70)
        
        config = LoRATrainingConfig()
        trainer = MedicalLoRATrainer(config)
        trainer.setup_model(previous_checkpoint=None)  # 初回はベースモデルから
        trainer.setup_lora()
        
        # 元のトレーニングデータで学習
        train_dataset = trainer.prepare_dataset(train_path)
        initial_checkpoint = trainer.train(train_dataset, iteration=0, previous_checkpoint=None)
        
        # メモリ解放
        del trainer
        torch.cuda.empty_cache()
        
        # 初回モデルの評価
        print("\n" + "=" * 70)
        print("📈 初回学習後の評価（Iteration 0）")
        print("=" * 70)
        initial_result = self.evaluate_model(
            model_name_or_path=initial_checkpoint,
            eval_data=eval_data,
            iteration=0,
            is_baseline=False
        )
        self.iteration_results.append(initial_result)
        
        # ステップ3: 反復学習ループ（動的データ生成 + 継続学習）
        previous_checkpoint = initial_checkpoint  # 前回のチェックポイントを保持
        
        for iteration in range(1, self.num_iterations + 1):
            print("\n" + "=" * 70)
            print(f"🔄 反復 {iteration}/{self.num_iterations}")
            print(f"   前回のチェックポイント: {previous_checkpoint}")
            print("=" * 70)
            
            # 前回の評価結果からincorrect/unsafeケースを抽出
            print(f"\n📊 Iteration {iteration-1}の評価結果から誤答ケースを抽出...")
            prev_result_path = self.output_dir / f"iteration_{iteration-1}_evaluation.json"
            
            with open(prev_result_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            
            # EvaluationResultオブジェクトを再構築
            from evaluator import EvaluationResult, DrugEvaluation
            prev_results = []
            for r in prev_data["results"]:
                drugs = [DrugEvaluation(**d) for d in r.get("drugs", [])]
                prev_results.append(EvaluationResult(
                    question=r["question"],
                    model_answer=r["model_answer"],
                    overall_label=r["overall_label"],
                    overall_is_harmful=r["overall_is_harmful"],
                    overall_score=r["overall_score"],
                    overall_reason=r["overall_reason"],
                    drugs=drugs,
                    timestamp=r["timestamp"],
                    expected_medications=r.get("expected_medications"),
                    correct_medications=r.get("correct_medications")  # ChatGPTの正しい薬剤リスト（JSON配列）
                ))
            
            # カリキュラム学習用データ作成
            # 誤答修正データ(A) + 成功例リハーサルデータ(B)
            train_data_path = self.create_curriculum_training_data(
                prev_results=prev_results,
                iteration=iteration
            )
            
            # LoRAファインチューニング（前回のチェックポイントから継続）
            config = LoRATrainingConfig()
            trainer = MedicalLoRATrainer(config)
            
            # 前回のチェックポイントから読み込み
            trainer.setup_model(previous_checkpoint=previous_checkpoint)
            trainer.setup_lora()
            
            # データセット準備
            train_dataset = trainer.prepare_dataset(train_data_path)
            
            # 学習実行（前回のチェックポイントから継続学習）
            checkpoint_path = trainer.train(
                train_dataset,
                iteration=iteration,
                previous_checkpoint=previous_checkpoint
            )
            
            # 次のiterationのために更新
            previous_checkpoint = checkpoint_path
            
            # メモリ解放
            del trainer
            torch.cuda.empty_cache()
            
            # 評価実行
            eval_result = self.evaluate_model(
                model_name_or_path=checkpoint_path,
                eval_data=eval_data,
                iteration=iteration,
                is_baseline=False
            )
            self.iteration_results.append(eval_result)
        
        # 最終レポート生成
        self.generate_final_report()
    
    def generate_final_report(self):
        """最終レポートの生成"""
        
        print("\n" + "=" * 70)
        print("📊 最終レポート")
        print("=" * 70)
        
        # 反復ごとのスコア推移
        print("\n【総合スコア・正答率・有害率の推移】")
        print("-" * 90)
        print(f"{'Iteration':<15} {'総合スコア':<20} {'正答率':<20} {'有害率':<20}")
        print("-" * 90)
        
        for result in self.iteration_results:
            if result["is_baseline"]:
                iteration_label = "Baseline"
            else:
                iteration_label = f"Iteration {result['iteration']}"
            
            metrics = result["metrics"]
            
            overall_mean = metrics['overall_score']['mean']
            overall_std = metrics['overall_score']['std']
            accuracy_rate = metrics.get('accuracy_rate', 0.0)
            harmful_rate = metrics.get('harmful_rate', 0.0)
            
            print(
                f"{iteration_label:<15} "
                f"{overall_mean:>6.3f} ± {overall_std:<6.3f}     "
                f"{accuracy_rate:>6.1%}              "
                f"{harmful_rate:>6.1%}"
            )
        
        print("-" * 90)
        
        # スコア改善の計算
        if len(self.iteration_results) > 1:
            initial_score = self.iteration_results[0]["metrics"]["overall_score"]["mean"]
            final_score = self.iteration_results[-1]["metrics"]["overall_score"]["mean"]
            improvement = final_score - initial_score
            improvement_pct = (improvement / initial_score) * 100 if initial_score > 0 else 0
            
            initial_harmful = self.iteration_results[0]["metrics"].get("harmful_rate", 0.0)
            final_harmful = self.iteration_results[-1]["metrics"].get("harmful_rate", 0.0)
            harmful_reduction = initial_harmful - final_harmful
            
            initial_label = "ベースライン" if self.iteration_results[0]["is_baseline"] else "初回学習後"
            
            print("\n【改善度】")
            print(f"  {initial_label}総合スコア: {initial_score:.3f}")
            print(f"  最終総合スコア:         {final_score:.3f}")
            print(f"  改善:                   {improvement:+.3f} ({improvement_pct:+.1f}%)")
            print(f"\n  {initial_label}有害率:     {initial_harmful:.1%}")
            print(f"  最終有害率:             {final_harmful:.1%}")
            print(f"  削減:                   {harmful_reduction:+.1%}")
        
        # レポートをファイルに保存
        report_path = self.output_dir / f"experiment_{self.experiment_id}_report.json"
        report_data = {
            "experiment_id": self.experiment_id,
            "base_model": self.base_model_name,
            "num_iterations": self.num_iterations,
            "results": self.iteration_results,
            "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat()
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 実験レポート保存: {report_path}")
        print("=" * 70)


def main():
    """メイン実行"""
    # .envから設定を読み込み
    base_model = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
    iterations = int(os.getenv("ITERATIONS", "2"))
    output_dir = os.getenv("OUTPUT_DIR", "results")
    num_train_samples = int(os.getenv("NUM_TRAIN_SAMPLES", "50"))
    num_eval_samples = int(os.getenv("NUM_EVAL_SAMPLES", "15"))  # 仕様書の15件に変更
    num_samples_per_question = int(os.getenv("NUM_SAMPLES_PER_QUESTION", "30"))  # N回サンプリング
    enable_baseline = os.getenv("ENABLE_BASELINE", "false").lower() == "true"  # デフォルト: false
    rehearsal_ratio = float(os.getenv("REHEARSAL_RATIO", "0.5"))  # 誤答数に対する成功例リハーサルデータの比率
    
    print("\n" + "=" * 70)
    print("📋 設定情報 (.envから読み込み)")
    print("=" * 70)
    print(f"ベースモデル: {base_model}")
    print(f"反復回数: {iterations}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"トレーニングサンプル数: {num_train_samples}")
    print(f"評価サンプル数: {num_eval_samples}")
    print(f"質問あたりのサンプリング回数: {num_samples_per_question}")
    print(f"ベースライン評価: {'有効' if enable_baseline else '無効'}")
    print(f"リハーサルデータ比率: {rehearsal_ratio}x (誤答数に対する倍率)")
    print("=" * 70)
    
    # ループの実行
    loop = IterativeTrainingLoop(
        base_model_name=base_model,
        num_iterations=iterations,
        output_dir=output_dir,
        num_samples_per_question=num_samples_per_question,
        enable_baseline=enable_baseline,
        rehearsal_ratio=rehearsal_ratio
    )
    
    loop.run()
    
    print("\n✅ 全ての処理が完了しました!")


if __name__ == "__main__":
    main()
