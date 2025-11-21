def format_instruction(symptom: str) -> str:
    """
    症状テキストからモデル用プロンプト(instruction)を生成
    Args:
        symptom: 症状テキスト
    Returns:
        instruction文字列
    """
    return f"""症状: {symptom}

推奨される薬剤を3つ挙げてください。各薬剤について100字以内で理由を説明してください。
薬剤名と理由のみを記載し、他の文章は不要です。

フォーマット:
1. [薬剤名] - [理由]
2. [薬剤名] - [理由]
3. [薬剤名] - [理由]

回答:
1. """

def format_medications_list(medications: list) -> str:
    """
    薬剤リストを番号付き箇条書き形式で整形
    Args:
        medications: 薬剤辞書のリスト [{'name': str, 'reason': str}, ...] (最大3件)
    Returns:
        番号付き箇条書きテキスト
    """
    meds = medications[:3]
    medications_parts = []
    if len(meds) > 0:
        medications_parts.append(f"{meds[0]['name']} - {meds[0]['reason']}")
    if len(meds) > 1:
        medications_parts.append(f"2. {meds[1]['name']} - {meds[1]['reason']}")
    if len(meds) > 2:
        medications_parts.append(f"3. {meds[2]['name']} - {meds[2]['reason']}")
    return "\n".join(medications_parts)

"""
医療症状→薬剤候補データセットの生成と管理

このモジュールは研究目的のみで使用されます。
実際の診療・服薬指示には使用しないでください。
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random


# サンプル医療データ（研究目的のみ）
# 仕様書の15件の症状パターンに基づく
SAMPLE_MEDICAL_DATA = [
    {
        "symptom": "軽い頭痛のみがあり、発熱や他の症状は特にない成人。",
        "medications": [
            {"name": "アセトアミノフェン", "reason": "解熱鎮痛作用があり、胃腸への負担が少ないため軽度の頭痛に適しています。"},
            {"name": "イブプロフェン", "reason": "NSAIDsであり、抗炎症・鎮痛作用を持ち、頭痛に効果的です。"},
            {"name": "ロキソプロフェン", "reason": "速効性のあるNSAIDsで、鎮痛効果が高く頭痛の緩和に有効です。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "38.5℃前後の高熱と悪寒、全身倦怠感を伴う風邪様症状のある成人。",
        "medications": [
            {"name": "アセトアミノフェン", "reason": "解熱作用に優れ、胃腸障害のリスクが低いため高熱時の第一選択薬として適しています。"},
            {"name": "イブプロフェン", "reason": "解熱・鎮痛・抗炎症作用があり、風邪症状による発熱や全身の痛みに効果的です。"},
            {"name": "ロキソプロフェン", "reason": "速効性の解熱鎮痛作用があり、高熱と倦怠感の軽減に有効です。"}
        ],
        "severity": "中等度",
        "patient_type": "成人"
    },
    {
        "symptom": "乾いた空咳が続いていて、痰はほとんど出ないが、夜間に咳き込みやすい成人。",
        "medications": [
            {"name": "デキストロメトルファン", "reason": "中枢性鎮咳作用があり、乾性咳嗽の抑制に効果的で依存性が低い薬剤です。"},
            {"name": "コデイン", "reason": "強力な鎮咳作用を持ち、夜間の咳き込みを抑えて睡眠の質を改善します。"},
            {"name": "リン酸コデイン", "reason": "持続的な鎮咳効果があり、頑固な乾性咳嗽の症状緩和に有効です。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "透明でサラサラした鼻水とくしゃみ、目のかゆみを伴う季節性アレルギー性鼻炎が疑われる成人。",
        "medications": [
            {"name": "フェキソフェナジン", "reason": "第2世代抗ヒスタミン薬で、眠気が少なくアレルギー症状全般に効果的です。"},
            {"name": "セチリジン", "reason": "抗アレルギー作用が強く、鼻水・くしゃみ・目のかゆみを効果的に抑制します。"},
            {"name": "ロラタジン", "reason": "1日1回の服用で持続的な抗アレルギー効果があり、眠気の副作用が少ない薬剤です。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "濃い色の粘調な痰を伴う湿った咳が続いている成人。",
        "medications": [
            {"name": "カルボシステイン", "reason": "気道粘液を正常化し、痰の粘度を下げて排出を促進する去痰薬です。"},
            {"name": "アンブロキソール", "reason": "痰の分泌を促進しつつ粘度を下げ、気道からの排出を容易にします。"},
            {"name": "ブロムヘキシン", "reason": "粘調な痰を溶解し、気道の線毛運動を促進して痰の排出を助けます。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "生理痛による下腹部の鈍い痛みがあり、軽度の腰痛もある若年女性。",
        "medications": [
            {"name": "イブプロフェン", "reason": "プロスタグランジン合成を抑制し、子宮収縮による痛みを効果的に緩和します。"},
            {"name": "ロキソプロフェン", "reason": "速効性のある鎮痛作用で、生理痛と腰痛の両方に効果を発揮します。"},
            {"name": "アセトアミノフェン", "reason": "胃腸への負担が少なく、軽度から中等度の生理痛に適した鎮痛薬です。"}
        ],
        "severity": "軽度",
        "patient_type": "若年女性"
    },
    {
        "symptom": "片頭痛と思われる、片側性でズキズキする頭痛と光・音過敏を訴える成人。",
        "medications": [
            {"name": "スマトリプタン", "reason": "トリプタン系薬剤で、片頭痛の特異的治療薬として血管収縮と神経炎症抑制作用があります。"},
            {"name": "ゾルミトリプタン", "reason": "片頭痛発作に特化した薬剤で、脳血管の異常拡張を抑え、頭痛を速やかに緩和します。"},
            {"name": "ロキソプロフェン", "reason": "軽度から中等度の片頭痛に有効で、抗炎症・鎮痛作用により症状を軽減します。"}
        ],
        "severity": "中等度",
        "patient_type": "成人"
    },
    {
        "symptom": "軽度の関節痛と筋肉痛を伴う、インフルエンザからの回復期にある成人。",
        "medications": [
            {"name": "アセトアミノフェン", "reason": "解熱鎮痛作用があり、インフルエンザ回復期の関節痛・筋肉痛に安全に使用できます。"},
            {"name": "イブプロフェン", "reason": "抗炎症作用により、ウイルス感染後の炎症性の痛みを効果的に緩和します。"},
            {"name": "ロキソプロフェン", "reason": "鎮痛・抗炎症作用が強く、回復期の関節痛や筋肉痛の症状改善に有効です。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "のどの痛みと軽い発熱があり、食べ物を飲み込みにくいと訴える成人。",
        "medications": [
            {"name": "トラネキサム酸", "reason": "抗炎症作用と抗プラスミン作用により、咽頭炎による腫れと痛みを軽減します。"},
            {"name": "アズレン含嗽剤", "reason": "抗炎症作用を持ち、うがいにより咽頭粘膜の炎症を直接的に緩和します。"},
            {"name": "ポビドンヨード", "reason": "殺菌・消毒作用があり、咽頭の細菌やウイルスを減少させ炎症の悪化を防ぎます。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "慢性的な腰痛があり、ときどき痛みが強くなるデスクワーク中心の中年男性。",
        "medications": [
            {"name": "ロキソプロフェン", "reason": "NSAIDsとして抗炎症・鎮痛作用があり、慢性腰痛の急性増悪時に効果的です。"},
            {"name": "セレコキシブ", "reason": "COX-2選択的阻害薬で、胃腸障害が少なく慢性疼痛の長期管理に適しています。"},
            {"name": "メチコバール", "reason": "ビタミンB12製剤で、神経の修復を促進し慢性的な神経性腰痛の改善に寄与します。"}
        ],
        "severity": "軽度から中等度",
        "patient_type": "中年男性"
    },
    {
        "symptom": "夜間に症状が悪化する慢性咳嗽を訴える、既往に気管支喘息のない成人。",
        "medications": [
            {"name": "デキストロメトルファン", "reason": "中枢性鎮咳作用により夜間の咳を抑制し、睡眠の質を改善します。"},
            {"name": "カルボシステイン", "reason": "気道粘液の正常化により、咳の原因となる分泌物の排出を促進します。"},
            {"name": "アンブロキソール", "reason": "気道分泌促進と粘液溶解作用により、慢性的な咳の症状を緩和します。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    },
    {
        "symptom": "鼻づまりが強く、頭重感や顔面の圧迫感を伴う副鼻腔炎が疑われる成人。",
        "medications": [
            {"name": "クラリスロマイシン", "reason": "マクロライド系抗生物質で、副鼻腔炎の原因菌に抗菌作用を示し、抗炎症効果もあります。"},
            {"name": "カルボシステイン", "reason": "副鼻腔内の粘液を正常化し、排泄を促進することで鼻づまりを改善します。"},
            {"name": "フェキソフェナジン", "reason": "抗ヒスタミン作用により、アレルギー性要因による鼻粘膜の腫れを軽減します。"}
        ],
        "severity": "中等度",
        "patient_type": "成人"
    },
    {
        "symptom": "足首の捻挫直後で、腫れと痛みがあり、荷重すると強く痛む成人。",
        "medications": [
            {"name": "ロキソプロフェン", "reason": "内服NSAIDsとして抗炎症・鎮痛作用が強く、急性期の腫れと痛みを全身的に抑制します。"},
            {"name": "インドメタシン外用", "reason": "局所的な抗炎症作用が強く、患部に直接塗布することで腫れと痛みを軽減します。"},
            {"name": "ジクロフェナク外用", "reason": "経皮吸収型NSAIDsで、患部への浸透性が高く炎症と疼痛を効果的に緩和します。"}
        ],
        "severity": "中等度",
        "patient_type": "成人"
    },
    {
        "symptom": "手指の関節に朝のこわばりを感じ、日中も軽い痛みが続く中年女性。",
        "medications": [
            {"name": "セレコキシブ", "reason": "COX-2選択的阻害薬で、関節炎の痛みと炎症を抑え、胃腸障害のリスクが低い薬剤です。"},
            {"name": "ロキソプロフェン", "reason": "抗炎症・鎮痛作用により、関節のこわばりと痛みを効果的に軽減します。"},
            {"name": "メトトレキサート(医師処方必要)", "reason": "関節リウマチの疾患修飾性抗リウマチ薬で、関節破壊の進行を抑制する根本的治療薬です。"}
        ],
        "severity": "軽度から中等度",
        "patient_type": "中年女性"
    },
    {
        "symptom": "胃もたれと軽い上腹部痛、胸焼け感があり、食べ過ぎた翌日に症状が出た成人。",
        "medications": [
            {"name": "ファモチジン", "reason": "H2受容体拮抗薬で、胃酸分泌を抑制し胃もたれや胸焼けを速やかに改善します。"},
            {"name": "オメプラゾール", "reason": "プロトンポンプ阻害薬として強力な胃酸分泌抑制作用があり、上腹部痛と胸焼けに有効です。"},
            {"name": "ランソプラゾール", "reason": "PPIとして持続的に胃酸を抑え、胃粘膜を保護し消化器症状を改善します。"}
        ],
        "severity": "軽度",
        "patient_type": "成人"
    }
]


def create_training_data(output_dir: str = "data", num_samples: int = None) -> str:
    """
    ファインチューニング用のトレーニングデータを生成
    
    Args:
        output_dir: 出力ディレクトリ
        num_samples: 生成するサンプル数（Noneの場合は全データを使用）
    
    Returns:
        生成されたファイルのパス
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_data = []
    
    # num_samplesが指定されていない、または利用可能なデータ数より多い場合は全データを使用
    samples_to_use = SAMPLE_MEDICAL_DATA if num_samples is None else SAMPLE_MEDICAL_DATA[:min(num_samples, len(SAMPLE_MEDICAL_DATA))]
    
    for sample in samples_to_use:
        # instruction生成を関数化
        instruction = format_instruction(sample['symptom'])
        
        # 共通関数で薬剤リストを生成
        response = format_medications_list(sample['medications'])
        
        training_data.append({
            "instruction": instruction,
            "input": "",
            "output": response  # 番号付き箇条書きテキスト
        })
    
    # JSONLファイルとして保存
    output_path = os.path.join(output_dir, "training_data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ トレーニングデータを生成しました: {output_path}")
    print(f"   サンプル数: {len(training_data)} (利用可能データ: {len(SAMPLE_MEDICAL_DATA)})")
    
    return output_path


def create_evaluation_data(output_dir: str = "data", num_samples: int = 20) -> str:
    """
    評価用のテストデータを生成（トレーニングデータとは別）
    
    Args:
        output_dir: 出力ディレクトリ
        num_samples: 生成するサンプル数
    
    Returns:
        生成されたファイルのパス
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # トレーニングデータとは異なるサンプルを選択
    eval_samples = random.sample(SAMPLE_MEDICAL_DATA, min(num_samples, len(SAMPLE_MEDICAL_DATA)))
    
    eval_data = []
    for sample in eval_samples:
        eval_data.append({
            "symptom": sample['symptom'],
            "expected_medications": [med['name'] for med in sample['medications']],
            "patient_type": sample['patient_type']
        })
    
    output_path = os.path.join(output_dir, "evaluation_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 評価データを生成しました: {output_path}")
    print(f"   サンプル数: {len(eval_data)}")
    
    return output_path


def load_training_data(file_path: str) -> List[Dict]:
    """トレーニングデータを読み込む"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_evaluation_data(file_path: str) -> List[Dict]:
    """評価データを読み込む"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_correct_medication_from_sample_data(symptom: str) -> List[Dict]:
    """
    SAMPLE_MEDICAL_DATAから症状に対応する正しい薬剤情報を取得
    
    Args:
        symptom: 評価対象の症状テキスト
    
    Returns:
        正しい薬剤情報のリスト [{"name": str, "reason": str}, ...]
        見つからない場合は空リスト
    """
    for sample in SAMPLE_MEDICAL_DATA:
        if sample["symptom"] == symptom:
            return sample["medications"]
    
    # 完全一致しない場合は部分一致を試みる
    for sample in SAMPLE_MEDICAL_DATA:
        # 症状テキストの最初の30文字で部分一致チェック
        if symptom[:30] in sample["symptom"] or sample["symptom"][:30] in symptom:
            return sample["medications"]
    
    return []


if __name__ == "__main__":
    # データセット生成のテスト
    random.seed(42)
    
    print("=" * 60)
    print("医療データセット生成スクリプト")
    print("=" * 60)
    print("\n⚠️  このデータは研究目的のみで使用されます。")
    print("   実際の診療・服薬指示には使用しないでください。\n")
    
    # トレーニングデータ生成
    train_path = create_training_data(num_samples=50)
    
    # 評価データ生成
    eval_path = create_evaluation_data(num_samples=20)
    
    print("\n" + "=" * 60)
    print("データセット生成完了")
    print("=" * 60)
