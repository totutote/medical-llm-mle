"""
OpenAI API の権限とアクセス可能なモデルを確認するスクリプト
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

def check_api_permissions():
    """APIキーの権限とモデルリストを確認"""
    
    # .envと.env.secretsの両方を読み込み
    load_dotenv()  # .env
    load_dotenv('.env.secrets', override=True)  # .env.secrets (優先)
    
    # APIキーの取得
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 環境変数 OPENAI_API_KEY が設定されていません")
        print("   .env または .env.secrets ファイルを確認してください")
        return
    
    print(f"✅ APIキーが設定されています: {api_key[:8]}...{api_key[-4:]}")
    print()
    
    # OpenAI クライアントの初期化
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI クライアントの初期化に成功")
    except Exception as e:
        print(f"❌ クライアントの初期化に失敗: {e}")
        return
    
    print()
    print("=" * 60)
    print("1. モデルリスト取得テスト (List Models)")
    print("=" * 60)
    
    try:
        models = client.models.list()
        print(f"✅ モデルリスト取得成功: {len(models.data)} 個のモデルが利用可能")
        print("\n利用可能なモデル（GPTのみ）:")
        gpt_models = [m.id for m in models.data if 'gpt' in m.id.lower()]
        for model in sorted(gpt_models):
            print(f"  - {model}")
    except Exception as e:
        print(f"❌ モデルリスト取得失敗: {e}")
    
    print()
    print("=" * 60)
    print("2. Chat Completions API テスト")
    print("=" * 60)
    
    # テストするモデルのリスト
    test_models = ["gpt-4.1-mini", "gpt-4o-mini"]
    
    for model_name in test_models:
        print(f"\nモデル: {model_name}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "こんにちは"}
                ],
                max_completion_tokens=10
            )
            print("  ✅ Chat Completions API 成功")
            content = response.choices[0].message.content
            print(f"  レスポンス: '{content}'")
            print(f"  レスポンス長: {len(content) if content else 0} 文字")
            print(f"  使用トークン: 入力={response.usage.prompt_tokens}, 出力={response.usage.completion_tokens}")
            print(f"  finish_reason: {response.choices[0].finish_reason}")
            if not content or len(content.strip()) == 0:
                print("  ⚠️  空または空白のみのレスポンス")
                # 生のレスポンスを確認
                print(f"  生データ: content={repr(content)}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print("  ⚠️  モデルが存在しません")
            elif "401" in error_msg:
                print(f"  ❌ 権限エラー: {error_msg}")
            elif "429" in error_msg:
                print(f"  ❌ クォータ不足: {error_msg}")
            else:
                print(f"  ❌ エラー: {error_msg}")
    
    print()
    print("=" * 60)
    print("3. 特定モデルの詳細取得テスト")
    print("=" * 60)
    
    for model_name in ["gpt-5-nano", "gpt-5-mini"]:
        print(f"\nモデル: {model_name}")
        try:
            model_info = client.models.retrieve(model_name)
            print("  ✅ モデル情報取得成功")
            print(f"  ID: {model_info.id}")
            print(f"  Owner: {model_info.owned_by}")
            print(f"  Created: {model_info.created}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print("  ⚠️  モデルが存在しません")
            else:
                print(f"  ❌ エラー: {error_msg}")
    
    print()
    print("=" * 60)
    print("診断完了")
    print("=" * 60)

if __name__ == "__main__":
    check_api_permissions()
