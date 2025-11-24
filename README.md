# medical-llm-mle
医療LLMファインチューニング課題

## 📊 実験レポート

実験結果と詳細な分析については [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) をご覧ください。

## 🐳 Dockerでの起動方法

### 前提条件
- Docker Desktop または Docker Engine
- NVIDIA GPU と NVIDIA Container Toolkit

### クイックスタート

1. **環境変数の設定**
```powershell
# .env.secretsファイルを作成してAPIキーを設定
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env.secrets
echo "HF_TOKEN=your_huggingface_token_here" >> .env.secrets
```

2. **Dockerイメージのビルドと起動**
```powershell
# イメージのビルドとコンテナ起動
docker-compose up --build
```

3. **コンテナ内でコマンドを実行**
```powershell
# 別のターミナルでコンテナに接続
docker-compose exec medical-llm bash

# コンテナ内で学習を実行
python src/main.py
```

開発ではdevcontainerを使用しているので、devcontainerしをうしても起動できます
