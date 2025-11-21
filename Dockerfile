FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# システムパッケージの更新とPythonのインストール
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pipのアップグレード
RUN python -m pip install --upgrade pip setuptools wheel

# 作業ディレクトリの作成
WORKDIR /workspace

# 依存関係ファイルのコピーとインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# プロジェクトファイルのコピー
COPY . .

# 非rootユーザーの作成
RUN useradd -m -s /bin/bash vscode && \
    chown -R vscode:vscode /workspace

USER vscode

# デフォルトコマンド
CMD ["/bin/bash"]
