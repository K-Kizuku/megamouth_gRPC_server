# ベースイメージを指定
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# pipenvのインストール
RUN pip install --no-cache-dir pipenv

# PipfileとPipfile.lockをコンテナにコピー
COPY Pipfile Pipfile.lock /app/

# Pythonパッケージのインストール
RUN pipenv install --system --deploy

# アプリケーションのソースコードをコンテナの作業ディレクトリにコピー
COPY . /app

# コンテナが起動した際に実行するコマンドを指定
CMD ["python", "./proto/server.py"]

# ポート50052を公開
EXPOSE 50052
