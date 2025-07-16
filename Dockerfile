# ベースとなる公式Pythonイメージを選択
FROM python:3.11-slim

# システムを最新の状態にし、必要なライブラリとTesseract OCRをインストール
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     tesseract-ocr \
#     tesseract-ocr-jpn \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y build-essential tesseract-ocr tesseract-ocr-jpn --no-install-recommends && rm -rf /var/lib/apt/lists/*


# 作業ディレクトリを設定
WORKDIR /app

# 必要なPythonライブラリのリストをコピーし、インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトの全ファイルを作業ディレクトリにコピー
COPY . .

# Botを実行するコマンド
CMD ["python", "bot.py"]
