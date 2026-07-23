FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    STAGE1_TEMPLATE=/data/templates/base.xlsx \
    STAGE1_OUTPUT_DIR=/data/outputs

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src

RUN mkdir -p /data/templates /data/outputs

ENTRYPOINT ["python", "src/stage1_close_report.py"]
