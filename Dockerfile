FROM python:3.12-slim

# 컨테이너 안에서 root 로 돌 이유가 없다. 사내 배포 검토에서 가장 먼저 지적받는 항목.
RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000

# 이미지에 curl 을 넣지 않기 위해 httpx 로 점검한다(이미 의존성에 있다).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code==200 else 1)"]

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
