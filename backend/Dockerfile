FROM python:3.11

WORKDIR /app

COPY pyproject.toml /app

COPY . /app

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn" , "src.api.main:app" , "--host", "0.0.0.0" , "--port", "8000"]

