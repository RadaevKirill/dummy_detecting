FROM python:3.8-slim-buster

WORKDIR /app
ENV BASE_DIR=/app
COPY . /app

RUN pip3 install --no-cache-dir --upgrade pip && pip3 install --no-cache-dir -r requirements.txt
RUN chmod +x /app/__main__.py