# dummy_detecting

Команда:

Сапов
Радаев
Милютин


```shell
# Сборка Docker образа
docker build -f ./Dockerfile -t python3.8-slim-buster-dummy-detection .
 
 # Запуск докер образа
docker run -v ./images:/app/images python3.8-slim-buster-dummy-detection python /app/__main__.py

#Запуск API
docker run -p 8000:8000  --mount type=bind,source=абсолютный путь до папки с изображениями,target=/app/images python3.8-slim-buster-dummy-detection uvicorn server:app --reload --host 
0.0.0.0 --port 8000
```
Абсолютный путь поправим
Образ опрашивает папку Images из репозитория раз в 2 секунды и выдает в нее TXT