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
```

Образ опрашивает папку Images из репозитория раз в 2 секунды и выдает в нее TXT