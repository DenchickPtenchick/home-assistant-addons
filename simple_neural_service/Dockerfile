FROM python:3.11-slim

# Устанавливаем необходимые зависимости
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Создаём рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем tflite-runtime с поддержкой Flex
RUN pip install tflite-runtime[flex]

# Копируем остальные файлы приложения
COPY . .

# Запускаем приложение
CMD ["python", "app.py"]
