#!/usr/bin/env bash
set -euo pipefail

APP_NAME="myapp"      # имя бинарника после сборки
PID_FILE="app.pid"
LOG_FILE="app.log"
PROTO_DIR="api/proto"
GENERATED_DIR="internal/generated"

echo ">>> Обновляем репозиторий"
git fetch origin master
git reset --hard origin/master

echo ">>> Генерируем gRPC код"
if [[ -f "$(which protoc)" ]]; then
    # Проверяем, установлен ли protoc
    if [[ -d "$PROTO_DIR" ]]; then
        echo ">>> Генерируем код из .proto файлов"
        protoc --go_out="$GENERATED_DIR" --go_opt=paths=source_relative \
               --go-grpc_out="$GENERATED_DIR" --go-grpc_opt=paths=source_relative \
               -I "$PROTO_DIR" \
               "$PROTO_DIR"/image_search/image_search.proto
        echo ">>> Генерация gRPC кода завершена"
    else
        echo ">>> Директория $PROTO_DIR не найдена, пропускаем генерацию"
    fi
else
    echo ">>> protoc не установлен, пропускаем генерацию gRPC кода"
    echo ">>> Установите: brew install protobuf (macOS) или apt-get install protobuf-compiler (Linux)"
fi

echo ">>> Обновляем зависимости"
go mod tidy

echo ">>> Собираем приложение"
go build -o "$APP_NAME" cmd/app/main.go

# Останавливаем прошлый процесс, если есть
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo ">>> Останавливаем предыдущий процесс (PID $OLD_PID)"
        kill "$OLD_PID" || true
        sleep 2  # Даем процессу время на graceful shutdown
    fi
    rm -f "$PID_FILE"
fi

echo ">>> Запускаем приложение"
nohup "./$APP_NAME" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo ">>> Приложение запущено (PID $NEW_PID)"
echo ">>> Логи: tail -f $LOG_FILE"