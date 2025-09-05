#!/usr/bin/env bash
set -euo pipefail

APP_NAME="myapp"
PID_FILE="app.pid"
LOG_FILE="app.log"
PROTO_DIR="api/proto"
GENERATED_DIR="internal/generated"

echo ">>> Проверяем установку protoc"
if ! command -v protoc &> /dev/null; then
    echo ">>> Устанавливаем protobuf-compiler"
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get update
        sudo apt-get install -y protobuf-compiler
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y protobuf-compiler
    else
        echo ">>> Не удалось определить ОС для установки protoc"
        exit 1
    fi
fi

echo ">>> Устанавливаем Go плагины для protoc"
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Добавляем GOBIН в PATH (ОЧЕНЬ ВАЖНО!)
export PATH="$PATH:$(go env GOPATH)/bin"

echo ">>> Проверяем установку плагинов"
if ! command -v protoc-gen-go &> /dev/null; then
    echo ">>> Ошибка: protoc-gen-go не установлен"
    echo ">>> GOPATH: $(go env GOPATH)"
    echo ">>> PATH: $PATH"
    exit 1
fi

if ! command -v protoc-gen-go-grpc &> /dev/null; then
    echo ">>> Ошибка: protoc-gen-go-grpc не установлен"
    exit 1
fi

echo ">>> Плагины установлены:"
echo ">>> protoc-gen-go: $(which protoc-gen-go)"
echo ">>> protoc-gen-go-grpc: $(which protoc-gen-go-grpc)"

echo ">>> Обновляем репозиторий"
git fetch origin master
git reset --hard origin/master

echo ">>> Генерируем gRPC код"
if [[ -d "$PROTO_DIR" ]]; then
    mkdir -p "$GENERATED_DIR/image_search"

    echo ">>> Генерируем код из $PROTO_DIR/image_search/image_search.proto"
    protoc --go_out="$GENERATED_DIR" --go_opt=paths=source_relative \
           --go-grpc_out="$GENERATED_DIR" --go-grpc_opt=paths=source_relative \
           -I "$PROTO_DIR" \
           image_search/image_search.proto

    if [[ -f "$GENERATED_DIR/image_search/image_search.pb.go" ]]; then
        echo ">>> Генерация успешно завершена"
        ls -la "$GENERATED_DIR/image_search/"
    else
        echo ">>> Ошибка: файлы не сгенерировались"
        exit 1
    fi
else
    echo ">>> Proto директория не найдена: $PROTO_DIR"
    exit 1
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