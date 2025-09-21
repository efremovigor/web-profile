#!/usr/bin/env bash
set -euo pipefail

APP_NAME="app"
PROTO_DIR="api/proto"
GENERATED_DIR="internal/generated"
PYTHON_APP_DIR="relative/image_search"
PYTHON_REQUIREMENTS="$PYTHON_APP_DIR/requirements.txt"
GO_SERVICE="web-profile-service"
PYTHON_SERVICE="image-service"

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
export PATH="$PATH:$(go env GOPATH)/bin"

echo ">>> Обновляем репозиторий"
git fetch origin master
git reset --hard origin/master

echo ">>> Генерируем gRPC код для Go"
mkdir -p "$GENERATED_DIR/image_search"
protoc --go_out="$GENERATED_DIR" --go_opt=paths=source_relative \
       --go-grpc_out="$GENERATED_DIR" --go-grpc_opt=paths=source_relative \
       -I "$PROTO_DIR" \
       image_search/image_search.proto

echo ">>> Генерируем gRPC код для Python"
cd "$PYTHON_APP_DIR"
python3 -m grpc_tools.protoc -I../../api/proto/image_search/ \
       --python_out=. \
       --grpc_python_out=. \
       ../../api/proto/image_search/image_search.proto
cd - > /dev/null

echo ">>> Устанавливаем Python зависимости"
# Используем python3 и pip3 чтобы избежать проблем с версиями
if command -v pip3 &> /dev/null; then
    pip3 install -r "$PYTHON_REQUIREMENTS"
elif command -v python3 -m pip &> /dev/null; then
    python3 -m pip install -r "$PYTHON_REQUIREMENTS"
else
    echo ">>> Ошибка: не найден pip3 или python3 -m pip"
    echo ">>> Устанавливаем pip3"
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get install -y python3-pip
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y python3-pip
    fi
    pip3 install -r "$PYTHON_REQUIREMENTS"
fi

echo ">>> Обновляем Go зависимости"
go mod tidy

echo ">>> Собираем Go приложение"
go build -o "$APP_NAME" cmd/app/main.go

echo ">>> Перезапускаем сервисы"
sudo systemctl daemon-reload

echo ">>> Перезапускаем Go сервис"
sudo systemctl restart "$GO_SERVICE"
sudo systemctl status "$GO_SERVICE" --no-pager -l

echo ">>> Перезапускаем Python сервис"
sudo systemctl restart "$PYTHON_SERVICE"
sudo systemctl status "$PYTHON_SERVICE" --no-pager -l

echo ">>> Деплой завершен успешно!"
echo ">>> Логи Go: sudo journalctl -u $GO_SERVICE -f"
echo ">>> Логи Python: sudo journalctl -u $PYTHON_SERVICE -f"