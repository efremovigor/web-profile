#!/usr/bin/env bash
set -euo pipefail

APP_NAME="myapp"
PID_FILE="app.pid"
PYTHON_PID_FILE="python_app.pid"
LOG_FILE="app.log"
PYTHON_LOG_FILE="python_app.log"
PROTO_DIR="api/proto"
GENERATED_DIR="internal/generated"
PYTHON_APP_DIR="relative/image_search"
PYTHON_REQUIREMENTS="$PYTHON_APP_DIR/requirements.txt"

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

echo ">>> Проверяем установку Python"
if ! command -v python3 &> /dev/null; then
    echo ">>> Устанавливаем Python 3"
    if [[ -f /etc/debian_version ]]; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    elif [[ -f /etc/redhat-release ]]; then
        sudo yum install -y python3 python3-pip
    else
        echo ">>> Не удалось определить ОС для установки Python"
        exit 1
    fi
fi

# Создаем симлинк python -> python3 если его нет
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        ln -s $(which python3) /usr/local/bin/python
    fi
fi

echo ">>> Устанавливаем Go плагины для protoc"
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Добавляем GOPATH в PATH
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

echo ">>> Генерируем gRPC код для Go"
if [[ -d "$PROTO_DIR" ]]; then
    mkdir -p "$GENERATED_DIR/image_search"

    echo ">>> Генерируем код из $PROTO_DIR/image_search/image_search.proto"
    protoc --go_out="$GENERATED_DIR" --go_opt=paths=source_relative \
           --go-grpc_out="$GENERATED_DIR" --go-grpc_opt=paths=source_relative \
           -I "$PROTO_DIR" \
           image_search/image_search.proto

    if [[ -f "$GENERATED_DIR/image_search/image_search.pb.go" ]]; then
        echo ">>> Генерация Go кода успешно завершена"
        ls -la "$GENERATED_DIR/image_search/"
    else
        echo ">>> Ошибка: Go файлы не сгенерировались"
        exit 1
    fi
else
    echo ">>> Proto директория не найдена: $PROTO_DIR"
    exit 1
fi

echo ">>> Генерируем gRPC код для Python"
if [[ -d "$PYTHON_APP_DIR" ]]; then
    echo ">>> Генерируем Python код из $PROTO_DIR/image_search/image_search.proto"
    cd "$PYTHON_APP_DIR"

    # Устанавливаем необходимые Python пакеты для генерации
    pip3 install grpcio-tools

    python3 -m grpc_tools.protoc -I../../api/proto/image_search/ \
           --python_out=. \
           --grpc_python_out=. \
           ../../api/proto/image_search/image_search.proto
    cd - > /dev/null

    if [[ -f "$PYTHON_APP_DIR/image_search_pb2.py" ]] && [[ -f "$PYTHON_APP_DIR/image_search_pb2_grpc.py" ]]; then
        echo ">>> Генерация Python кода успешно завершена"
        ls -la "$PYTHON_APP_DIR/" | grep image_search
    else
        echo ">>> Ошибка: Python файлы не сгенерировались"
        exit 1
    fi
else
    echo ">>> Python app директория не найдена: $PYTHON_APP_DIR"
    exit 1
fi

echo ">>> Устанавливаем Python зависимости"
if [[ -f "$PYTHON_REQUIREMENTS" ]]; then
    echo ">>> Устанавливаем зависимости из $PYTHON_REQUIREMENTS"
    pip3 install -r "$PYTHON_REQUIREMENTS"
else
    echo ">>> requirements.txt не найден: $PYTHON_REQUIREMENTS"
    exit 1
fi

echo ">>> Обновляем Go зависимости"
go mod tidy

echo ">>> Собираем Go приложение"
go build -o "$APP_NAME" cmd/app/main.go

# Останавливаем прошлые процессы Go, если есть
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo ">>> Останавливаем предыдущий Go процесс (PID $OLD_PID)"
        kill "$OLD_PID" || true
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Останавливаем прошлые процессы Python, если есть
if [[ -f "$PYTHON_PID_FILE" ]]; then
    OLD_PYTHON_PID=$(cat "$PYTHON_PID_FILE")
    if ps -p "$OLD_PYTHON_PID" > /dev/null 2>&1; then
        echo ">>> Останавливаем предыдущий Python процесс (PID $OLD_PYTHON_PID)"
        kill "$OLD_PYTHON_PID" || true
        sleep 2
    fi
    rm -f "$PYTHON_PID_FILE"
fi

echo ">>> Запускаем Go приложение"
nohup "./$APP_NAME" > "$LOG_FILE" 2>&1 &
GO_PID=$!
echo "$GO_PID" > "$PID_FILE"
echo ">>> Go приложение запущено (PID $GO_PID)"
echo ">>> Логи Go: tail -f $LOG_FILE"

echo ">>> Запускаем Python демон"
cd "$PYTHON_APP_DIR"
nohup python3 image_search.py > "../$PYTHON_LOG_FILE" 2>&1 &
PYTHON_PID=$!
cd - > /dev/null
echo "$PYTHON_PID" > "$PYTHON_PID_FILE"
echo ">>> Python демон запущен (PID $PYTHON_PID)"
echo ">>> Логи Python: tail -f $PYTHON_LOG_FILE"

echo ">>> Деплой завершен успешно!"