#!/usr/bin/env bash
set -euo pipefail

APP_NAME="myapp"      # имя бинарника после сборки
PID_FILE="app.pid"
LOG_FILE="app.log"

echo ">>> Обновляем репозиторий"
git fetch origin master
git reset --hard origin/master

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
    fi
    rm -f "$PID_FILE"
fi

echo ">>> Запускаем приложение"
nohup "./$APP_NAME" > "$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo ">>> Приложение запущено (PID $NEW_PID)"