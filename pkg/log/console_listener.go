package log

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"reflect"
	"time"
)

type MessageType string

const (
	ExceptionMessage MessageType = "exception"
	InfoMessage      MessageType = "info"
	MetricMessage    MessageType = "metric"
)

type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
)

type ConsoleListener struct {
	out *os.File
}

func NewConsoleListener() *ConsoleListener {
	return &ConsoleListener{out: os.Stdout}
}

func (c *ConsoleListener) Log(msg Message) {
	_, err := fmt.Fprint(c.out, msg.Decorate()+"\r\n")
	if err != nil {
		log.Print(Exception(fmt.Sprintf("unable to send data to console: %s", err), nil).Decorate())
	}
}

type Message struct {
	Time    time.Time
	Level   Level
	Message interface{}
	Request *http.Request
	Type    MessageType
}

func (msg Message) Decorate() string {
	return msg.fill()
}

func (msg Message) fill() string {
	return msg.Message.(string)
}

func Exception(msg interface{}, r *http.Request) Message {
	return Message{
		Time:    time.Now(),
		Level:   WARN,
		Message: msg,
		Request: r,
		Type:    ExceptionMessage,
	}
}

func Error(msg interface{}, r *http.Request) Message {
	return Message{
		Time:    time.Now(),
		Level:   ERROR,
		Message: msg,
		Request: r,
		Type:    ExceptionMessage,
	}
}

func Info(msg interface{}, r *http.Request) Message {
	return Message{
		Time:    time.Now(),
		Level:   INFO,
		Message: msg,
		Request: r,
		Type:    InfoMessage,
	}
}

func Debug(msg interface{}, r *http.Request) Message {
	return Message{
		Time:    time.Now(),
		Level:   DEBUG,
		Message: msg,
		Request: r,
		Type:    InfoMessage,
	}
}

func NewLogger(minLevel Level, listeners []Listener) *Logger {
	l := &Logger{
		minLevel:  minLevel,
		listeners: make(map[string]Listener),
	}

	for _, listener := range listeners {
		l.listeners[reflect.TypeOf(listener).Elem().Name()] = listener
	}

	return l
}

type Listener interface {
	Log(msg Message)
}
type Logger struct {
	listeners map[string]Listener
	minLevel  Level
}

func (l *Logger) Write(p []byte) (n int, err error) {
	l.Log(Debug(string(p), nil))
	return len(p), nil
}

func (l *Logger) Log(m Message) {
	if m.Level < l.minLevel {
		return
	}
	for _, listener := range l.listeners {
		listener.Log(m)
	}
}
