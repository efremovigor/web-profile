package internal

import (
	"github.com/joho/godotenv"
	"log"
	"os"
	"strconv"
	"time"
	"web_profile/pkg/consts/web"
)

const envPath = "./config/.env"

type Config struct {
	Env Env
}

type Env struct {
	Type               string
	Domain             string
	TelegramToken      string
	TelegramChatId     string
	ImageSearchHost    string
	ImageSearchPort    string
	ImageSearchTimeout string
}

func (c Config) IsDev() bool {
	return c.Env.Type != "prod"
}

func (c Config) GetHttpsUrl() string {
	if c.IsDev() {
		return web.HTTPProtocol + "://" + c.GetHttpsSocket()
	}
	return web.HTTPSProtocol + "://" + c.GetHttpsSocket()
}

func (c Config) GetHttpSocket() string {
	if c.IsDev() {
		return "127.0.0.1:" + strconv.Itoa(web.DevHTTPPort)
	}
	return ":" + strconv.Itoa(web.HTTPPort)
}

func (c Config) GetHttpsSocket() string {
	if c.IsDev() {
		return "127.0.0.1:" + strconv.Itoa(web.DevHTTPSPort)
	}
	return c.Env.Domain + ":" + strconv.Itoa(web.HTTPSPort)
}

func (c Config) GetDomain() string {
	if c.IsDev() {
		return "127.0.0.1:" + strconv.Itoa(web.DevHTTPSPort)
	}
	return c.Env.Domain
}

func (c Config) GetTelegramChatId() string {
	return c.Env.TelegramChatId
}

func (c Config) GetTelegramToken() string {
	return c.Env.TelegramToken
}

func (c Config) GetImageSearchHost() string {
	return c.Env.ImageSearchHost
}

func (c Config) GetImageSearchPort() int {
	val, err := strconv.Atoi(c.Env.ImageSearchPort)
	if err != nil {
		log.Fatalf("Error loading IMAGE_SEARCH_PORT value")
	}
	return val
}

func (c Config) GetImageSearchTimeout() time.Duration {
	return 30 * time.Second
}

func NewConfig() Config {
	err := godotenv.Load(envPath)

	if err != nil {
		log.Fatalf("Error loading .env_config file")
	}

	return Config{
		Env: Env{
			Domain:             os.Getenv("DOMAIN"),
			Type:               os.Getenv("ENV"),
			TelegramToken:      os.Getenv("TELEGRAM_TOKEN"),
			TelegramChatId:     os.Getenv("TELEGRAM_CHAT_ID"),
			ImageSearchHost:    os.Getenv("IMAGE_SEARCH_HOST"),
			ImageSearchPort:    os.Getenv("IMAGE_SEARCH_PORT"),
			ImageSearchTimeout: os.Getenv("IMAGE_SEARCH_TIMEOUT"),
		},
	}
}
