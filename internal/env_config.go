package internal

import (
	"github.com/joho/godotenv"
	"log"
	"os"
	"strconv"
	"web_profile/pkg/consts/web"
)

const envPath = "./config/.env"

type Config struct {
	Env Env
}

type Env struct {
	Type   string
	Domain string
}

func (e Config) IsDev() bool {
	return e.Env.Type != "prod"
}

func (e Config) GetHttpsUrl() string {
	if e.IsDev() {
		return web.HTTPProtocol + "://" + e.GetHttpsSocket()
	}
	return web.HTTPSProtocol + "://" + e.GetHttpsSocket()
}

func (e Config) GetHttpSocket() string {
	if e.IsDev() {
		return "127.0.0.1:" + strconv.Itoa(web.DevHTTPPort)
	}
	return ":" + strconv.Itoa(web.HTTPPort)
}

func (e Config) GetHttpsSocket() string {
	if e.IsDev() {
		return "127.0.0.1:" + strconv.Itoa(web.DevHTTPSPort)
	}
	return e.Env.Domain + ":" + strconv.Itoa(web.HTTPSPort)
}

func NewConfig() Config {
	err := godotenv.Load(envPath)

	if err != nil {
		log.Fatalf("Error loading .env_config file")
	}

	return Config{
		Env: Env{
			Domain: os.Getenv("DOMAIN"),
			Type:   os.Getenv("ENV"),
		},
	}
}
