package env

import (
	"github.com/joho/godotenv"
	"log"
	"os"
)

const envPath = "./.env"

type Config struct {
	Env    string
	Domain string
}

func (e Config) IsDev() bool {
	return e.Env != "prod"
}

func (e Config) GetHttpsUrl() string {
	if e.IsDev() {
		return "http://" + e.GetHttpsSocket()
	}
	return "https://" + e.GetHttpsSocket()
}

func (e Config) GetHttpSocket() string {
	if e.IsDev() {
		return e.Domain + ":8080"
	}
	return e.Domain + ":80"
}

func (e Config) GetHttpsSocket() string {
	if e.IsDev() {
		return e.Domain + ":8081"
	}
	return e.Domain + ":443"
}

func LoadConfig() Config {
	err := godotenv.Load(envPath)

	if err != nil {
		log.Fatalf("Error loading .env file")
	}

	return Config{
		Domain: os.Getenv("DOMAIN"),
		Env:    os.Getenv("ENV"),
	}
}
