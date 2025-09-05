package server

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"io"
	"net/http"
	"net/url"
	"web_profile/pkg/log"
)

type TelegramConfig interface {
	GetTelegramChatId() string
	GetTelegramToken() string
}

func GetInTouchHandler(config TelegramConfig, logger *log.Logger) gin.HandlerFunc {
	type Request struct {
		Name    string `json:"name"`
		Email   string `json:"email"`
		Subject string `json:"subject"`
		Msg     string `json:"msg"`
	}

	return func(c *gin.Context) {
		var req Request
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Неверный формат JSON"})
			return
		}

		if req.Name == "" || req.Email == "" || req.Subject == "" || req.Msg == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Все поля обязательны"})
			return
		}

		if err := sendToTelegram(config, req.Name, req.Email, req.Subject, req.Msg); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error of sending message to telegram: %s", err), nil))
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"status": "Сообщение отправлено!"})
	}
}

func sendToTelegram(config TelegramConfig, name, email, subject, msg string) error {
	apiURL := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", config.GetTelegramToken())

	text := fmt.Sprintf(
		"📩 <b>на GetInTouch кто-то пишет</b>\n\n"+
			"👤 <b>Имя:</b> %s\n"+
			"📧 <b>Email:</b> %s\n"+
			"📝 <b>Тема:</b> %s\n"+
			"💬 <b>Сообщение:</b>\n%s",
		name, email, subject, msg,
	)

	resp, err := http.PostForm(apiURL, url.Values{
		"chat_id":    {config.GetTelegramChatId()},
		"text":       {text},
		"parse_mode": {"HTML"},
	})

	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return fmt.Errorf("telegram API error: %s\nresponse: %s", resp.Status, string(body))
	}

	return nil
}
