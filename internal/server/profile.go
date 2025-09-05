package server

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"html/template"
	"net/http"
	"web_profile/pkg/log"
)

func ProfilePage(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.ExecuteTemplate(c.Writer, "profile.html", nil); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error rendering Template: %s", err), nil))
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
}
