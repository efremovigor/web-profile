package server

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func Page404() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.HTML(http.StatusNotFound, "404.html", nil)
	}
}
