package server

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

func profilePage(c *gin.Context) {
	c.Header("Content-Type", "text/html; charset=utf-8")
	c.HTML(http.StatusOK, "profile.html", nil)
}
