package server

import (
	"github.com/gin-gonic/gin"
	"web_profile/pkg/server"
)

type Server struct {
	server.Server
	TelegramConfig    TelegramConfig
	ImageSearchConfig ImageSearchClientConfig
}

func (s Server) Routing(r *gin.Engine) *gin.Engine {
	r.GET("/profile", ProfilePage(s.Template, s.Logger))
	r.GET("/image-search-about", ImageSearchAboutPage(s.Template, s.Logger))
	r.GET("/image-search", ImageSearchPage(s.Template, s.Logger))
	r.POST("/send", GetInTouchHandler(s.TelegramConfig, s.Logger))
	r.Static("/static", "./static")

	r.NoRoute(Page404(s.Template, s.Logger))

	r.POST("/image-search", s.ImageSearchHandler(s.ImageSearchConfig))
	r.GET("/health/image-search", s.HealthCheckHandler(s.ImageSearchConfig))

	return r
}
