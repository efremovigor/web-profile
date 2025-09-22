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
	allFiles := []string{
		"static/template/profile.html",
		"static/template/404.html",
		"static/template/demo/image_search/image_search.html",
		"static/template/demo/image_search/image_search_about.html",
	}

	r.LoadHTMLFiles(allFiles...)

	r.GET("/profile", ProfilePage())
	r.POST("/send", GetInTouchHandler(s.TelegramConfig, s.Logger))
	r.Static("/static", "./static")

	r.NoRoute(Page404())

	r.GET("/image-search-about", ImageSearchAboutPage())
	r.GET("/image-search", ImageSearchPage())
	r.POST("/image-search", s.ImageSearchHandler(s.ImageSearchConfig))
	r.GET("/health/image-search", s.HealthCheckHandler(s.ImageSearchConfig))

	return r
}
