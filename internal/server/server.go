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
		"static/template/demo/face_control/face_control.html",
	}

	r.LoadHTMLFiles(allFiles...)

	r.GET("/profile", profilePage)
	r.POST("/send", GetInTouchHandler(s.TelegramConfig, s.Logger))
	r.Static("/static", "./static")

	r.NoRoute(page404)

	r.GET("/image-search-about", imageSearchAboutPage)
	r.GET("/image-search", imageSearchPage)
	r.POST("/image-search", s.ImageSearchHandler(s.ImageSearchConfig))
	r.GET("/image-search/health", s.HealthCheckHandler(s.ImageSearchConfig))

	r.GET("/face-control", faceControlPage)
	r.GET("/face-control/health", faceControlHealthCheck)
	r.POST("/face-control/registration/start", startRegistration)
	r.POST("/face-control/registration/process", processRegistrationFrame)
	r.POST("/face-control/registration/cancel", cancelRegistration)

	r.POST("/face-control/auth", authenticate)
	r.POST("/face-control/auth/start", startLiveness)
	r.POST("/face-control/auth/update", updateLiveness)

	return r
}
