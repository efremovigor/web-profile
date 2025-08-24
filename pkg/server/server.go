package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"html/template"
	"net/http"
	"web_profile/internal"
	"web_profile/pkg/consts/web"
	"web_profile/pkg/log"
)

func NewServer(logger *log.Logger, config *internal.Config) Server {

	return Server{
		logger: logger,
		config: config,
	}
}

type Server struct {
	logger *log.Logger
	config *internal.Config
}

func (s *Server) Run(ctx context.Context) error {

	g, ctx := errgroup.WithContext(ctx)
	gin.DefaultWriter = s.logger

	g.Go(func() error {
		defer ctx.Done()
		s.logger.Log(log.Info(fmt.Sprintf("listening http://%s", s.config.GetHttpSocket()), nil))
		httpServer := &http.Server{Addr: s.config.GetHttpSocket(), Handler: RedirectHandler(s.config)}
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			s.logger.Log(log.Exception(fmt.Sprintf("error working https server: %s", err), nil))
		}
		return nil
	})

	// Создаём общий роутер для приложения
	router := gin.New()
	router.Use(gin.LoggerWithWriter(s.logger), CustomRecovery(s.logger))

	s.setupRoutes(router)
	// HTTPS-сервер
	httpsServer := &http.Server{
		Addr:    s.config.GetHttpsSocket(), // например ":443"
		Handler: router,
	}
	if s.config.IsDev() {
		httpsServer.Addr = s.config.GetHttpsSocket()
	}
	g.Go(s.runServer(httpsServer, true))
	g.Go(s.shutdownServer(ctx, httpsServer))

	err := g.Wait()
	if err != nil {
		s.logger.Log(log.Error(fmt.Sprintf("error waiting https server: %s", err), nil))
	}

	return err
}

func (s *Server) runServer(server *http.Server, isHttps bool) func() error {
	if isHttps && !s.config.IsDev() {
		return func() error {
			s.logger.Log(log.Info(fmt.Sprintf("listening on %s://%s\n", web.HTTPSProtocol, s.config.Env.Domain), nil))
			m := &autocert.Manager{
				Cache:      autocert.DirCache("certs"),
				Prompt:     autocert.AcceptTOS,
				HostPolicy: autocert.HostWhitelist(s.config.Env.Domain),
			}
			server.TLSConfig = &tls.Config{GetCertificate: m.GetCertificate}

			if err := server.Serve(m.Listener()); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Log(log.Exception(fmt.Sprintf("error listening and serving: %s\n", err), nil))
				return err
			}
			return nil
		}
	} else {
		return func() error {
			s.logger.Log(log.Info(fmt.Sprintf("listening %s", server.Addr), nil))
			if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Log(log.Exception(fmt.Sprintf("error working https server: %s", err), nil))
				return err
			}
			return nil
		}
	}
}

func (s *Server) shutdownServer(ctx context.Context, httpsServer *http.Server) func() error {
	return func() error {
		<-ctx.Done()
		if err := httpsServer.Shutdown(ctx); err != nil {
			s.logger.Log(log.Exception(fmt.Sprintf("https server shutdown error: %s", err), nil))
			return err
		}
		return nil
	}
}

func RedirectHandler(config *internal.Config) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, config.GetHttpsUrl()+r.RequestURI, http.StatusMovedPermanently)
		},
	)
}

func (s *Server) setupRoutes(router *gin.Engine) {
	// Включаем middleware
	router.Use(gin.LoggerWithWriter(s.logger), CustomRecovery(s.logger))
	// Пример маршрута
	router.GET("/", ProfileHandler(s.config, s.logger))
	router.Static("/static", "./static") // статика, если понадобится

}

func CustomRecovery(l *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if r := recover(); r != nil {
				var msg string
				switch e := r.(type) {
				case error:
					l.Log(log.Error("unexpectable error: "+e.Error(), c.Request))
					msg = "internal server error"
				default:
					l.Log(log.Error(fmt.Sprintf("critical error: %v", r), c.Request))
					msg = "internal server error"
				}
				if !c.IsAborted() {
					c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": msg})
				}
				c.Abort()
			}
		}()

		c.Next()
	}
}

func ProfileHandler(_ *internal.Config, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		tmpl := template.Must(template.ParseFiles("static/index.html"))

		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.Execute(c.Writer, nil); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error rendering template: %s", err), nil))
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
}
