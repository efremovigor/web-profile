package server

import (
	"context"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"html/template"
	"net/http"
	"web_profile/pkg/log"
)

type ServerConfig interface {
	GetHttpsSocket() string
	GetHttpSocket() string
	GetDomain() string
	GetHttpsUrl() string
	IsDev() bool
}

func NewServer(logger *log.Logger, template *template.Template, config ServerConfig) Server {
	return Server{
		Logger:   logger,
		Template: template,
		Config:   config,
	}
}

type Server struct {
	Logger      *log.Logger
	Template    *template.Template
	Config      ServerConfig
	SetupRoutes func(r *gin.Engine) *gin.Engine
}

func (s *Server) Run(ctx context.Context) error {
	g, ctx := errgroup.WithContext(ctx)
	gin.DefaultWriter = s.Logger

	s.runRedirectServer(g, ctx)
	s.runWebServer(g, ctx)

	err := g.Wait()
	if err != nil {
		s.Logger.Log(log.Error(fmt.Sprintf("error waiting https server: %s", err), nil))
	}

	return err
}

func (s *Server) runRedirectServer(g *errgroup.Group, ctx context.Context) {
	httpServer := &http.Server{Addr: s.Config.GetHttpSocket(), Handler: redirectHandler(s.Config)}
	g.Go(s.runServer(httpServer, false))
	g.Go(s.shutdownServer(ctx, httpServer))
}

func (s *Server) runWebServer(g *errgroup.Group, ctx context.Context) {
	router := gin.New()
	router.Use(gin.LoggerWithWriter(s.Logger), CustomRecovery(s.Logger))

	httpsServer := &http.Server{
		Addr:    s.Config.GetHttpsSocket(),
		Handler: s.SetupRoutes(router),
	}
	if s.Config.IsDev() {
		httpsServer.Addr = s.Config.GetHttpsSocket()
	}
	g.Go(s.runServer(httpsServer, true))
	g.Go(s.shutdownServer(ctx, httpsServer))
}

func (s *Server) runServer(server *http.Server, isHttps bool) func() error {
	if isHttps && !s.Config.IsDev() {
		return func() error {
			s.Logger.Log(log.Info(fmt.Sprintf("listening on https://%s", s.Config.GetDomain()), nil))
			m := &autocert.Manager{
				Cache:      autocert.DirCache("certs"),
				Prompt:     autocert.AcceptTOS,
				HostPolicy: autocert.HostWhitelist(s.Config.GetDomain()),
			}
			if err := server.Serve(m.Listener()); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.Logger.Log(log.Exception(fmt.Sprintf("error listening and serving: %s\n", err), nil))
				return err
			}
			return nil
		}
	} else {
		return func() error {
			s.Logger.Log(log.Info(fmt.Sprintf("listening http://%s", server.Addr), nil))
			if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.Logger.Log(log.Exception(fmt.Sprintf("%s - error working server: %s", server.Addr, err), nil))
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
			s.Logger.Log(log.Exception(fmt.Sprintf("https server shutdown error: %s", err), nil))
			return err
		}
		return nil
	}
}

func redirectHandler(config ServerConfig) http.Handler {
	return http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, config.GetHttpsUrl()+r.RequestURI, http.StatusMovedPermanently)
		},
	)
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
