package server

import (
	"context"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"web_profile/internal"
	"web_profile/pkg/log"
)

func NewServer(logger *log.Logger, config *internal.Config) Server {

	return Server{
		logger: logger,
		config: config,
	}
}

type Server struct {
	logger   *log.Logger
	config   *internal.Config
	template *template.Template
}

func (s *Server) SetTemplate(template *template.Template) {
	s.template = template
}

func (s *Server) Run(ctx context.Context) error {

	g, ctx := errgroup.WithContext(ctx)
	gin.DefaultWriter = s.logger

	s.runRedirectServer(g, ctx)
	s.runWebServer(g, ctx)

	err := g.Wait()
	if err != nil {
		s.logger.Log(log.Error(fmt.Sprintf("error waiting https server: %s", err), nil))
	}

	return err
}

func (s *Server) runRedirectServer(g *errgroup.Group, ctx context.Context) {
	httpServer := &http.Server{Addr: s.config.GetHttpSocket(), Handler: RedirectHandler(s.config)}
	g.Go(s.runServer(httpServer, false))
	g.Go(s.shutdownServer(ctx, httpServer))
}

func (s *Server) runWebServer(g *errgroup.Group, ctx context.Context) {
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
}

func (s *Server) runServer(server *http.Server, isHttps bool) func() error {

	if isHttps && !s.config.IsDev() {
		return func() error {
			s.logger.Log(log.Info(fmt.Sprintf("listening on https://%s", s.config.Env.Domain), nil))
			m := &autocert.Manager{
				Cache:      autocert.DirCache("certs"),
				Prompt:     autocert.AcceptTOS,
				HostPolicy: autocert.HostWhitelist(s.config.Env.Domain),
			}
			if err := server.Serve(m.Listener()); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Log(log.Exception(fmt.Sprintf("error listening and serving: %s\n", err), nil))
				return err
			}
			return nil
		}
	} else {
		return func() error {
			s.logger.Log(log.Info(fmt.Sprintf("listening https://%s", server.Addr), nil))
			if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Log(log.Exception(fmt.Sprintf("%s - error working server: %s", server.Addr, err), nil))
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
	// Пример маршрута
	router.GET("/", ProfileHandler(s.template, s.logger))
	router.POST("/send", GetInTouchHandler(s.config, s.logger))
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

func ProfileHandler(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
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
func GetInTouchHandler(config *internal.Config, logger *log.Logger) gin.HandlerFunc {
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

func sendToTelegram(config *internal.Config, name, email, subject, msg string) error {
	apiURL := fmt.Sprintf("https://api.telegram.org/bot%s/sendMessage", config.Env.TelegramToken)

	text := fmt.Sprintf(
		"📩 <b>на %s кто-то пишет</b>\n\n"+
			"👤 <b>Имя:</b> %s\n"+
			"📧 <b>Email:</b> %s\n"+
			"📝 <b>Тема:</b> %s\n"+
			"💬 <b>Сообщение:</b>\n%s",
		config.Env.Domain, name, email, subject, msg,
	)

	resp, err := http.PostForm(apiURL, url.Values{
		"chat_id":    {config.Env.TelegramChatId}, // тут число или @username канала
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
