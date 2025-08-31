package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"html/template"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"
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
			s.logger.Log(log.Info(fmt.Sprintf("listening http://%s", server.Addr), nil))
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
	router.GET("/profile", ProfilePage(s.template, s.logger))
	router.GET("/image-search-about", ImageSearchAboutPage(s.template, s.logger))
	router.GET("/image-search", ImageSearchPage(s.template, s.logger))
	router.POST("/send", GetInTouchHandler(s.config, s.logger))
	router.Static("/static", "./static") // статика, если понадобится
	// Настройка Python сервера
	pythonConfig := PythonServerConfig{
		Host:    "localhost", // или IP вашего Python сервера
		Port:    9999,
		Timeout: 30 * time.Second,
	}

	// Добавление роутов
	router.POST("/image-search", ImageSearchHandler(pythonConfig))
	router.GET("/health/image-search", HealthCheckHandler(pythonConfig))

	router.NoRoute(Page404(s.template, s.logger))
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

func Page404(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.ExecuteTemplate(c.Writer, "404.html", nil); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error rendering template: %s", err), nil))
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
}

func ProfilePage(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.ExecuteTemplate(c.Writer, "profile.html", nil); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error rendering template: %s", err), nil))
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
}

func ImageSearchAboutPage(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.ExecuteTemplate(c.Writer, "image_search_about.html", nil); err != nil {
			logger.Log(log.Exception(fmt.Sprintf("error rendering template: %s", err), nil))
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
	}
}

func ImageSearchPage(tmpl *template.Template, logger *log.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Устанавливаем заголовок Content-Type
		c.Header("Content-Type", "text/html; charset=utf-8")

		// Выполняем шаблон и записываем результат в ResponseWriter
		if err := tmpl.ExecuteTemplate(c.Writer, "image_search.html", nil); err != nil {
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

// SearchRequest структура запроса к Python серверу
type SearchRequest struct {
	TopK      int  `json:"top_k"`
	ImageSize int  `json:"image_size"`
	UseCache  bool `json:"use_cache"`
}

// SearchResult структура результата
type SearchResult struct {
	ProductID   string  `json:"product_id"`
	CategoryID  string  `json:"category_id"`
	Similarity  float64 `json:"similarity"`
	Distance    float64 `json:"distance"`
	ProductURL  string  `json:"product_url"`
	CategoryURL string  `json:"category_url"`
	ImageURL    string  `json:"image_url"`
}

// SearchResponse структура ответа от Python сервера
type SearchResponse struct {
	Success        bool           `json:"success"`
	Results        []SearchResult `json:"results,omitempty"`
	Error          string         `json:"error,omitempty"`
	ProcessingTime struct {
		EmbeddingSeconds float64 `json:"embedding_seconds"`
		SearchSeconds    float64 `json:"search_seconds"`
		TotalSeconds     float64 `json:"total_seconds"`
	} `json:"processing_time,omitempty"`
	Device string `json:"device,omitempty"`
	Cached bool   `json:"cached,omitempty"`
}

// Config конфигурация Python сервера
type PythonServerConfig struct {
	Host    string
	Port    int
	Timeout time.Duration
}

// DefaultConfig дефолтная конфигурация
var DefaultConfig = PythonServerConfig{
	Host:    "localhost",
	Port:    9999,
	Timeout: 30 * time.Second,
}

// searchImage отправляет изображение на Python сервер и возвращает результат
func searchImage(imageData []byte, topK int, useCache bool, config PythonServerConfig) (*SearchResponse, error) {
	serverAddr := fmt.Sprintf("%s:%d", config.Host, config.Port)

	// Установка соединения
	conn, err := net.DialTimeout("tcp", serverAddr, config.Timeout)
	if err != nil {
		return nil, fmt.Errorf("connection to python server failed: %v", err)
	}
	defer conn.Close()

	// Установка таймаута на операции
	conn.SetDeadline(time.Now().Add(config.Timeout))

	// Подготовка метаданных
	request := SearchRequest{
		TopK:      topK,
		ImageSize: len(imageData),
		UseCache:  useCache,
	}

	// Сериализация метаданных
	metadataJSON, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("JSON marshaling failed: %v", err)
	}

	// Отправка заголовка (размер метаданных)
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(metadataJSON)))
	_, err = conn.Write(header)
	if err != nil {
		return nil, fmt.Errorf("header send failed: %v", err)
	}

	// Отправка метаданных
	_, err = conn.Write(metadataJSON)
	if err != nil {
		return nil, fmt.Errorf("metadata send failed: %v", err)
	}

	// Отправка данных изображения
	_, err = conn.Write(imageData)
	if err != nil {
		return nil, fmt.Errorf("image data send failed: %v", err)
	}

	// Получение ответа: заголовок
	responseHeader := make([]byte, 4)
	_, err = io.ReadFull(conn, responseHeader)
	if err != nil {
		return nil, fmt.Errorf("response header read failed: %v", err)
	}

	responseSize := binary.BigEndian.Uint32(responseHeader)

	// Получение данных ответа
	responseData := make([]byte, responseSize)
	_, err = io.ReadFull(conn, responseData)
	if err != nil {
		return nil, fmt.Errorf("response data read failed: %v", err)
	}

	// Парсинг JSON ответа
	var response SearchResponse
	err = json.Unmarshal(responseData, &response)
	if err != nil {
		return nil, fmt.Errorf("JSON unmarshaling failed: %v", err)
	}

	return &response, nil
}

// ImageSearchRequest запрос от пользователя
type ImageSearchRequest struct {
	TopK    int  `form:"top_k" json:"top_k" binding:"min=1,max=100"`
	NoCache bool `form:"no_cache" json:"no_cache"`
}

// ImageSearchHandler обработчик поиска по изображению
func ImageSearchHandler(config PythonServerConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()

		// Парсинг параметров запроса
		//var requestParams ImageSearchRequest
		//if err := c.ShouldBindQuery(&requestParams); err != nil {
		//	c.JSON(http.StatusBadRequest, gin.H{
		//		"success": false,
		//		"error":   fmt.Sprintf("Invalid query parameters: %v", err),
		//	})
		//	return
		//}

		// Получение файла из запроса
		file, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Image file is required",
			})
			return
		}

		// Проверка размера файла (максимум 10MB)
		if file.Size > 10*1024*1024 {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Image size exceeds 10MB limit",
			})
			return
		}

		// Проверка типа файла
		allowedTypes := map[string]bool{
			"image/jpeg": true,
			"image/jpg":  true,
			"image/png":  true,
		}
		if !allowedTypes[file.Header.Get("Content-Type")] {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Unsupported image format. Supported: JPEG, PNG",
			})
			return
		}

		// Чтение файла
		openedFile, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"error":   fmt.Sprintf("Failed to open file: %v", err),
			})
			return
		}
		defer openedFile.Close()

		imageData, err := io.ReadAll(openedFile)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"error":   fmt.Sprintf("Failed to read file: %v", err),
			})
			return
		}

		// Вызов Python сервера
		response, err := searchImage(
			imageData,
			9,
			false,
			config,
		)

		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success":     false,
				"error":       fmt.Sprintf("Search failed: %v", err),
				"proxy_error": true,
			})
			return
		}

		// Добавляем время проксирования
		proxyTime := time.Since(startTime).Seconds()
		if response.ProcessingTime.TotalSeconds > 0 {
			response.ProcessingTime.TotalSeconds += proxyTime
		}

		// Возвращаем ответ от Python сервера
		if response.Success {
			c.JSON(http.StatusOK, response)
		} else {
			c.JSON(http.StatusInternalServerError, response)
		}
	}
}

// HealthCheckHandler проверка здоровья Python сервера
func HealthCheckHandler(config PythonServerConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		serverAddr := fmt.Sprintf("%s:%d", config.Host, config.Port)

		conn, err := net.DialTimeout("tcp", serverAddr, 5*time.Second)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    err.Error(),
				"endpoint": serverAddr,
			})
			return
		}
		defer conn.Close()

		c.JSON(http.StatusOK, gin.H{
			"status":   "healthy",
			"service":  "python_search",
			"endpoint": serverAddr,
		})
	}
}
