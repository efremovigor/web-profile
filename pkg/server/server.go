package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
	pb "web_profile"

	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"html/template"
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
	router := gin.New()
	router.Use(gin.LoggerWithWriter(s.logger))

	s.setupRoutes(router)

	httpsServer := &http.Server{
		Addr:    s.config.GetHttpsSocket(),
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

// GRPCClientConfig конфигурация gRPC клиента
type GRPCClientConfig struct {
	Host    string
	Port    int
	Timeout time.Duration
}

// DefaultGRPCConfig дефолтная конфигурация gRPC
var DefaultGRPCConfig = GRPCClientConfig{
	Host:    "localhost",
	Port:    9999,
	Timeout: 30 * time.Second,
}

// createGRPCClient создает gRPC клиент
func createGRPCClient(config GRPCClientConfig) (pb.ImageSearchServiceClient, func(), error) {
	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx,
		fmt.Sprintf("%s:%d", config.Host, config.Port),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC connection failed: %v", err)
	}

	client := pb.NewImageSearchServiceClient(conn)
	return client, func() { conn.Close() }, nil
}

func (s *Server) setupRoutes(router *gin.Engine) {
	router.GET("/profile", ProfilePage(s.template, s.logger))
	router.GET("/image-search-about", ImageSearchAboutPage(s.template, s.logger))
	router.GET("/image-search", ImageSearchPage(s.template, s.logger))
	router.POST("/send", GetInTouchHandler(s.config, s.logger))
	router.Static("/static", "./static") // статика, если понадобится

	router.NoRoute(Page404(s.template, s.logger))

	grpcConfig := GRPCClientConfig{
		Host:    "localhost",
		Port:    9999,
		Timeout: 30 * time.Second,
	}

	router.POST("/image-search", s.ImageSearchHandler(grpcConfig))
	router.GET("/health/image-search", s.HealthCheckHandler(grpcConfig))
}

// ImageSearchRequest запрос от пользователя
type ImageSearchRequest struct {
	TopK    int  `form:"top_k" json:"top_k" binding:"min=1,max=100"`
	NoCache bool `form:"no_cache" json:"no_cache"`
}

// ImageSearchHandler обработчик поиска по изображению через gRPC
func (s *Server) ImageSearchHandler(config GRPCClientConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		startTime := time.Now()

		// Получение файла из запроса
		file, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Image file is required",
			})
			return
		}

		// Проверка размера файла
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

		// Создание gRPC клиента
		client, closeFn, err := createGRPCClient(config)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success":     false,
				"error":       fmt.Sprintf("gRPC connection failed: %v", err),
				"proxy_error": true,
			})
			return
		}
		defer closeFn()

		// Вызов gRPC метода
		ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
		defer cancel()

		request := &pb.SearchRequest{
			ImageData: imageData,
			TopK:      9,
			UseCache:  false,
		}

		response, err := client.SearchImage(ctx, request)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"success":     false,
				"error":       fmt.Sprintf("Search failed: %v", err),
				"proxy_error": true,
			})
			return
		}

		// Конвертация gRPC response в JSON response
		jsonResponse := convertToJSONResponse(response)

		// Добавляем время проксирования
		proxyTime := time.Since(startTime).Seconds()
		if jsonResponse.ProcessingTime.TotalSeconds > 0 {
			jsonResponse.ProcessingTime.TotalSeconds += proxyTime
		}

		if jsonResponse.Success {
			c.JSON(http.StatusOK, jsonResponse)
		} else {
			c.JSON(http.StatusInternalServerError, jsonResponse)
		}
	}
}

// HealthCheckHandler проверка здоровья через gRPC
func (s *Server) HealthCheckHandler(config GRPCClientConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		client, closeFn, err := createGRPCClient(config)
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    err.Error(),
				"endpoint": fmt.Sprintf("%s:%d", config.Host, config.Port),
			})
			return
		}
		defer closeFn()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		response, err := client.HealthCheck(ctx, &pb.HealthCheckRequest{})
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    err.Error(),
				"endpoint": fmt.Sprintf("%s:%d", config.Host, config.Port),
			})
			return
		}

		if !response.Healthy {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    response.Status,
				"endpoint": fmt.Sprintf("%s:%d", config.Host, config.Port),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":   "healthy",
			"service":  "python_search",
			"endpoint": fmt.Sprintf("%s:%d", config.Host, config.Port),
			"details":  response.Status,
		})
	}
}

// convertToJSONResponse конвертирует gRPC response в JSON структуру
func convertToJSONResponse(response *pb.SearchResponse) *SearchResponse {
	jsonResponse := &SearchResponse{
		Success: response.Success,
		Error:   response.Error,
		Device:  response.Device,
		Cached:  response.Cached,
	}

	if response.ProcessingTime != nil {
		jsonResponse.ProcessingTime.EmbeddingSeconds = response.ProcessingTime.EmbeddingSeconds
		jsonResponse.ProcessingTime.SearchSeconds = response.ProcessingTime.SearchSeconds
		jsonResponse.ProcessingTime.TotalSeconds = response.ProcessingTime.TotalSeconds
	}

	for _, result := range response.Results {
		jsonResult := SearchResult{
			ProductID:   result.ProductId,
			CategoryID:  result.CategoryId,
			Similarity:  result.Similarity,
			Distance:    result.Distance,
			ProductURL:  result.ProductUrl,
			CategoryURL: result.CategoryUrl,
			ImageURL:    result.ImageUrl,
		}
		jsonResponse.Results = append(jsonResponse.Results, jsonResult)
	}

	return jsonResponse
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

// SearchResponse структура ответа
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
