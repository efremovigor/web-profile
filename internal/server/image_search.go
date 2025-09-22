package server

import (
	"context"
	"fmt"
	"github.com/gin-gonic/gin"
	"io"
	"net/http"
	"time"
	pb "web_profile/internal/generated/image_search"
	"web_profile/pkg/client"
)

func ImageSearchAboutPage() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.HTML(http.StatusOK, "image_search_about.html", nil)
	}
}

func ImageSearchPage() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Content-Type", "text/html; charset=utf-8")
		c.HTML(http.StatusOK, "image_search.html", nil)
	}
}

type ImageSearchClientConfig interface {
	GetImageSearchHost() string
	GetImageSearchPort() int
	GetImageSearchTimeout() time.Duration
}

// ImageSearchHandler обработчик поиска по изображению через gRPC
func (s *Server) ImageSearchHandler(config ImageSearchClientConfig) gin.HandlerFunc {
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
		grpcClient, closeFn, err := client.CreateGRPCClient(client.GRPCClientConfig{Host: config.GetImageSearchHost(), Port: config.GetImageSearchPort(), Timeout: config.GetImageSearchTimeout()})
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
		ctx, cancel := context.WithTimeout(context.Background(), config.GetImageSearchTimeout())
		defer cancel()

		request := &pb.SearchRequest{
			ImageData: imageData,
			TopK:      9,
			UseCache:  false,
		}

		response, err := grpcClient.SearchImage(ctx, request)
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
			jsonResponse.ProcessingTime.TotalSeconds = proxyTime
		}

		if jsonResponse.Success {
			c.JSON(http.StatusOK, jsonResponse)
		} else {
			c.JSON(http.StatusInternalServerError, jsonResponse)
		}
	}
}

// HealthCheckHandler проверка здоровья через gRPC
func (s *Server) HealthCheckHandler(config ImageSearchClientConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		grpcClient, closeFn, err := client.CreateGRPCClient(client.GRPCClientConfig{Host: config.GetImageSearchHost(), Port: config.GetImageSearchPort(), Timeout: config.GetImageSearchTimeout()})
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    err.Error(),
				"endpoint": fmt.Sprintf("%s:%d", config.GetImageSearchHost(), config.GetImageSearchPort()),
			})
			return
		}
		defer closeFn()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		response, err := grpcClient.HealthCheck(ctx, &pb.HealthCheckRequest{})
		if err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    err.Error(),
				"endpoint": fmt.Sprintf("%s:%d", config.GetImageSearchHost(), config.GetImageSearchPort()),
			})
			return
		}

		if !response.Healthy {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"status":   "unhealthy",
				"service":  "python_search",
				"error":    response.Status,
				"endpoint": fmt.Sprintf("%s:%d", config.GetImageSearchHost(), config.GetImageSearchPort()),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"status":   "healthy",
			"service":  "python_search",
			"endpoint": fmt.Sprintf("%s:%d", config.GetImageSearchHost(), config.GetImageSearchPort()),
			"details":  response.Status,
		})
	}
}

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
