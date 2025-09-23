package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"io"
	"net/http"
	"time"
)

func faceControlPage(c *gin.Context) {
	c.Header("Content-Type", "text/html; charset=utf-8")
	c.HTML(http.StatusOK, "face_control.html", nil)
}

func faceControlHealthCheck(c *gin.Context) {
	pythonHealth, err := checkPythonHealth()
	if err != nil {
		c.JSON(http.StatusInternalServerError, HealthResponse{
			Status:   "error",
			Analyzer: err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, HealthResponse{
		Status:   "ok",
		Analyzer: pythonHealth,
	})
}

func startRegistration(c *gin.Context) {
	var req RegistrationRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	if req.FirstName == "" || req.LastName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "First name and last name are required"})
		return
	}

	response, err := analize("/api/start_registration", req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if errorMsg, exists := response["error"].(string); exists && errorMsg != "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": errorMsg})
		return
	}

	c.JSON(http.StatusOK, response)
}

func cancelRegistration(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	if req.SessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	response, err := analize("/api/cancel_registration", req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func processRegistrationFrame(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
		Frame     string `json:"frame"`
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	if req.SessionID == "" || req.Frame == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID and frame data are required"})
		return
	}

	// Логируем размер frame данных для отладки
	frameSize := len(req.Frame)
	fmt.Printf("Processing registration frame for session %s, frame size: %d bytes\n", req.SessionID, frameSize)

	pythonRequest := map[string]interface{}{
		"session_id": req.SessionID,
		"frame":      req.Frame,
	}

	response, err := analize("/api/process_registration", pythonRequest)
	if err != nil {
		fmt.Printf("Error calling Python for registration: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	fmt.Printf("Python registration response: %+v\n", response)
	c.JSON(http.StatusOK, response)
}

func checkPythonHealth() (string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://localhost:5001" + "/health")
	if err != nil {
		return "unhealthy", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "unhealthy", fmt.Errorf("python server returned status: %d", resp.StatusCode)
	}

	return "healthy", nil
}

func analize(endpoint string, requestData interface{}) (map[string]interface{}, error) {
	var response map[string]interface{}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Создаем HTTP клиент с таймаутом
	client := &http.Client{Timeout: 30 * time.Second}

	resp, err := client.Post("http://localhost:5001"+endpoint, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python endpoint %s: %v", endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("python server returned status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v. Body: %s", err, string(body[:200]))
	}

	return response, nil
}

func startLiveness(c *gin.Context) {
	response, err := analize("/start_liveness", nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func updateLiveness(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
		Frame     string `json:"frame"`
		Image     string `json:"image"` // Добавляем поддержку обоих параметров
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	// Используем frame или image
	imageData := req.Image
	if imageData == "" {
		imageData = req.Frame
	}

	if req.SessionID == "" || imageData == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID and image data are required"})
		return
	}

	pythonRequest := map[string]interface{}{
		"session_id": req.SessionID,
		"image":      imageData,
	}

	fmt.Printf("Update liveness - Session: %s, Data size: %d bytes\n", req.SessionID, len(imageData))

	response, err := analize("/update_liveness", pythonRequest)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func authenticate(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
		Frame     string `json:"frame"`
		Image     string `json:"image"` // Добавляем поддержку обоих параметров
	}
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	// Используем frame или image
	imageData := req.Image
	if imageData == "" {
		imageData = req.Frame
	}

	if req.SessionID == "" || imageData == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID and image data are required"})
		return
	}

	pythonRequest := map[string]interface{}{
		"session_id": req.SessionID,
		"image":      imageData,
	}

	fmt.Printf("Authenticate - Session: %s, Data size: %d bytes\n", req.SessionID, len(imageData))

	response, err := analize("/authenticate", pythonRequest)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

type HealthResponse struct {
	Status   string `json:"status"`
	Analyzer string `json:"analyzer"`
}

type RegistrationRequest struct {
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
}

type FrameRequest struct {
	SessionID string `json:"session_id"`
	Frame     string `json:"frame"`
}
