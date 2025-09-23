class Utils {
    static baseURL = 'http://localhost:8080';
    static debugMode = true;

    // Логирование с временной меткой
    static log(message, type = 'info') {
        if (!this.debugMode) return;

        const timestamp = new Date().toLocaleTimeString();
        const styles = {
            info: 'color: blue;',
            error: 'color: red;',
            success: 'color: green;',
            warn: 'color: orange;'
        };

        console.log(`%c[${timestamp}] ${message}`, styles[type] || 'color: black;');
    }

    // Показать статус сообщение
    static showStatus(element, message, type = 'info') {
        const statusElement = document.getElementById(element);
        if (!statusElement) return;

        const instructionElement = statusElement.querySelector('.instructions');
        if (instructionElement) {
            instructionElement.textContent = message;
            instructionElement.className = 'instructions';

            if (type === 'error') {
                instructionElement.classList.add('error');
            } else if (type === 'success') {
                instructionElement.classList.add('success');
            }
        }
    }

    // Показать отладочную информацию
    static showDebugInfo(elementId, message) {
        if (!this.debugMode) return;

        const debugElement = document.getElementById(elementId);
        if (debugElement) {
            debugElement.textContent = `[DEBUG] ${new Date().toLocaleTimeString()}: ${message}`;
            debugElement.style.display = 'block';
        }
    }

    // Проверка здоровья системы
    static async checkSystemHealth() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`);
            const data = await response.json();
            this.log(`System health: ${data.status}`, data.status === 'ok' ? 'success' : 'error');
            return data;
        } catch (error) {
            this.log(`Health check failed: ${error.message}`, 'error');
            return { status: 'error', message: 'System unavailable' };
        }
    }

    // Управление камерой
    static async startCamera(videoElementId) {
        const video = document.getElementById(videoElementId);
        if (!video) {
            throw new Error('Video element not found');
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });

            video.srcObject = stream;
            await video.play();

            return stream;
        } catch (error) {
            this.log(`Camera error: ${error.message}`, 'error');
            throw error;
        }
    }

    static stopCamera(videoElementId) {
        const video = document.getElementById(videoElementId);
        if (video && video.srcObject) {
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
    }

    // Capture frame from video
    static captureFrame(videoElementId, canvasElementId) {
        const video = document.getElementById(videoElementId);
        const canvas = document.getElementById(canvasElementId);

        if (!video || !canvas) {
            throw new Error('Video or canvas element not found');
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        return canvas.toDataURL('image/jpeg', 0.8);
    }

    // API вызовы
    static async apiCall(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            this.log(`API call: ${endpoint}`, 'info');
            const response = await fetch(`${this.baseURL}${endpoint}`, mergedOptions);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            this.log(`API error (${endpoint}): ${error.message}`, 'error');
            throw error;
        }
    }

    // Валидация данных
    static validateName(name, fieldName = 'Имя') {
        if (!name || name.trim().length === 0) {
            throw new Error(`${fieldName} не может быть пустым`);
        }

        if (name.length < 2) {
            throw new Error(`${fieldName} должно содержать至少 2 символа`);
        }

        if (!/^[a-zA-Zа-яА-ЯёЁ\s\-]+$/i.test(name)) {
            throw new Error(`${fieldName} содержит недопустимые символы`);
        }

        return name.trim();
    }
}