class FaceRecognitionApp {
    constructor() {
        this.baseURL = '';
        this.initializeElements();
        this.initializeEventListeners();
        this.currentTab = 'registration';
        this.registrationSessionId = null;
        this.authSessionId = null;
        this.isProcessing = false;
        this.debugMode = true;
        this.stageRequirements = [];
        this.stageIndexByName = {};
    }

    initializeElements() {
        this.tabs = document.querySelectorAll('.tab');
        this.tabContents = document.querySelectorAll('.tab-content');

        this.firstNameInput = document.getElementById('firstName');
        this.lastNameInput = document.getElementById('lastName');
        this.registrationVideo = document.getElementById('registrationVideo');
        this.registrationCanvas = document.getElementById('registrationCanvas');
        this.registrationVideoContainer = this.registrationVideo ? this.registrationVideo.parentElement : null;
        this.startRegistrationBtn = document.getElementById('startRegistration');
        this.stopRegistrationBtn = document.getElementById('stopRegistration');
        this.registrationStatus = document.getElementById('registrationStatus');
        this.qualityInfo = document.getElementById('qualityInfo');
        this.registrationStages = document.getElementById('registrationStages');
        this.debugInfo = document.getElementById('debugInfo');

        this.authVideo = document.getElementById('authVideo');
        this.authCanvas = document.getElementById('authCanvas');
        this.authVideoContainer = this.authVideo ? this.authVideo.parentElement : null;
        this.startAuthBtn = document.getElementById('startAuth');
        this.stopAuthBtn = document.getElementById('stopAuth');
        this.authStatus = document.getElementById('authStatus');
        this.authProgress = document.getElementById('authProgress');
        this.authResults = document.getElementById('authResults');
        this.userResult = document.getElementById('userResult');
        this.authDebugInfo = document.getElementById('authDebugInfo');

        this.registrationCtx = this.registrationCanvas.getContext('2d');
        this.authCtx = this.authCanvas.getContext('2d');
    }

    initializeEventListeners() {
        this.tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabName = tab.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });

        this.startRegistrationBtn.addEventListener('click', () => this.startRegistration());
        this.stopRegistrationBtn.addEventListener('click', () => this.stopRegistration());

        this.startAuthBtn.addEventListener('click', () => this.startAuthentication());
        this.stopAuthBtn.addEventListener('click', () => this.stopAuthentication());

        this.firstNameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.startRegistration();
        });
        this.lastNameInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.startRegistration();
        });
    }

    switchTab(tabName) {
        this.tabs.forEach(tab => tab.classList.remove('active'));
        this.tabContents.forEach(content => content.classList.remove('active'));

        document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        this.stopRegistration();
        this.stopAuthentication();
    }

    async startRegistration() {
        const firstName = this.firstNameInput.value.trim();
        const lastName = this.lastNameInput.value.trim();

        if (!firstName || !lastName) {
            this.showStatus('registration', 'error', 'Введите имя и фамилию');
            return;
        }

        try {
            this.showStatus('registration', 'info', 'Запуск камеры...');
            this.showDebugInfo('registration', 'Запрос доступа к камере...');

            await this.startCamera('registration');

            this.showDebugInfo('registration', 'Камера запущена. Отправка запроса на сервер...');

            const response = await fetch(this.baseURL + '/face-control/registration/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    first_name: firstName,
                    last_name: lastName
                })
            });

            this.showDebugInfo('registration', `Статус ответа: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();

            this.showDebugInfo('registration', 'Ответ от сервера получен: ' + JSON.stringify(data).substring(0, 200) + '...');

            if (data.error) {
                this.showStatus('registration', 'error', data.error);
                return;
            }

            this.registrationSessionId = data.session_id;
            this.startRegistrationBtn.disabled = true;
            this.stopRegistrationBtn.disabled = false;

            if (data.stages && Array.isArray(data.stages)) {
                this.updateStages(data.stages);
                this.showStatus('registration', 'info', 'Регистрация начата. Следуйте инструкциям.');
            } else {
                const defaultStages = [
                    { name: 'center', instruction: 'Смотрите прямо', min_embeddings: 10 },
                    { name: 'right', instruction: 'Поверните влево', min_embeddings: 6 },
                    { name: 'left', instruction: 'Поверните вправо', min_embeddings: 6 }
                ];
                this.updateStages(defaultStages);
                this.showStatus('registration', 'info', 'Регистрация начата. Следуйте инструкциям.');
            }

            this.processRegistrationFrames();

        } catch (error) {
            console.error('Registration error:', error);
            this.showStatus('registration', 'error', `Ошибка: ${error.message}`);
            this.showDebugInfo('registration', `Ошибка: ${error.message}`);
        }
    }

    async processRegistrationFrames() {
        if (!this.registrationSessionId || this.isProcessing) return;
        this.isProcessing = true;
        try {
            const canvas = this.registrationCanvas;
            canvas.width = this.registrationVideo.videoWidth;
            canvas.height = this.registrationVideo.videoHeight;

            this.registrationCtx.drawImage(this.registrationVideo, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);

            this.showDebugInfo('registration', 'Отправка кадра на обработку...');

            const response = await fetch(this.baseURL + '/face-control/registration/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.registrationSessionId,
                    frame: frameData
                })
            });

            this.showDebugInfo('registration', `Статус обработки кадра: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();

            this.showDebugInfo('registration', 'Результат обработки: ' + JSON.stringify(data).substring(0, 150) + '...');

            if (data.error) {
                this.showStatus('registration', 'error', data.error);
                this.stopRegistration();
                return;
            }

            if (data.status === 'already_registered') {
                this.showStatus('registration', 'error', data.message);
                this.stopRegistration(true);
                return;
            }

            if (data.status === 'completed') {
                this.finalizeRegistrationUI();
                this.showStatus('registration', 'success', data.message || 'Регистрация завершена успешно!');
                this.stopRegistration(true);
                return;
            }

            if (data.status === 'timeout') {
                this.showStatus('registration', 'error', data.message || 'Время вышло');
                this.stopRegistration(true);
                return;
            }

            this.updateRegistrationUI(data);

            if (this.registrationSessionId) {
                setTimeout(() => this.processRegistrationFrames(), 100);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
            this.showDebugInfo('registration', `Ошибка обработки кадра: ${error.message}`);
            if (this.registrationSessionId) {
                setTimeout(() => this.processRegistrationFrames(), 100);
            }
        } finally {
            this.isProcessing = false;
        }
    }

    updateRegistrationUI(data) {
        const stages = this.registrationStages.querySelectorAll('.stage');
        const currentStage = typeof data.current_stage === 'number' ? data.current_stage : 0;
        const stageName = data.stage_name || 'center';

        if (Array.isArray(data.stages_progress)) {
            data.stages_progress.forEach(sp => {
                const idx = this.stageIndexByName[sp.name];
                if (idx === undefined) return;
                const stageDiv = stages[idx];
                if (!stageDiv) return;
                const progress = stageDiv.querySelector('.progress');
                const neededLocal = typeof sp.needed === 'number' ? sp.needed : (this.stageRequirements[idx]?.needed || 10);
                const collectedLocal = typeof sp.collected === 'number' ? sp.collected : 0;
                if (progress) progress.textContent = `${collectedLocal}/${neededLocal}`;
                stageDiv.classList.remove('active', 'completed');
                if (sp.completed || collectedLocal >= neededLocal) {
                    stageDiv.classList.add('completed');
                } else if (sp.active) {
                    stageDiv.classList.add('active');
                }
            });
        } else {
            const needed = typeof data.needed === 'number' ? data.needed : (this.stageRequirements[currentStage]?.needed || 10);
            const collected = typeof data.collected === 'number' ? data.collected : 0;
            stages.forEach((stage, index) => {
                const progress = stage.querySelector('.progress');
                stage.classList.remove('active', 'completed');
                if (index === currentStage) {
                    if (progress) progress.textContent = `${collected}/${needed}`;
                    stage.classList.add('active');
                } else if (index < currentStage) {
                    stage.classList.add('completed');
                    const req = this.stageRequirements[index];
                    if (req && progress) progress.textContent = `${req.needed}/${req.needed}`;
                }
            });
        }

        const neededForStatus = this.stageRequirements[currentStage]?.needed || (typeof data.needed === 'number' ? data.needed : 10);
        const collectedForStatus = typeof data.collected === 'number' ? data.collected : 0;
        let statusMessage = `Этап: ${this.getStageName(stageName)} | `;
        statusMessage += `Собрано: ${collectedForStatus}/${neededForStatus}`;

        if (data.quality_issues && Array.isArray(data.quality_issues)) {
            this.qualityInfo.innerHTML = data.quality_issues.map(issue =>
                `<div class="quality-issues">⚠️ ${issue}</div>`
            ).join('');
        } else {
            this.qualityInfo.innerHTML = '<div class="success">✓ Качество хорошее</div>';
        }

        this.showStatus('registration', 'info', statusMessage);
    }

    getStageName(stage) {
        const names = { 'center': 'Прямо', 'right': 'Влево', 'left': 'Вправо' };
        return names[stage] || stage;
    }

    updateStages(stages) {
        const stagesContainer = this.registrationStages;
        stagesContainer.innerHTML = '';

        const stagesToUse = Array.isArray(stages) ? stages : [
            { name: 'center', min_embeddings: 10 },
            { name: 'right', min_embeddings: 6 },
            { name: 'left', min_embeddings: 6 }
        ];

        this.stageRequirements = stagesToUse.map(s => ({ name: s.name, needed: s.min_embeddings || 10 }));
        this.stageIndexByName = {};
        this.stageRequirements.forEach((s, i) => { this.stageIndexByName[s.name] = i; });

        stagesToUse.forEach((stage) => {
            const stageDiv = document.createElement('div');
            stageDiv.className = 'stage';
            stageDiv.setAttribute('data-stage', stage.name);
            stageDiv.innerHTML = `
                        <h3>${this.getStageName(stage.name)}</h3>
                        <div class="progress">0/${stage.min_embeddings || 10}</div>
                    `;
            stagesContainer.appendChild(stageDiv);
        });
    }

    finalizeRegistrationUI() {
        const stages = this.registrationStages.querySelectorAll('.stage');
        stages.forEach((stageDiv) => {
            const name = stageDiv.getAttribute('data-stage');
            const req = this.stageRequirements[this.stageIndexByName[name]];
            const needed = req ? req.needed : 0;
            const progress = stageDiv.querySelector('.progress');
            if (progress && needed) progress.textContent = `${needed}/${needed}`;
            stageDiv.classList.remove('active');
            stageDiv.classList.add('completed');
        });
    }

    stopRegistration(preserveStatus = false) {
        if (this.registrationSessionId) {
            fetch(this.baseURL + '/face-control/registration/cancel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_id: this.registrationSessionId })
            }).catch(error => {
                console.error('Cancel registration error:', error);
            });
        }

        this.registrationSessionId = null;
        this.startRegistrationBtn.disabled = false;
        this.stopRegistrationBtn.disabled = true;
        if (!preserveStatus) {
            this.showStatus('registration', 'info', 'Регистрация остановлена');
        }
        this.showDebugInfo('registration', 'Регистрация остановлена');
        this.stopCamera('registration');
    }

    async startAuthentication() {
        try {
            this.showStatus('authentication', 'info', 'Запуск камеры...');
            this.showDebugInfo('authentication', 'Запуск проверки живости...');

            await this.startCamera('authentication');

            const response = await fetch(this.baseURL + '/face-control/auth/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            this.showDebugInfo('authentication', `Статус запуска проверки: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            this.showDebugInfo('authentication', 'Ответ проверки живости: ' + JSON.stringify(data).substring(0, 100) + '...');

            if (data.error) {
                this.showStatus('authentication', 'error', data.error);
                return;
            }

            this.authSessionId = data.session_id;
            this.startAuthBtn.disabled = true;
            this.stopAuthBtn.disabled = false;
            this.authResults.style.display = 'none';

            this.showStatus('authentication', 'info', data.instruction || 'Следуйте инструкциям');
            this.processAuthenticationFrames();
        } catch (error) {
            console.error('Authentication error:', error);
            this.showStatus('authentication', 'error', `Ошибка: ${error.message}`);
            this.showDebugInfo('authentication', `Ошибка: ${error.message}`);
        }
    }

    async processAuthenticationFrames() {
        if (!this.authSessionId || this.isProcessing) return;
        this.isProcessing = true;
        try {
            const canvas = this.authCanvas;
            canvas.width = this.authVideo.videoWidth;
            canvas.height = this.authVideo.videoHeight;

            this.authCtx.drawImage(this.authVideo, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);

            this.showDebugInfo('authentication', 'Отправка кадра для проверки живости...');

            const response = await fetch(this.baseURL + '/face-control/auth/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.authSessionId, image: frameData })
            });

            this.showDebugInfo('authentication', `Статус проверки кадра: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            this.showDebugInfo('authentication', 'Результат проверки: ' + JSON.stringify(data).substring(0, 150) + '...');

            if (data.error) {
                this.showStatus('authentication', 'error', data.error);
                this.stopAuthentication(true);
                return;
            }

            if (data.completed) {
                await this.authenticateUser(frameData);
                return;
            }

            this.authProgress.innerHTML = `
            <div>Шаг ${(data.current_step || 0) + 1} из ${data.total_steps || 4}</div>
            <div class="instructions">${data.instruction || 'Следуйте инструкциям'}</div>
        `;

            if (this.authSessionId) {
                setTimeout(() => this.processAuthenticationFrames(), 100);
            }
        } catch (error) {
            console.error('Error processing auth frame:', error);
            this.showDebugInfo('authentication', `Ошибка обработки кадра: ${error.message}`);
            if (this.authSessionId) {
                setTimeout(() => this.processAuthenticationFrames(), 100);
            }
        } finally {
            this.isProcessing = false;
        }
    }

    async authenticateUser(frameData) {
        try {
            this.showStatus('authentication', 'info', 'Проверка личности...');
            this.showDebugInfo('authentication', 'Начало аутентификации...');

            const response = await fetch(this.baseURL + '/face-control/auth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: this.authSessionId, image: frameData })
            });

            this.showDebugInfo('authentication', `Статус аутентификации: ${response.status}`);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            this.showDebugInfo('authentication', 'Результат аутентификации: ' + JSON.stringify(data).substring(0, 200) + '...');

            if (data.authenticated) {
                this.showAuthenticationResult(data);
            } else {
                this.showStatus('authentication', 'error', data.error || 'Пользователь не найден');
                this.stopAuthentication(true);
            }
        } catch (error) {
            console.error('Authentication error:', error);
            this.showStatus('authentication', 'error', `Ошибка аутентификации: ${error.message}`);
            this.showDebugInfo('authentication', `Ошибка аутентификации: ${error.message}`);
            this.stopAuthentication(true);
        }
    }

    showAuthenticationResult(data) {
        this.authResults.style.display = 'block';
        this.userResult.innerHTML = `
                    <div class="user-card success">
                        <h3>✅ Пользователь найден!</h3>
                        <p><strong>Имя:</strong> ${data.user?.first_name || 'Неизвестно'} ${data.user?.last_name || ''}</p>
                        <p><strong>Дата регистрации:</strong> ${data.user?.registration_date || 'Неизвестно'}</p>
                        <p><strong>Схожесть:</strong> ${(data.similarity * 100).toFixed(1)}%</p>
                    </div>
                `;
        this.stopAuthentication();
    }

    stopAuthentication(preserveStatus = false) {
        this.authSessionId = null;
        this.startAuthBtn.disabled = false;
        this.stopAuthBtn.disabled = true;
        if (!preserveStatus) {
            this.showStatus('authentication', 'info', 'Поиск остановлен');
        }
        this.showDebugInfo('authentication', 'Поиск остановлен');
        this.stopCamera('authentication');
    }

    async startCamera(type) {
        const video = type === 'registration' ? this.registrationVideo : this.authVideo;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false
            });
            video.srcObject = stream;
            await video.play();
            const canvas = type === 'registration' ? this.registrationCanvas : this.authCanvas;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Show container when camera is active
            const container = type === 'registration' ? this.registrationVideoContainer : this.authVideoContainer;
            if (container) container.style.display = 'block';
        } catch (error) {
            this.showStatus(type, 'error', 'Не удалось получить доступ к камере');
            this.showDebugInfo(type, `Ошибка камеры: ${error.message}`);
            throw error;
        }
    }

    stopCamera(type) {
        const video = type === 'registration' ? this.registrationVideo : this.authVideo;
        if (video.srcObject) {
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            video.srcObject = null;
        }
        // Hide container when camera is stopped
        const container = type === 'registration' ? this.registrationVideoContainer : this.authVideoContainer;
        if (container) container.style.display = 'none';
    }

    showStatus(type, status, message) {
        const statusElement = type === 'registration' ? this.registrationStatus : this.authStatus;
        const instructionElement = statusElement.querySelector('.instructions');
        instructionElement.textContent = message;
        instructionElement.className = 'instructions';
        if (status === 'error') {
            instructionElement.classList.add('error');
        } else if (status === 'success') {
            instructionElement.classList.add('success');
        }
    }

    showDebugInfo(type, message) {
        if (!this.debugMode) return;
        const debugElement = type === 'registration' ? this.debugInfo : this.authDebugInfo;
        debugElement.textContent = `[DEBUG] ${new Date().toLocaleTimeString()}: ${message}`;
        debugElement.style.display = 'block';
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(this.baseURL + '/face-control/health');
            const data = await response.json();
            console.log('System health:', data);
            return data;
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error', message: 'System unavailable' };
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new FaceRecognitionApp();
    window.faceApp = app;
    app.checkSystemHealth().then(health => {
        if (health.status === 'ok') {
            console.log('✅ System is healthy');
            app.showDebugInfo('registration', 'Система работает нормально');
        } else {
            console.warn('⚠️ System issues detected');
            app.showDebugInfo('registration', 'Проблемы с системой: ' + health.message);
        }
    });
});



