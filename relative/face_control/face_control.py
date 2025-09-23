from flask import Flask, request, jsonify
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import json
import os
import base64
import time
from datetime import datetime
import uuid
from threading import RLock
from flask_cors import CORS
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)

class FaceAnalysisServer:
    def __init__(self):
        logger.info("Initializing FaceAnalysis...")
        self.face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=-1, det_size=(320, 320))
        self.users_file = "../../tmp/face_control.json"
        self.users = self.load_users()
        self.similarity_threshold = 0.6
        self.consistency_threshold = 0.7  # Порог для проверки постоянства лица

        self.liveness_sessions = {}
        self.registration_sessions = {}
        self.lock = RLock()

        self.stage_timeout = 25
        self.min_embeddings_per_stage = {
            "center": 10,
            "right": 6,
            "left": 6
        }
        logger.info("FaceAnalysis initialized successfully")

    def load_users(self):
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading users: {e}")
                return {}
        return {}

    def save_users(self):
        try:
            users_serializable = {}
            for user_id, user_data in self.users.items():
                users_serializable[user_id] = {
                    "first_name": user_data["first_name"],
                    "last_name": user_data["last_name"],
                    "embedding": [float(x) for x in user_data["embedding"]],
                    "registration_date": user_data["registration_date"],
                    "liveness_check": user_data["liveness_check"],
                    "embeddings_count": user_data["embeddings_count"],
                    "quality_checked": user_data["quality_checked"]
                }

            with open(self.users_file, 'w') as f:
                json.dump(users_serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    def decode_image(self, image_data):
        """Декодирует base64 изображение"""
        try:
            # Убираем префикс data:image/jpeg;base64, если есть
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode image")
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def calculate_similarity(self, embedding1, embedding2):
        """Вычисляет косинусное сходство между двумя эмбеддингами"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)  # Преобразуем в стандартный float
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def is_user_already_registered(self, new_embedding):
        """Проверяет, не зарегистрирован ли уже пользователь с похожим лицом"""
        with self.lock:
            for user_id, user_data in self.users.items():
                stored_embedding = user_data["embedding"]
                similarity = self.calculate_similarity(new_embedding, stored_embedding)

                if similarity > self.similarity_threshold:
                    logger.info(f"User {user_id} already registered with similarity: {similarity:.4f}")
                    return True, user_id, similarity

            return False, None, 0.0

    def check_face_consistency(self, current_embedding, reference_embeddings):
        """Проверяет, что текущее лицо соответствует предыдущим эмбеддингам сессии"""
        if not reference_embeddings:
            return True, 1.0  # Нет предыдущих эмбеддингов для сравнения

        similarities = []
        for ref_embedding in reference_embeddings:
            similarity = self.calculate_similarity(current_embedding, ref_embedding)
            similarities.append(similarity)

        avg_similarity = float(np.mean(similarities)) if similarities else 0.0  # Преобразуем в float
        is_consistent = avg_similarity > self.consistency_threshold

        logger.info(f"Face consistency check: avg_similarity={avg_similarity:.4f}, consistent={is_consistent}")
        return is_consistent, avg_similarity

    def start_registration(self, first_name, last_name):
        with self.lock:
            user_id = f"{first_name}_{last_name}".lower()

            if user_id in self.users:
                return {"error": "Пользователь с таким именем уже существует"}

            session_id = str(uuid.uuid4())

            stages = [
                {"name": "center", "instruction": "Смотрите прямо", "completed": False,
                 "embeddings": [], "min_embeddings": self.min_embeddings_per_stage["center"]},
                {"name": "right", "instruction": "Поверните вправо", "completed": False,
                 "embeddings": [], "min_embeddings": self.min_embeddings_per_stage["right"]},
                {"name": "left", "instruction": "Поверните влево", "completed": False,
                 "embeddings": [], "min_embeddings": self.min_embeddings_per_stage["left"]}
            ]

            self.registration_sessions[session_id] = {
                "user_id": user_id,
                "first_name": first_name,
                "last_name": last_name,
                "stages": stages,
                "current_stage": 0,
                "stage_start_time": time.time(),
                "stage_timeout": self.stage_timeout,
                "reference_embeddings": [],  # Эмбеддинги для проверки постоянства
                "consistency_checked": False
            }

            logger.info(f"Started registration for {first_name} {last_name}, session: {session_id}")
            return {
                "session_id": session_id,
                "stages": stages,
                "message": "Регистрация начата"
            }

    def process_registration_frame(self, session_id, image_data):
        """Обрабатывает кадр для регистрации пользователя"""
        logger.info(f"Processing frame for session: {session_id}")

        with self.lock:
            if session_id not in self.registration_sessions:
                logger.error(f"Session not found: {session_id}")
                return {"error": "Сессия не найдена"}

            session_data = self.registration_sessions[session_id]
            current_stage_idx = session_data["current_stage"]
            stage = session_data["stages"][current_stage_idx]

        # Декодируем изображение
        frame = self.decode_image(image_data)
        if frame is None:
            return {"error": "Не удалось декодировать изображение"}

        # Анализируем лица
        faces = self.face_app.get(frame)
        logger.info(f"Found {len(faces)} faces in the frame")

        head_pose = "unknown"
        quality_ok = False
        quality_issues = []
        det_score = 0
        face_embedding = None
        consistency_ok = True
        consistency_score = 1.0

        if len(faces) == 1:
            face = faces[0]
            det_score = float(face.det_score)  # Преобразуем в стандартный float
            head_pose = self.get_head_pose(face)
            quality_ok, quality_issues = self.check_face_quality(face, frame)

            logger.info(f"Face detected: pose={head_pose}, quality_ok={quality_ok}, required_pose={stage['name']}")

            if head_pose == stage["name"] and quality_ok and not stage["completed"]:
                face_embedding = face.embedding.tolist()

                # Проверяем постоянство лица (кроме первого кадра)
                with self.lock:
                    if session_data["reference_embeddings"]:
                        consistency_ok, consistency_score = self.check_face_consistency(
                            face_embedding, session_data["reference_embeddings"]
                        )

                    if consistency_ok:
                        # Добавляем в reference embeddings для будущих проверок
                        session_data["reference_embeddings"].append(face_embedding)
                        logger.info(f"Face consistency check passed: {consistency_score:.4f}")
                    else:
                        logger.warning(f"Face consistency check failed: {consistency_score:.4f}")

                logger.info(f"Valid face found for stage {stage['name']}")

        with self.lock:
            if session_id not in self.registration_sessions:
                return {"error": "Сессия не найдена"}

            session_data = self.registration_sessions[session_id]
            current_stage_idx = session_data["current_stage"]
            stage = session_data["stages"][current_stage_idx]

            # Добавляем embedding если подходит и прошло проверку постоянства
            if (face_embedding and head_pose == stage["name"] and quality_ok and
                    consistency_ok and not stage["completed"] and
                    len(stage["embeddings"]) < stage["min_embeddings"] + 5):

                stage["embeddings"].append(face_embedding)
                logger.info(f"Added embedding to stage {stage['name']}, total: {len(stage['embeddings'])}")

            # Проверяем завершение этапа
            if len(stage["embeddings"]) >= stage["min_embeddings"] and not stage["completed"]:
                stage["completed"] = True
                all_stages_completed = all(s["completed"] for s in session_data["stages"])

                if all_stages_completed:
                    # Завершаем регистрацию
                    all_embeddings = []
                    for s in session_data["stages"]:
                        all_embeddings.extend(s["embeddings"])

                    # Создаем усредненный эмбеддинг
                    avg_embedding = np.mean(all_embeddings, axis=0).tolist()

                    # Проверяем, не зарегистрирован ли уже пользователь
                    already_registered, existing_user_id, similarity = self.is_user_already_registered(avg_embedding)

                    if already_registered:
                        # Удаляем сессию
                        del self.registration_sessions[session_id]

                        logger.warning(f"User already registered: {existing_user_id} (similarity: {similarity:.4f})")
                        return {
                            "status": "already_registered",
                            "message": f"Пользователь уже зарегистрирован (сходство: {similarity:.2%})",
                            "existing_user_id": existing_user_id,
                            "similarity": similarity
                        }

                    user_id = session_data["user_id"]

                    self.users[user_id] = {
                        "first_name": session_data["first_name"],
                        "last_name": session_data["last_name"],
                        "embedding": avg_embedding,
                        "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "liveness_check": True,
                        "embeddings_count": len(all_embeddings),
                        "quality_checked": True
                    }

                    self.save_users()
                    del self.registration_sessions[session_id]

                    logger.info(f"Registration completed for {user_id}")
                    return {
                        "status": "completed",
                        "message": "Регистрация завершена успешно!",
                        "user_id": user_id
                    }
                else:
                    # Переходим к следующему этапу
                    session_data["current_stage"] += 1
                    session_data["stage_start_time"] = time.time()
                    logger.info(f"Moving to next stage: {session_data['current_stage']}")

            # Подготавливаем ответ - ВАЖНО: преобразуем все NumPy типы в стандартные Python типы
            current_stage_idx = session_data["current_stage"]
            stage = session_data["stages"][current_stage_idx]

            response = {
                "status": "processing",
                "current_stage": int(current_stage_idx),  # Преобразуем в int
                "stage_name": stage["name"],
                "collected": int(len(stage["embeddings"])),  # Преобразуем в int
                "needed": int(stage["min_embeddings"]),  # Преобразуем в int
                "head_pose": head_pose,
                "quality_ok": bool(quality_ok),  # Преобразуем в bool
                "quality_issues": quality_issues,
                "faces_count": int(len(faces)),  # Преобразуем в int
                "det_score": float(det_score),  # Преобразуем в float
                "stage_completed": bool(stage["completed"]),  # Преобразуем в bool
                "face_consistent": bool(consistency_ok),  # Преобразуем в bool
                "consistency_score": float(consistency_score)  # Преобразуем в float
            }

            logger.info(f"Response: {response}")

            # Проверяем таймаут
            if not stage["completed"]:
                elapsed_time = time.time() - session_data["stage_start_time"]
                if elapsed_time > session_data["stage_timeout"]:
                    response["status"] = "timeout"
                    response["message"] = f"Время вышло на этапе '{stage['name']}'!"
                    if session_id in self.registration_sessions:
                        del self.registration_sessions[session_id]

        return response

    def get_head_pose(self, face):
        """Определяет позу головы"""
        try:
            if not hasattr(face, 'kps') or face.kps is None:
                return "unknown"

            left_eye, right_eye, nose = face.kps[:3]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            if eye_distance < 1e-6:
                return "unknown"

            eyes_center = (left_eye + right_eye) / 2
            nose_offset = nose[0] - eyes_center[0]
            normalized_offset = nose_offset / eye_distance

            if normalized_offset < -0.15:
                return "left"
            elif normalized_offset > 0.15:
                return "right"
            else:
                return "center"
        except Exception as e:
            logger.error(f"Error getting head pose: {e}")
            return "unknown"

    def check_face_quality(self, face, frame):
        """Проверяет качество обнаруженного лица"""
        quality_issues = []
        try:
            det_score = float(face.det_score)
            if det_score < 0.7:
                quality_issues.append("Низкое качество распознавания")

            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]

            min_face_size = 80
            if face_width < min_face_size or face_height < min_face_size:
                quality_issues.append("Подойдите ближе")
            elif face_width > 400 or face_height > 400:
                quality_issues.append("Отойдите немного")

            if not hasattr(face, 'kps') or face.kps is None or len(face.kps) < 5:
                quality_issues.append("Не видно черт лица")

            return len(quality_issues) == 0, quality_issues
        except Exception as e:
            logger.error(f"Error checking face quality: {e}")
            return False, ["Ошибка проверки качества"]

    def start_liveness_check(self):
        """Начинает проверку живости"""
        session_id = str(time.time())
        self.liveness_sessions[session_id] = {
            'start_time': time.time(),
            'poses': [],
            'required_poses': ['center', 'right', 'left', 'center'],
            'current_step': 0,
            'completed': False,
            'last_pose_time': time.time(),
            'reference_embedding': None  # Для проверки постоянства лица
        }
        logger.info(f"Started liveness check session: {session_id}")
        return {"session_id": session_id, "instruction": "Смотрите прямо в камеру"}

    def update_liveness_check(self, session_id, image_data):
        """Обновляет проверку живости"""
        logger.info(f"Updating liveness for session: {session_id}")

        if session_id not in self.liveness_sessions:
            logger.error(f"Liveness session not found: {session_id}")
            return {"error": "Сессия не найдена"}

        session = self.liveness_sessions[session_id]

        # Проверяем таймаут
        if time.time() - session['start_time'] > 30:
            del self.liveness_sessions[session_id]
            return {"error": "Время вышло", "completed": False}

        # Декодируем изображение
        frame = self.decode_image(image_data)
        if frame is None:
            return {"error": "Не удалось декодировать изображение"}

        # Анализируем лица
        faces = self.face_app.get(frame)
        logger.info(f"Found {len(faces)} faces for liveness check")

        if len(faces) != 1:
            return {
                "session_id": session_id,
                "instruction": "Покажите одно лицо в камеру",
                "current_step": session['current_step'],
                "total_steps": len(session['required_poses']),
                "completed": False
            }

        face = faces[0]
        current_pose = self.get_head_pose(face)
        current_embedding = face.embedding.tolist()

        # Проверяем постоянство лица
        if session['reference_embedding'] is None:
            session['reference_embedding'] = current_embedding
        else:
            consistency_ok, consistency_score = self.check_face_consistency(
                current_embedding, [session['reference_embedding']]
            )
            if not consistency_ok:
                return {
                    "session_id": session_id,
                    "instruction": "Обнаружено другое лицо! Продолжайте проверку с исходным лицом",
                    "current_step": session['current_step'],
                    "total_steps": len(session['required_poses']),
                    "completed": False,
                    "face_changed": True
                }

        session['poses'].append(current_pose)

        # Проверяем текущую требуемую позу
        required_pose = session['required_poses'][session['current_step']]

        if current_pose == required_pose:
            # Проверяем стабильность позы
            stable_count = sum(1 for p in session['poses'][-10:] if p == required_pose)

            if stable_count >= 5:
                session['current_step'] += 1
                session['last_pose_time'] = time.time()

                if session['current_step'] >= len(session['required_poses']):
                    session['completed'] = True
                    logger.info(f"Liveness check completed for session: {session_id}")
                    return {
                        "session_id": session_id,
                        "instruction": "Проверка завершена!",
                        "current_step": session['current_step'],
                        "total_steps": len(session['required_poses']),
                        "completed": True
                    }
                else:
                    next_pose = session['required_poses'][session['current_step']]
                    instructions = {
                        "center": "Смотрите прямо в камеру",
                        "right": "Поверните голову влево",
                        "left": "Поверните голову вправо"
                    }
                    return {
                        "session_id": session_id,
                        "instruction": instructions[next_pose],
                        "current_step": session['current_step'],
                        "total_steps": len(session['required_poses']),
                        "completed": False
                    }

        # Возвращаем текущую инструкцию
        instructions = {
            "center": "Смотрите прямо в камеру",
            "right": "Поверните голову влево",
            "left": "Поверните голову вправо"
        }

        return {
            "session_id": session_id,
            "instruction": instructions[required_pose],
            "current_step": session['current_step'],
            "total_steps": len(session['required_poses']),
            "current_pose": current_pose,
            "completed": False
        }

    def authenticate_with_liveness(self, session_id, image_data):
        """Аутентификация после проверки живости"""
        logger.info(f"Authenticating with liveness for session: {session_id}")

        if session_id not in self.liveness_sessions or not self.liveness_sessions[session_id]['completed']:
            return {"authenticated": False, "error": "Liveness check not completed"}

        # Декодируем изображение
        frame = self.decode_image(image_data)
        if frame is None:
            return {"authenticated": False, "error": "Не удалось декодировать изображение"}

        # Анализируем лица
        faces = self.face_app.get(frame)
        logger.info(f"Found {len(faces)} faces for authentication")

        if len(faces) != 1:
            return {"authenticated": False, "error": "No face detected"}

        face = faces[0]
        current_embedding = face.embedding

        max_similarity = 0
        authenticated_user = None

        for user_id, user_data in self.users.items():
            stored_embedding = np.array(user_data['embedding'])
            similarity = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
            )

            if similarity > max_similarity:
                max_similarity = similarity
                authenticated_user = user_data

        if authenticated_user and max_similarity > self.similarity_threshold:
            # Создаем безопасный ответ
            safe_user_data = {
                'first_name': authenticated_user.get('first_name'),
                'last_name': authenticated_user.get('last_name'),
                'registration_date': authenticated_user.get('registration_date')
            }

            del self.liveness_sessions[session_id]

            logger.info(f"Authentication successful for user: {authenticated_user.get('first_name')}")
            return {
                'authenticated': True,
                'user': safe_user_data,
                'similarity': float(max_similarity),
                'liveness_passed': True
            }

        logger.info("Authentication failed - no matching user found")
        return {'authenticated': False, 'error': 'Authentication failed - пользователь не найден'}

    def cleanup_sessions(self):
        """Очищает старые сессии"""
        current_time = time.time()

        # Очистка сессий регистрации
        expired_registrations = []
        for session_id, session_data in self.registration_sessions.items():
            elapsed_time = current_time - session_data.get('stage_start_time', current_time)
            if elapsed_time > 300:
                expired_registrations.append(session_id)

        for session_id in expired_registrations:
            del self.registration_sessions[session_id]
            logger.info(f"Cleaned up expired registration session: {session_id}")

        # Очистка сессий аутентификации
        expired_liveness = []
        for session_id, session_data in self.liveness_sessions.items():
            if current_time - session_data['start_time'] > 300:
                expired_liveness.append(session_id)

        for session_id in expired_liveness:
            del self.liveness_sessions[session_id]
            logger.info(f"Cleaned up expired liveness session: {session_id}")

        return {
            "active_registrations": len(self.registration_sessions),
            "active_authentications": len(self.liveness_sessions)
        }

# Инициализация сервера анализа
analysis_server = FaceAnalysisServer()

# Роуты Python сервера
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "Python analysis server running",
        "users_count": len(analysis_server.users)
    })

@app.route('/api/start_registration', methods=['POST'])
def start_registration():
    try:
        data = request.json
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()

        if not first_name or not last_name:
            return jsonify({"error": "Имя и фамилия обязательны"}), 400

        result = analysis_server.start_registration(first_name, last_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in start_registration: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_registration', methods=['POST'])
def process_registration():
    try:
        data = request.json
        frame_data = data.get('frame', '')
        session_id = data.get('session_id', '')

        if not frame_data or not session_id:
            return jsonify({"error": "Отсутствуют данные кадра или идентификатор сессии"}), 400

        result = analysis_server.process_registration_frame(session_id, frame_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in process_registration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cancel_registration', methods=['POST'])
def cancel_registration():
    try:
        data = request.json
        session_id = data.get('session_id', '')

        with analysis_server.lock:
            if session_id in analysis_server.registration_sessions:
                del analysis_server.registration_sessions[session_id]
                logger.info(f"Cancelled registration for session: {session_id}")

        return jsonify({"message": "Регистрация отменена"})
    except Exception as e:
        logger.error(f"Error in cancel_registration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_liveness', methods=['POST'])
def start_liveness():
    try:
        result = analysis_server.start_liveness_check()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in start_liveness: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_liveness', methods=['POST'])
def update_liveness():
    try:
        data = request.json
        session_id = data.get('session_id')
        image_data = data.get('image')

        if not session_id or not image_data:
            return jsonify({'error': 'Missing session_id or image'}), 400

        result = analysis_server.update_liveness_check(session_id, image_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in update_liveness: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        data = request.json
        session_id = data.get('session_id')
        image_data = data.get('image')

        if not session_id or not image_data:
            return jsonify({'error': 'Missing session_id or image'}), 400

        result = analysis_server.authenticate_with_liveness(session_id, image_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in authenticate: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Python analysis server running on port 5001")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)