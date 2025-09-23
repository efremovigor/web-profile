import io
import json
import sys
import signal
import time
import os
import threading
from typing import Optional

import numpy as np
import torch
import open_clip
from PIL import Image, ImageFile
import faiss
import pandas as pd
import grpc
from concurrent import futures
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
import image_search_pb2
import image_search_pb2_grpc
from logging.handlers import TimedRotatingFileHandler
import logging

# Увеличим порог для частично повреждённых изображений (помогает валидации)
ImageFile.LOAD_TRUNCATED_IMAGES = False

# --- Настройка логирования с ротацией (без дублирования) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Предотвращаем повторную установку хендлеров (при повторном импортировании)
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Снижает шум от библиотек, которые могут логировать двоичные данные
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("open_clip").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

# Защита от "бомб распаковки" — ограничение по пикселям
Image.MAX_IMAGE_PIXELS = 64_000_000


class Config:
    """Конфигурационные параметры сервиса"""
    PORT = 9999
    MAX_WORKERS = 10
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit
    DEFAULT_TOP_K = 10
    SEARCH_K = 1000
    CHUNK_SIZE = 10000  # для потоковой обработки CSV
    MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "2"))


class HealthServicer(health_pb2_grpc.HealthServicer):
    def __init__(self, search_service):
        self.search_service = search_service

    def Check(self, request, context):
        try:
            if (self.search_service.model is not None and
                    self.search_service.index is not None and
                    self.search_service.meta is not None):
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.SERVING
                )
            else:
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
        except Exception:
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )


class ImageSearchService(image_search_pb2_grpc.ImageSearchServiceServicer):
    def __init__(self, csv_path: str = 'petrovich_feed.csv'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.csv_path = csv_path
        self.healthy = False

        self._model = None
        self._preprocess = None
        self._index = None
        self._meta = None
        self._product_images_dict = None
        # семафор на количество одновременных инферов
        self._infer_sem = threading.Semaphore(value=Config.MAX_CONCURRENT_INFER)

        try:
            self.load_model_and_index()
            self.load_product_images()
            self.healthy = True
            logger.info("✅ Сервис инициализирован успешно")
        except Exception as e:
            # Не логируем двоичные данные — только текст ошибки
            logger.exception(f"❌ Ошибка инициализации сервиса: {e}")
            self.healthy = False

    def load_product_images(self) -> None:
        if self._product_images_dict is not None:
            return

        logger.info(f"📊 Загрузка изображений из {self.csv_path}...")
        start_time = time.time()
        self._product_images_dict = {}

        try:
            chunk_count = 0
            total_rows = 0
            for chunk in pd.read_csv(
                self.csv_path,
                usecols=['id', 'picture1'],
                dtype={'id': 'str'},
                na_filter=False,
                chunksize=Config.CHUNK_SIZE
            ):
                chunk_count += 1
                total_rows += len(chunk)
                chunk['id'] = chunk['id'].astype(str).str.strip()
                chunk['picture1'] = chunk['picture1'].astype(str).str.strip()
                # фильтруем пустые и 'nan' текстовые значения
                mask = chunk['id'].ne('') & chunk['picture1'].ne('') & chunk['picture1'].ne('nan')
                self._product_images_dict.update(dict(zip(chunk.loc[mask, 'id'], chunk.loc[mask, 'picture1'])))

            load_time = time.time() - start_time
            logger.info(
                f"✅ Загружено {len(self._product_images_dict)} изображений из {total_rows} строк за {load_time:.2f}с"
            )
            logger.info(f"📊 Обработано чанков: {chunk_count}")

        except FileNotFoundError:
            logger.warning(f"⚠️ CSV файл {self.csv_path} не найден. Продолжаем без изображений.")
            self._product_images_dict = {}
        except pd.errors.EmptyDataError:
            logger.warning(f"⚠️ CSV файл {self.csv_path} пуст.")
            self._product_images_dict = {}
        except pd.errors.ParserError as e:
            logger.exception(f"❌ Ошибка парсинга CSV: {e}")
            self._product_images_dict = {}
        except Exception as e:
            logger.exception(f"❌ Неожиданная ошибка загрузки CSV: {e}")
            self._product_images_dict = {}

    def get_product_image(self, product_id: str) -> Optional[str]:
        if not self._product_images_dict:
            return None
        product_id_str = str(product_id).strip()
        if product_id_str in self._product_images_dict:
            return self._product_images_dict[product_id_str]
        product_id_clean = product_id_str.lstrip('0')
        if product_id_clean and product_id_clean in self._product_images_dict:
            return self._product_images_dict[product_id_clean]
        return None

    @property
    def model(self):
        return self._model

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def index(self):
        return self._index

    @property
    def meta(self):
        return self._meta

    @property
    def product_images_dict(self):
        return self._product_images_dict

    def load_model_and_index(self) -> None:
        logger.info("🔄 Загрузка модели и индекса...")
        start_time = time.time()
        try:
            # Загружаем модель и препроцессоры
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained="laion2b_s32b_b82k"
            )
            # Переводим модель на устройство
            self._model.eval().to(self.device)

            # Загружаем faiss индекс и мета
            self._index = faiss.read_index("index/faiss.index")
            with open("index/meta.json", "r", encoding="utf-8") as f:
                self._meta = json.load(f)

            # Простая проверка
            assert self._index.ntotal == len(self._meta), "index/meta size mismatch"

            load_time = time.time() - start_time
            logger.info(f"✅ Модель и индекс загружены за {load_time:.2f}с")
            logger.info(f"📊 Размер индекса: {self._index.ntotal} элементов")
            logger.info(f"🎯 Устройство: {self.device.upper()}")
        except Exception as e:
            logger.exception(f"❌ Ошибка загрузки ресурсов: {e}")
            raise

    def _embed_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                # применяем preprocess (ожидается torchvision-подобный трансформ)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Ограничиваем число одновременных инферов
            with self._infer_sem:
                if self.device == "cuda":
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                        features = self.model.encode_image(image_input)
                        features = torch.nn.functional.normalize(features, dim=-1)
                else:
                    with torch.no_grad():
                        features = self.model.encode_image(image_input)
                        features = torch.nn.functional.normalize(features, dim=-1)

            arr = features.cpu().numpy().astype("float32")
            arr = np.ascontiguousarray(arr)
            return arr
        except Exception as e:
            # Не включаем двоичные данные в лог — только текст ошибки
            logger.exception(f"Ошибка генерации эмбеддинга: {e}")
            raise

    def _create_search_response(self, success: bool = True,
                                error: Optional[str] = None) -> image_search_pb2.SearchResponse:
        response = image_search_pb2.SearchResponse()
        response.success = success
        response.error = error or ""
        response.device = self.device
        response.cached = False
        return response

    def _validate_image_data(self, image_data: bytes) -> Optional[str]:
        if not image_data:
            return "Empty image data"
        if len(image_data) > Config.MAX_IMAGE_SIZE:
            return f"Image too large (max {Config.MAX_IMAGE_SIZE} bytes)"
        try:
            # Без verify(), чтобы не терять метаданные, просто открываем и проверяем размер
            with Image.open(io.BytesIO(image_data)) as im:
                width, height = im.size
                if width * height > Image.MAX_IMAGE_PIXELS:
                    return "Image too large in pixel count"
        except Exception as e:
            # Возвращаем короткое сообщение об ошибке — не включаем бинарные данные
            logger.debug(f"Невалидное изображение: {e}")
            return f"Invalid image data: {str(e)}"
        return None

    def SearchImage(self, request, context):
        if not self.healthy:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Service is not healthy")
            return self._create_search_response(False, "Service is not healthy")

        total_start = time.time()
        try:
            validation_error = self._validate_image_data(request.image_data)
            if validation_error:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(validation_error)
                return self._create_search_response(False, validation_error)

            embed_start = time.time()
            query_embedding = self._embed_image(request.image_data)
            embed_time = time.time() - embed_start

            search_start = time.time()
            # top_k из запроса (если задано > 0), иначе дефолт
            try:
                top_k = int(request.top_k) if request.top_k > 0 else Config.DEFAULT_TOP_K
            except Exception:
                top_k = Config.DEFAULT_TOP_K

            oversample = 10
            # корректируем параметр search_k
            search_k = min(self.index.ntotal, max(Config.SEARCH_K, top_k * oversample))
            # faiss.search ожидает массив shape (n_queries, dim) и возвращает (distances, indices)
            distances, indices = self.index.search(query_embedding, search_k)
            search_time = time.time() - search_start

            processing_start = time.time()
            response = self._create_search_response()
            seen_products = set()
            results_count = 0

            # Проходим результаты и формируем ответ (без логирования двоичных данных)
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                if idx >= len(self.meta):
                    continue
                if results_count >= top_k:
                    break

                item = self.meta[idx]
                product_id = item.get("product_id")
                if product_id in seen_products:
                    continue
                seen_products.add(product_id)

                image_url = self.get_product_image(product_id)

                result = response.results.add()
                result.product_id = product_id
                result.category_id = item.get("category_id", "")
                # В зависимости от метрики индекса dist может означать similarity (IP) или расстояние;
                # оставляем текущую логику, как было — не меняем формат респонса.
                result.similarity = float(dist)
                result.distance = float(1.0 - dist)
                result.product_url = f"https://petrovich.ru/product/{product_id}/"
                result.category_url = f"https://petrovich.ru/catalog/{item.get('category_id','')}/"
                if image_url:
                    result.image_url = image_url

                results_count += 1

            processing_time = time.time() - processing_start
            total_time = time.time() - total_start

            # Заполняем времена (округлённо)
            response.processing_time.embedding_seconds = round(embed_time, 3)
            response.processing_time.search_seconds = round(search_time + processing_time, 3)
            response.processing_time.total_seconds = round(total_time, 3)

            logger.info(f"🔍 Поиск завершен: {len(response.results)} результатов за {total_time:.3f}с")
            return response
        except Exception as e:
            total_time = time.time() - total_start
            # Логируем исключение текстом (без двоичных данных)
            logger.exception(f"Ошибка поиска: {e} (время: {total_time:.3f}с)")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return self._create_search_response(False, str(e))

    def HealthCheck(self, request, context):
        response = image_search_pb2.HealthCheckResponse()

        # Проверяем статусы компонентов
        checks = {
            "model_loaded": self._model is not None,
            "index_loaded": self._index is not None,
            "meta_loaded": self._meta is not None,
            "csv_loaded": self._product_images_dict is not None,
        }

        # Определяем общий статус здоровья
        all_healthy = all(checks.values())
        response.healthy = all_healthy
        response.status = "healthy" if all_healthy else "unhealthy"

        checks["gpu_available"] = torch.cuda.is_available()

        # Дополнительная информация в лог для отладки
        if not all_healthy:
            failed_checks = [name for name, status in checks.items() if not status]
            logger.warning(f"Health check failed for: {', '.join(failed_checks)}")

        return response

    def graceful_shutdown(self, server, timeout: int = 5) -> None:
        logger.info("🛑 Начинаем graceful shutdown...")
        try:
            ev = server.stop(timeout)
            # server.stop возвращает объект с wait()
            if ev is not None:
                try:
                    ev.wait(timeout)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Ошибка при остановке сервера: {e}")

        # Освобождаем ресурсы модели
        try:
            if hasattr(self, '_model'):
                del self._model
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("✅ Ресурсы освобождены, сервер остановлен")


def serve():
    try:
        csv_path = 'feed.csv'
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=Config.MAX_WORKERS),
            options=[
                ('grpc.max_send_message_length', 16 * 1024 * 1024),
                ('grpc.max_receive_message_length', 16 * 1024 * 1024),
            ],
        )

        search_service = ImageSearchService(csv_path=csv_path)

        image_search_pb2_grpc.add_ImageSearchServiceServicer_to_server(search_service, server)
        health_servicer = HealthServicer(search_service)
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

        port = Config.PORT
        server.add_insecure_port(f'[::]:{port}')
        server.start()

        logger.info(f"🚀 gRPC сервер запущен на порту {port}")
        logger.info(f"📊 Статус сервиса: {'healthy' if search_service.healthy else 'unhealthy'}")
        logger.info("📡 Ожидание gRPC запросов...")

        def _graceful_shutdown(sig, frame):
            # вызываем метод сервиса, чтобы освободить ресурсы
            search_service.graceful_shutdown(server, timeout=5)
            sys.exit(0)

        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

        server.wait_for_termination()
    except Exception as e:
        logger.exception(f"❌ Фатальная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    serve()
