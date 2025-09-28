from flask import Flask, request, jsonify
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch
from collections import Counter
import logging

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загружаем модели и индексы
def load_models_and_indexes():
    """Загружаем все модели и соответствующие индексы"""

    # Модели
    e5_model = SentenceTransformer("intfloat/multilingual-e5-large")
    sber_model = SentenceTransformer("sberbank-ai/sbert_large_nlu_ru")
    baai_model = SentenceTransformer("BAAI/bge-m3")

    # Индексы FAISS
    e5_index = faiss.read_index("faiss_index/products_index.index")
    sber_index = faiss.read_index("faiss_index_sber/products_index.index")
    baai_index = faiss.read_index("faiss_index_baai/products_index.index")

    # Метаданные
    with open("faiss_index/products_index_metadata.pkl", 'rb') as f:
        e5_metadata = pickle.load(f)

    with open("faiss_index_sber/products_index_metadata.pkl", 'rb') as f:
        sber_metadata = pickle.load(f)

    with open("faiss_index_baai/products_index_metadata.pkl", 'rb') as f:
        baai_metadata = pickle.load(f)

    return {
        'e5': {
            'model': e5_model,
            'index': e5_index,
            'metadata': e5_metadata
        },
        'sber': {
            'model': sber_model,
            'index': sber_index,
            'metadata': sber_metadata
        },
        'baai': {
            'model': baai_model,
            'index': baai_index,
            'metadata': baai_metadata
        }
    }

# Глобальные переменные с моделями и индексами
models_data = load_models_and_indexes()

def search_in_model(query, model_type, k=9):
    """Поиск в конкретной модели"""
    model_data = models_data[model_type]

    # Подготавливаем запрос для модели
    if model_type == 'e5':
        query_text = f"query: {query}"
    else:
        query_text = query

    # Создаем эмбеддинг запроса
    query_embedding = model_data['model'].encode([query_text]).astype('float32')

    # Ищем в индексе
    distances, indices = model_data['index'].search(query_embedding, k)

    # Собираем результаты
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(model_data['metadata']['metadata']):
            product_data = model_data['metadata']['metadata'][idx]
            results.append({
                'product_id': product_data['product_id'],
                'product_name': product_data['product_name'],
                'category_name': product_data['category_name'],
                'image_url': product_data.get('image_url', ''),
                'score': float(distances[0][i]),
                'model_type': model_type
            })

    return results

def get_top_categories(results, top_k=5):
    """Получаем топ категорий из результатов поиска"""
    categories = [result['category_name'] for result in results if result['category_name']]
    category_counts = Counter(categories)

    # Сортируем по частоте и берем топ
    top_categories = category_counts.most_common(top_k)

    return [{'category': cat, 'count': count} for cat, count in top_categories]

def generate_html(results=None, query='', error=None):
    """Генерируем HTML страницу"""
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Поиск товаров</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .search-container {{ margin-bottom: 30px; }}
            .search-input {{ width: 100%; padding: 12px; font-size: 16px; margin-bottom: 10px; }}
            .search-button {{ padding: 12px 24px; font-size: 16px; background: #007bff; color: white; border: none; cursor: pointer; }}
            .results-container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .model-results {{ flex: 1; min-width: 300px; }}
            .model-title {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
            .product-card {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }}
            .product-score {{ color: #666; font-size: 12px; }}
            .categories-list {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            .category-item {{ margin-bottom: 5px; }}
            .error {{ color: red; padding: 10px; background: #ffe6e6; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="search-container">
            <h1>Поиск товаров</h1>
            <form action="/search" method="get">
                <input type="text" name="q" class="search-input" placeholder="Введите поисковый запрос..." value="{query}">
                <button type="submit" class="search-button">Найти</button>
            </form>
        </div>
    '''

    if error:
        html += f'<div class="error">Ошибка: {error}</div>'

    if results:
        html += '<div class="results-container">'

        # E5 модель
        html += '''
            <div class="model-results">
                <div class="model-title" style="border-left: 4px solid #007bff;">
                    <h2>Multilingual E5 Large</h2>
                    <small>Найдено: {} товаров</small>
                </div>
        '''.format(len(results['e5']['results']))

        for product in results['e5']['results']:
            html += f'''
                <div class="product-card">
                    <h3>{product['product_name']}</h3>
                    <p><strong>Категория:</strong> {product['category_name']}</p>
                    <p class="product-score">Score: {product['score']:.4f}</p>
            '''
            if product['image_url']:
                html += f'<img src="{product["image_url"]}" alt="{product["product_name"]}" style="max-width: 200px; max-height: 150px;">'
            html += '</div>'

        html += '<div class="categories-list"><h4>Топ категорий (E5):</h4>'
        for cat in results['e5']['top_categories']:
            html += f'<div class="category-item">{cat["category"]} ({cat["count"]})</div>'
        html += '</div></div>'

        # Sber модель
        html += '''
            <div class="model-results">
                <div class="model-title" style="border-left: 4px solid #28a745;">
                    <h2>Sberbank Russian Model</h2>
                    <small>Найдено: {} товаров</small>
                </div>
        '''.format(len(results['sber']['results']))

        for product in results['sber']['results']:
            html += f'''
                <div class="product-card">
                    <h3>{product['product_name']}</h3>
                    <p><strong>Категория:</strong> {product['category_name']}</p>
                    <p class="product-score">Score: {product['score']:.4f}</p>
            '''
            if product['image_url']:
                html += f'<img src="{product["image_url"]}" alt="{product["product_name"]}" style="max-width: 200px; max-height: 150px;">'
            html += '</div>'

        html += '<div class="categories-list"><h4>Топ категорий (Sber):</h4>'
        for cat in results['sber']['top_categories']:
            html += f'<div class="category-item">{cat["category"]} ({cat["count"]})</div>'
        html += '</div></div>'

        # BAAI модель
        html += '''
            <div class="model-results">
                <div class="model-title" style="border-left: 4px solid #dc3545;">
                    <h2>BAAI BGE-M3 Model</h2>
                    <small>Найдено: {} товаров</small>
                </div>
        '''.format(len(results['baai']['results']))

        for product in results['baai']['results']:
            html += f'''
                <div class="product-card">
                    <h3>{product['product_name']}</h3>
                    <p><strong>Категория:</strong> {product['category_name']}</p>
                    <p class="product-score">Score: {product['score']:.4f}</p>
            '''
            if product['image_url']:
                html += f'<img src="{product["image_url"]}" alt="{product["product_name"]}" style="max-width: 200px; max-height: 150px;">'
            html += '</div>'

        html += '<div class="categories-list"><h4>Топ категорий (BAAI):</h4>'
        for cat in results['baai']['top_categories']:
            html += f'<div class="category-item">{cat["category"]} ({cat["count"]})</div>'
        html += '</div></div>'

        html += '</div>'  # закрываем results-container

    html += '</body></html>'
    return html

@app.route('/')
def index():
    """Главная страница с поиском"""
    return generate_html()

@app.route('/search')
def search():
    """Обработка поискового запроса"""
    query = request.args.get('q', '').strip()

    if not query:
        return generate_html()

    logger.info(f"Поисковый запрос: {query}")

    try:
        # Ищем во всех моделях
        e5_results = search_in_model(query, 'e5', k=9)
        sber_results = search_in_model(query, 'sber', k=9)
        baai_results = search_in_model(query, 'baai', k=9)

        # Получаем топ категории для каждой модели
        e5_categories = get_top_categories(e5_results, 5)
        sber_categories = get_top_categories(sber_results, 5)
        baai_categories = get_top_categories(baai_results, 5)

        results = {
            'e5': {
                'results': e5_results,
                'top_categories': e5_categories
            },
            'sber': {
                'results': sber_results,
                'top_categories': sber_categories
            },
            'baai': {
                'results': baai_results,
                'top_categories': baai_categories
            }
        }

        return generate_html(results=results, query=query)

    except Exception as e:
        logger.error(f"Ошибка при поиске: {e}")
        return generate_html(error=str(e), query=query)

@app.route('/api/search')
def api_search():
    """JSON API для поиска"""
    query = request.args.get('q', '').strip()

    if not query:
        return jsonify({'error': 'Пустой запрос'}), 400

    try:
        # Ищем во всех моделях
        e5_results = search_in_model(query, 'e5', k=9)
        sber_results = search_in_model(query, 'sber', k=9)
        baai_results = search_in_model(query, 'baai', k=9)

        # Получаем топ категории для каждой модели
        e5_categories = get_top_categories(e5_results, 5)
        sber_categories = get_top_categories(sber_results, 5)
        baai_categories = get_top_categories(baai_results, 5)

        response = {
            'query': query,
            'results': {
                'e5_model': {
                    'products': e5_results,
                    'top_categories': e5_categories
                },
                'sber_model': {
                    'products': sber_results,
                    'top_categories': sber_categories
                },
                'baai_model': {
                    'products': baai_results,
                    'top_categories': baai_categories
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Ошибка при API поиске: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Запуск веб-сервиса...")
    app.run(host='0.0.0.0', port=5000, debug=True)