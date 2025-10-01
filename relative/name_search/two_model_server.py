from flask import Flask, request, jsonify
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
# CrossEncoder опционально — если установлен, он будет использован
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

import torch
from collections import Counter
import logging
import concurrent.futures
import json
import os
import re
from typing import List, Dict
import html as html_module  # для безопасного экранирования

# ========== Конфигурация ==========
TOP_K = 9                      # сколько возвращать пользователю для каждой модели
RAW_K = 50                     # сколько извлекать из FAISS перед перерангировкой
RERANK_TOP_N = 30              # сколько кандидатов отправлять в cross-encoder (по модели)
ALPHA_SEMANTIC = 0.7           # вес семантики (модель/FAISS)
BETA_KEYWORD = 0.3             # вес ключевых слов
GAMMA_CE = 0.8                 # если есть cross-encoder — его доля в финальном скоре
STRICT_FILTER = False          # если True — жёстко фильтруем по token match (удаляем "левые")
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SYNONYMS_FILE = "synonyms_optimized.json"
# =================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("product_search")

app = Flask(__name__)

models_data = None  # загрузится при старте

# ---------- вспомогательные ----------
def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r'\w+', text.lower())
    tokens = [t for t in tokens if len(t) > 1]  # пропускаем односимвольные токены
    return tokens

def keyword_score(query_tokens: List[str], product: Dict) -> float:
    if not query_tokens:
        return 0.0
    name = product.get('product_name', '') or ''
    category = product.get('category_name', '') or ''
    desc = product.get('description', '') or ''
    text_tokens = set(tokenize(' '.join([name, category, desc])))
    if not text_tokens:
        return 0.0
    matched = sum(1 for t in set(query_tokens) if t in text_tokens)
    return matched / len(set(query_tokens))

def normalize_list(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return [0.5 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]

# ---------- Synonyms (если нужен) ----------
class SynonymSearcher:
    def __init__(self, synonyms_dict):
        self.synonyms_dict = synonyms_dict

    def get_synonyms(self, word):
        return self.synonyms_dict.get(word.lower(), [word])

    def expand_query(self, query, max_variants=3):
        words = query.lower().split()
        expanded = [query]
        for word in words:
            syns = self.get_synonyms(word)
            for s in syns:
                if s != word and len(expanded) < max_variants:
                    newq = query.replace(word, s)
                    if newq not in expanded:
                        expanded.append(newq)
        return expanded[:max_variants]

def load_synonyms(path=SYNONYMS_FILE):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            logger.info(f"Loaded {len(d)} synonyms")
            return SynonymSearcher(d)
        else:
            logger.warning("Synonyms file not found, skipping")
            return None
    except Exception as e:
        logger.error("Error loading synonyms: %s", e)
        return None

# ---------- загрузка моделей и индексов ----------
def load_models_and_indexes():
    logger.info("Loading models and FAISS indexes...")

    logger.info("Loading E5 model...")
    e5_model = SentenceTransformer("intfloat/multilingual-e5-large")
    logger.info("Loading Sber model...")
    sber_model = SentenceTransformer("sberbank-ai/sbert_large_nlu_ru")
    logger.info("Loading BAAI model...")
    baai_model = SentenceTransformer("BAAI/bge-m3")

    logger.info("Reading FAISS indexes...")
    e5_index = faiss.read_index("faiss_index/products_index.index")
    sber_index = faiss.read_index("faiss_index_sber/products_index.index")
    baai_index = faiss.read_index("faiss_index_baai/products_index.index")

    logger.info("Loading metadata pickles...")
    with open("faiss_index/products_index_metadata.pkl", 'rb') as f:
        e5_meta = pickle.load(f)
    with open("faiss_index_sber/products_index_metadata.pkl", 'rb') as f:
        sber_meta = pickle.load(f)
    with open("faiss_index_baai/products_index_metadata.pkl", 'rb') as f:
        baai_meta = pickle.load(f)

    synonyms = load_synonyms()

    cross_encoder = None
    if CROSS_ENCODER_AVAILABLE:
        try:
            logger.info(f"Loading CrossEncoder {RERANK_MODEL_NAME} ...")
            cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
            logger.info("CrossEncoder loaded")
        except Exception as e:
            logger.warning("Could not load CrossEncoder: %s", e)
            cross_encoder = None
    else:
        logger.info("CrossEncoder not available (not installed)")

    logger.info("All models loaded")
    return {
        'e5': {'model': e5_model, 'index': e5_index, 'metadata': e5_meta},
        'sber': {'model': sber_model, 'index': sber_index, 'metadata': sber_meta},
        'baai': {'model': baai_model, 'index': baai_index, 'metadata': baai_meta},
        'synonyms': synonyms,
        'cross_encoder': cross_encoder
    }

# ---------- поиск в модели (FAISS) ----------
def search_in_model_raw(query: str, model_type: str, k: int = RAW_K) -> List[Dict]:
    md = models_data[model_type]
    model = md['model']
    idx = md['index']
    # e5 special prompt
    qtext = f"query: {query}" if model_type == 'e5' else query

    emb = model.encode([qtext], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    distances, indices = idx.search(emb, k)

    results = []
    for pos, idx_val in enumerate(indices[0]):
        if idx_val < len(md['metadata']['metadata']):
            m = md['metadata']['metadata'][idx_val]
            results.append({
                'product_id': m.get('product_id'),
                'product_name': m.get('product_name', '') if isinstance(m, dict) else str(m),
                'category_name': m.get('category_name', '') if isinstance(m, dict) else '',
                'image_url': m.get('image_url', '') if isinstance(m, dict) else '',
                'description': m.get('description', '') if isinstance(m, dict) else '',
                'raw_distance': float(distances[0][pos]) if distances is not None else None,
                'model_rank': pos,  # 0..k-1
                'model_type': model_type
            })
    return results

# ---------- rerank per model (keyword + semantic + optional cross-encoder) ----------
def rerank_per_model(candidates: List[Dict], query: str, model_type: str, top_k: int = TOP_K, use_cross_encoder: bool = True) -> List[Dict]:
    if not candidates:
        return []

    q_tokens = tokenize(query)

    # strict filter (если включена) — оставляем только продукты, содержащие хотя бы один токен запроса в name или category
    if STRICT_FILTER and q_tokens:
        filtered = []
        for c in candidates:
            text = ' '.join([c.get('product_name', ''), c.get('category_name', '')])
            text_tokens = set(tokenize(text))
            if any(t in text_tokens for t in q_tokens):
                filtered.append(c)
        logger.info(f"[{model_type}] STRICT_FILTER enabled: {len(candidates)} -> {len(filtered)}")
        candidates = filtered
        if not candidates:
            return []

    # semantic score на основе model_rank (чем ближе к 0 — тем лучше)
    ranks = [c.get('model_rank', 9999) for c in candidates]
    max_rank = max(ranks) if ranks else 0
    if max_rank == 0:
        semantic_scores = [1.0 for _ in ranks]
    else:
        # 1 - (rank/max_rank) -> 1..0
        semantic_scores = [1.0 - (r / max_rank) for r in ranks]

    # keyword scores
    kw_scores = [keyword_score(q_tokens, c) for c in candidates]

    # normalize
    semantic_norm = normalize_list(semantic_scores)
    kw_norm = normalize_list(kw_scores)

    # combine
    combined_scores = []
    for i, c in enumerate(candidates):
        combined = ALPHA_SEMANTIC * semantic_norm[i] + BETA_KEYWORD * kw_norm[i]
        combined_scores.append(combined)
        c['semantic_score'] = semantic_norm[i]
        c['keyword_score'] = kw_norm[i]
        c['combined_score'] = combined

    # attach combined score and sort descending
    for i, c in enumerate(candidates):
        c['_combined_raw'] = combined_scores[i]

    candidates_sorted = sorted(candidates, key=lambda x: x['_combined_raw'], reverse=True)

    # Optionally rerank top N with cross-encoder (если доступен и включён)
    ce = models_data.get('cross_encoder')
    if use_cross_encoder and ce is not None:
        top_n = min(RERANK_TOP_N, len(candidates_sorted))
        to_rerank = candidates_sorted[:top_n]
        pairs = []
        for c in to_rerank:
            # build text for candidate (name + category + short description)
            txt = ' '.join([c.get('product_name', ''), c.get('category_name', ''), c.get('description', '')])
            pairs.append((query, txt))
        try:
            ce_scores = ce.predict(pairs)  # higher = more relevant
            ce_norm = normalize_list(list(ce_scores))
            # combine CE score with previous combined
            for i, c in enumerate(to_rerank):
                prev = c['_combined_raw']
                new_final = GAMMA_CE * ce_norm[i] + (1.0 - GAMMA_CE) * prev
                c['ce_score'] = float(ce_scores[i])
                c['ce_score_norm'] = ce_norm[i]
                c['final_score'] = new_final
            # replace top_n with ce-reordered by final_score
            to_rerank = sorted(to_rerank, key=lambda x: x['final_score'], reverse=True)
            candidates_sorted = to_rerank + candidates_sorted[top_n:]
        except Exception as e:
            logger.warning(f"[{model_type}] Cross-encoder failed: {e}")
            # в случае ошибки оставляем старую сортировку
    else:
        # если CE не используется — final_score = combined_score (normalized)
        combined_vals = [c['_combined_raw'] for c in candidates_sorted]
        comb_norm = normalize_list(combined_vals)
        for i, c in enumerate(candidates_sorted):
            c['final_score'] = comb_norm[i]

    # обрезаем до top_k и убираем промежуточные поля которые не нужны в ответе
    final = candidates_sorted[:top_k]
    for c in final:
        # гарантируем numeric поля корректно сериализуемы
        c['final_score'] = float(c.get('final_score', 0.0))
        c['semantic_score'] = float(c.get('semantic_score', 0.0))
        c['keyword_score'] = float(c.get('keyword_score', 0.0))
        # удаляем внутренние
        c.pop('_combined_raw', None)
        c.pop('model_rank', None)
        c.pop('raw_distance', None)
    logger.info(f"[{model_type}] returning {len(final)} items (from {len(candidates)})")
    return final

# ---------- public search per model ----------
def search_model_pipeline(query: str, model_type: str, use_synonyms: bool = False, max_variants: int = 3, use_ce: bool = True):
    # If synonyms are used, expand (but we will only run FAISS on original + first variant for speed)
    query_variants = [query]
    if use_synonyms and models_data['synonyms']:
        try:
            query_variants = models_data['synonyms'].expand_query(query, max_variants=max_variants)
        except Exception as e:
            logger.warning("Synonym expansion failed: %s", e)
            query_variants = [query]

    # We'll run raw search for each variant and merge candidates (but keep separate model processing)
    all_candidates = []
    for qv in query_variants:
        try:
            raw = search_in_model_raw(qv, model_type, k=RAW_K)
            # mark which variant found it
            for r in raw:
                r.setdefault('found_by_variant', qv)
            all_candidates.extend(raw)
        except Exception as e:
            logger.error(f"Search in {model_type} failed for variant '{qv}': {e}")
            continue

    # deduplicate by product_id, keep best model_rank if multiple variants hit
    dedup = {}
    for cand in all_candidates:
        pid = cand.get('product_id')
        if pid is None:
            continue
        if pid not in dedup:
            dedup[pid] = cand
        else:
            # keep one with best model_rank (smaller)
            if cand.get('model_rank', 9999) < dedup[pid].get('model_rank', 9999):
                dedup[pid] = cand

    candidates = list(dedup.values())

    # rerank per model
    final = rerank_per_model(candidates, query, model_type, top_k=TOP_K, use_cross_encoder=use_ce)
    return final

# ---------- УСТОЙЧИВЫЙ генератор HTML (исправленный версткой) ----------
def generate_html(results=None, query='', error=None, used_synonyms=None, use_synonyms=True, use_ce=True):
    """
    Рендерит страницу с результатами по трём моделям.
    Безопасно экранирует текст, показывает плейсхолдер для изображений,
    делает адаптивные карточки и корректно обрабатывает пустые поля.
    """
    query_esc = html_module.escape(query or '')
    checkbox_syn = "checked" if use_synonyms else ""
    checkbox_ce = "checked" if use_ce else ""

    # Начало HTML + стиль
    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Поиск товаров — результаты</title>
<style>
  body {{ font-family: Inter, Arial, sans-serif; margin: 16px; color: #222; background: #fafafa; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
  h1 {{ margin:0; font-size:20px; }}
  form.search {{ width:100%; display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }}
  input[type="text"] {{ flex:1 1 320px; padding:10px 12px; border-radius:8px; border:1px solid #ddd; font-size:14px; }}
  .controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  button.submit {{ background:#007bff; color:white; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; }}
  .models-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin-top:16px; }}
  .model-card {{ background:white; border:1px solid #e6e9ee; border-radius:10px; padding:12px; box-shadow: 0 1px 2px rgba(16,24,40,0.03); }}
  .model-card h2 {{ margin:0 0 8px 0; font-size:16px; }}
  .products-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }}
  .product {{ border:1px solid #eee; border-radius:8px; padding:10px; background:#fff; display:flex; gap:10px; align-items:flex-start; }}
  .product .thumb {{ width:96px; height:72px; flex-shrink:0; background:#f0f0f0; display:flex; align-items:center; justify-content:center; border-radius:6px; overflow:hidden; }}
  .product img {{ max-width:100%; max-height:100%; display:block; object-fit:cover; }}
  .product .meta {{ flex:1; min-width:0; }}
  .product .title {{ font-weight:600; font-size:14px; color:#0f1724; margin-bottom:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .product .cat {{ font-size:12px; color:#6b7280; margin-bottom:6px; }}
  .product .scores {{ font-size:12px; color:#475569; display:flex; gap:8px; flex-wrap:wrap; }}
  .empty {{ padding:12px; color:#6b7280; background:#fff; border-radius:8px; border:1px dashed #e6e9ee; }}
  footer {{ margin-top:18px; font-size:13px; color:#6b7280; }}
  @media (max-width:520px) {{
    .product {{ flex-direction:column; align-items:flex-start; }}
    .product .thumb {{ width:100%; height:160px; }}
  }}
</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Поиск товаров</h1>
      <div>Запрос: <strong>{query_esc}</strong></div>
    </header>

    <form class="search" action="/search" method="get" autocomplete="off">
      <input type="text" name="q" placeholder="Введите запрос..." value="{query_esc}">
      <div class="controls">
        <label><input type="checkbox" name="use_synonyms" value="true" {checkbox_syn}> синонимы</label>
        <label><input type="checkbox" name="use_ce" value="true" {checkbox_ce}> cross-encoder</label>
        <button type="submit" class="submit">Найти</button>
      </div>
    </form>
'''

    if error:
        html += f'<div style="color:#b91c1c; margin-top:8px;">Ошибка: {html_module.escape(str(error))}</div>'

    if used_synonyms and len(used_synonyms) > 1:
        used_html = ', '.join(html_module.escape(u) for u in used_synonyms)
        html += f'<div style="background:#eef8ff;padding:8px;border-left:4px solid #007bff;border-radius:6px;margin-top:8px;">Варианты запроса: {used_html}</div>'

    # Рендер моделей
    html += '<div class="models-grid">'

    models_order = ['e5', 'sber', 'baai']
    model_names = {'e5': 'Multilingual E5 Large', 'sber': 'Sber RBERT', 'baai': 'BAAI BGE-M3'}

    for mk in models_order:
        model_res = (results or {}).get(mk, [])
        safe_title = html_module.escape(model_names.get(mk, mk.upper()))
        html += f'<div class="model-card"><h2>{safe_title} — найдено: {len(model_res)}</h2>'

        if not model_res:
            html += '<div class="empty">Нет результатов</div></div>'
            continue

        html += '<div class="products-grid">'
        for p in model_res:
            pname = html_module.escape(str(p.get('product_name') or '—'))
            pcate = html_module.escape(str(p.get('category_name') or '—'))
            img = html_module.escape(str(p.get('image_url') or ''))
            found_by = html_module.escape(str(p.get('found_by_variant') or ''))
            # безопасное форматирование чисел
            final_score = p.get('final_score', 0.0)
            semantic = p.get('semantic_score', 0.0)
            keyword = p.get('keyword_score', 0.0)
            try:
                fs_txt = f"{float(final_score):.4f}"
            except Exception:
                fs_txt = str(final_score)
            try:
                sem_txt = f"{float(semantic):.3f}"
            except Exception:
                sem_txt = str(semantic)
            try:
                kw_txt = f"{float(keyword):.3f}"
            except Exception:
                kw_txt = str(keyword)

            thumb_html = f'<div class="thumb"><img src="{img}" alt=""></div>' if img else '<div class="thumb">—</div>'

            html += f'''
            <div class="product">
              {thumb_html}
              <div class="meta">
                <div class="title" title="{pname}">{pname}</div>
                <div class="cat">{pcate}</div>
                <div class="scores">
                  <div>final: {fs_txt}</div>
                  <div>sem: {sem_txt}</div>
                  <div>kw: {kw_txt}</div>
                  <div style="color:#9ca3af;">variant: {found_by}</div>
                </div>
              </div>
            </div>
            '''

        html += '</div></div>'  # products-grid, model-card

    html += '</div>'  # models-grid

    html += '''
    <footer>
      Подсказка: если результаты «ушли» в другие категории, попробуй включить строгую фильтрацию (STRICT_FILTER=True в коде) либо увеличить значение RAW_K.
    </footer>
  </div>
</body>
</html>
'''
    return html

# ========== Flask endpoints ==========
@app.route('/')
def index():
    return generate_html(use_synonyms=False, query='')

@app.route('/search')
def search_page():
    query = request.args.get('q', '').strip()
    use_synonyms = 'use_synonyms' in request.args and request.args.get('use_synonyms', 'false').lower() == 'true'
    use_ce = 'use_ce' in request.args and request.args.get('use_ce', 'false').lower() == 'true'

    if not query:
        return generate_html(use_synonyms=use_synonyms, query=query, use_ce=use_ce)

    if models_data is None:
        return generate_html(error="Модели ещё загружаются, попробуйте позже", query=query, use_synonyms=use_synonyms, use_ce=use_ce)

    used_variants = [query]
    if use_synonyms and models_data['synonyms']:
        used_variants = models_data['synonyms'].expand_query(query, max_variants=5)

    results = {}
    # Для сравнения — запускаем каждую модель отдельно
    for m in ['e5', 'sber', 'baai']:
        try:
            res = search_model_pipeline(query, m, use_synonyms=use_synonyms, max_variants=5, use_ce=use_ce)
            # ensure 'found_by_variant' exists on each result for display (may be missing)
            for r in res:
                if 'found_by_variant' not in r:
                    r['found_by_variant'] = ''
            results[m] = res
        except Exception as e:
            logger.error("Search pipeline failed for %s: %s", m, e)
            results[m] = []

    return generate_html(results=results, query=query, used_synonyms=used_variants if use_synonyms else None, use_synonyms=use_synonyms, use_ce=use_ce)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').strip()
    use_synonyms = 'use_synonyms' in request.args and request.args.get('use_synonyms', 'false').lower() == 'true'
    use_ce = 'use_ce' in request.args and request.args.get('use_ce', 'false').lower() == 'true'
    max_variants = int(request.args.get('max_variants', 3))

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    if models_data is None:
        return jsonify({'error': 'Models not loaded yet'}), 503

    response = {'query': query, 'use_synonyms': use_synonyms, 'models': {}}
    for m in ['e5', 'sber', 'baai']:
        try:
            res = search_model_pipeline(query, m, use_synonyms=use_synonyms, max_variants=max_variants, use_ce=use_ce)
            kw_matches = sum(1 for r in res if r.get('keyword_score', 0) > 0)
            response['models'][m] = {
                'products': res,
                'keyword_matches_count': kw_matches,
                'returned': len(res)
            }
        except Exception as e:
            logger.error("API search failed for %s: %s", m, e)
            response['models'][m] = {'error': str(e)}

    return jsonify(response)

@app.route('/api/synonyms/<word>')
def api_synonyms(word):
    if not word:
        return jsonify({'error': 'Empty word'}), 400
    if models_data is None:
        return jsonify({'error': 'Models not loaded'}), 503
    if models_data['synonyms']:
        syns = models_data['synonyms'].get_synonyms(word)
        return jsonify({'word': word, 'synonyms': syns})
    else:
        return jsonify({'word': word, 'synonyms': [word], 'warning': 'Synonyms not loaded'})

# ========== main ==========
if __name__ == '__main__':
    logger.info("Starting service and loading models...")
    models_data = load_models_and_indexes()
    app.run(host='0.0.0.0', port=8080, debug=False)
