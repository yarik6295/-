"""
Handler Locator — локальный поиск МЕСТА в коде, где определён конкретный
обработчик/функция/маршрут/стиль, полностью на вашем компьютере, без
платных API.

Стек:
  - rank_bm25                — лексический поиск (без ML, легковесный)
  - sentence-transformers     — эмбеддинги с поддержкой русского языка
  - ChromaDB (in-memory)      — векторная база данных
  - Ollama                    — локальная LLM для подтверждения находки
  - Streamlit                 — графический интерфейс в браузере

Установка:
    pip install streamlit sentence-transformers chromadb requests rank_bm25

Ollama (опционально, для объяснений находок):
    1. Скачать: https://ollama.com/download
    2. В терминале: ollama pull qwen2.5-coder
    3. Ollama сама поднимает локальный сервер на http://localhost:11434

Запуск:
    streamlit run code_rag.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import requests
import streamlit as st

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Не найдена библиотека 'sentence-transformers'.\n\nУстановите: `pip install sentence-transformers`")
    st.stop()

try:
    import chromadb
except ImportError:
    st.error("Не найдена библиотека 'chromadb'.\n\nУстановите: `pip install chromadb`")
    st.stop()

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("Не найдена библиотека 'rank_bm25'.\n\nУстановите: `pip install rank_bm25`")
    st.stop()


# ============================================================================
# Конфигурация
# ============================================================================

CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".html", ".css"}

EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".idea", ".vscode", "target", "env",
}

MAX_SYMBOL_LINES = 60      # сколько строк тела символа сохраняем максимум
MAX_SYMBOL_CHARS = 2500    # и сколько символов текста максимум

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 8          # сколько финальных кандидатов уходит в LLM
CANDIDATE_POOL_SIZE = 25   # сколько кандидатов берём из КАЖДОГО из двух поисков перед слиянием
RRF_K = 60                 # константа Reciprocal Rank Fusion (стандартное значение)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TIMEOUT_SECONDS = 120

FALLBACK_MODEL_NAME = "llama3.2"
PREFERRED_MODELS = [
    "qwen2.5-coder", "qwen2.5-coder:latest",
    "qwen3-coder", "qwen3-coder:30b", "qwen3-coder:latest",
    "codellama", "codellama:latest",
]

# Ключевые слова языков, которые regex для "метод-шортхенд" (`name(...) {`)
# не должен принимать за имя обработчика/функции.
JS_CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "function", "return",
    "else", "try", "do", "with",
}


# ============================================================================
# Модель данных
# ============================================================================

@dataclass
class CodeSymbol:
    """Один найденный в коде символ: функция, метод, класс, роут,
    обработчик события или CSS-правило — с ТОЧНЫМИ границами в файле."""
    id: str
    file_path: str
    name: str
    kind: str          # function | method | class | route | listener | handler-ref | element(<tag>) | inline-handler | style
    signature: str
    start_line: int
    end_line: int
    text: str

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def searchable_text(self) -> str:
        """Текст, который уходит и в BM25, и в эмбеддинги: имя важнее всего,
        поэтому оно повторяется, дальше сигнатура и превью тела."""
        return f"{self.kind} {self.name} {self.name}\n{self.signature}\n{self.text[:400]}"


class AnalysisError(Exception):
    """Сбой самого запроса к Ollama (сеть, парсинг JSON и т.п.) —
    не путать с тем, что модель честно не нашла подходящего символа."""


# ============================================================================
# Шаг 1: сканирование проекта
# ============================================================================

def collect_source_files(root: Path) -> list[Path]:
    """Рекурсивно находит файлы с кодом, пропуская служебные папки."""
    return [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix in CODE_EXTENSIONS
        and not any(part in EXCLUDED_DIRS for part in path.parts)
    ]


def read_file_safely(path: Path) -> str | None:
    """Устойчивое к битым кодировкам и правам доступа чтение файла."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    except (OSError, PermissionError):
        return None


# ============================================================================
# Шаг 2: разбор символов (функции / методы / классы / обработчики / роуты)
# ============================================================================

def find_indented_block_end(lines: list[str], start_idx: int, indent: int) -> int:
    """Для Python: конец блока — первая непустая строка с отступом <= indent."""
    end = start_idx
    for j in range(start_idx + 1, min(len(lines), start_idx + 400)):
        line = lines[j]
        if not line.strip():
            end = j
            continue
        if (len(line) - len(line.lstrip())) <= indent:
            break
        end = j
    return end


def find_brace_block_end(lines: list[str], start_idx: int) -> int:
    """Для C-подобного синтаксиса: конец блока — строка, где счётчик
    фигурных скобок впервые возвращается к нулю после открытия."""
    depth = 0
    opened = False
    for j in range(start_idx, min(len(lines), start_idx + 400)):
        depth += lines[j].count("{") - lines[j].count("}")
        if "{" in lines[j]:
            opened = True
        if opened and depth <= 0:
            return j
    return min(start_idx + 30, len(lines) - 1)


RawSymbol = tuple[str, str, int, int, str]  # name, kind, start_line(1-idx), end_line(1-idx), signature


def _extract_python(lines: list[str]) -> list[RawSymbol]:
    results: list[RawSymbol] = []
    decorators: list[str] = []
    for i, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if stripped.startswith("@"):
            decorators.append(stripped)
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        m_def = re.match(r"(async\s+)?def\s+(\w+)\s*\(", stripped)
        m_class = re.match(r"class\s+(\w+)", stripped)
        if m_def or m_class:
            name = m_def.group(2) if m_def else m_class.group(1)
            kind = "class" if m_class else ("method" if indent > 0 else "function")
            if any(re.search(r"\.(route|get|post|put|delete|patch)\(", d) for d in decorators):
                kind = "route"
            end = find_indented_block_end(lines, i, indent)
            results.append((name, kind, i + 1, end + 1, stripped))
        decorators = []
    return results


_JS_DEF_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?function\s+(\w+)\s*\("), "function"),
    (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^=]*\)\s*=>"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function"), "function"),
    (re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*\{"), "method"),
]
_JS_ROUTE = re.compile(r"\b(?:app|router)\.(get|post|put|delete|patch)\(\s*['\"`]([^'\"`]+)['\"`]")
_JS_LISTENER = re.compile(r"addEventListener\(\s*['\"`](\w+)['\"`]")
_JSX_HANDLER = re.compile(r"\bon([A-Z]\w*)\s*=\s*\{?\s*(\w+)")


def _extract_js(lines: list[str]) -> list[RawSymbol]:
    results: list[RawSymbol] = []
    for i, line in enumerate(lines):
        matched = False
        for pattern, kind in _JS_DEF_PATTERNS:
            m = pattern.match(line)
            if m:
                name = m.group(1)
                if kind == "method" and name in JS_CONTROL_KEYWORDS:
                    continue
                end = find_brace_block_end(lines, i)
                results.append((name, kind, i + 1, end + 1, line.strip()))
                matched = True
                break
        if matched:
            continue
        m_route = _JS_ROUTE.search(line)
        if m_route:
            results.append((f"{m_route.group(1).upper()} {m_route.group(2)}", "route",
                             i + 1, min(i + 3, len(lines)), line.strip()))
            continue
        m_listener = _JS_LISTENER.search(line)
        if m_listener:
            results.append((m_listener.group(1), "listener", i + 1, min(i + 3, len(lines)), line.strip()))
            continue
        m_jsx = _JSX_HANDLER.search(line)
        if m_jsx:
            results.append((m_jsx.group(2), "handler-ref", i + 1, i + 1, line.strip()))
    return results


_GO_FUNC = re.compile(r"^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(")


def _extract_go(lines: list[str]) -> list[RawSymbol]:
    results: list[RawSymbol] = []
    for i, line in enumerate(lines):
        m = _GO_FUNC.match(line)
        if m:
            end = find_brace_block_end(lines, i)
            results.append((m.group(1), "function", i + 1, end + 1, line.strip()))
    return results


_RUST_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)")


def _extract_rust(lines: list[str]) -> list[RawSymbol]:
    results: list[RawSymbol] = []
    for i, line in enumerate(lines):
        m = _RUST_FN.match(line)
        if m:
            end = find_brace_block_end(lines, i)
            results.append((m.group(1), "function", i + 1, end + 1, line.strip()))
    return results


_HTML_ID = re.compile(r"<(\w+)[^>]*\bid=[\"']([\w\-]+)[\"']")
_HTML_INLINE_HANDLER = re.compile(r"\bon(\w+)=[\"']([^\"']{0,60})")
_SCRIPT_OPEN = re.compile(r"<script\b[^>]*>", re.IGNORECASE)
_SCRIPT_CLOSE = re.compile(r"</script\s*>", re.IGNORECASE)
_SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc=", re.IGNORECASE)


def _extract_html(lines: list[str]) -> list[RawSymbol]:
    """
    HTML часто содержит не только разметку, но и весь JS целиком внутри
    <script>...</script> (типичная однофайловая веб-страница). Если
    сканировать только теги, вся логика приложения (функции,
    addEventListener, обработчики) остаётся невидимой. Поэтому здесь
    отдельно вырезаются JS-блоки и прогоняются через тот же JS-парсер,
    а найденные строки пересчитываются в номера строк исходного файла.
    """
    results: list[RawSymbol] = []
    in_script = False
    script_start_idx = 0
    script_lines: list[str] = []

    def flush_script() -> None:
        if not script_lines:
            return
        for name, kind, s_start, s_end, sig in _extract_js(script_lines):
            # s_start/s_end — 1-индексные номера СТРОК ВНУТРИ блока скрипта;
            # переводим их в номера строк исходного HTML-файла.
            results.append((name, kind, script_start_idx + s_start, script_start_idx + s_end, sig))

    for i, line in enumerate(lines):
        if not in_script:
            for m in _HTML_ID.finditer(line):
                tag, elid = m.group(1), m.group(2)
                results.append((elid, f"element(<{tag}>)", i + 1, min(i + 3, len(lines)), line.strip()[:200]))
            for m in _HTML_INLINE_HANDLER.finditer(line):
                results.append((m.group(1), "inline-handler", i + 1, i + 1, line.strip()[:200]))

            if _SCRIPT_OPEN.search(line) and not _SCRIPT_SRC.search(line):
                in_script = True
                script_start_idx = i + 1  # тело скрипта начинается со следующей строки
                script_lines = []
                # <script>код на той же строке — редкий случай, но подхватим
                after = _SCRIPT_OPEN.split(line, maxsplit=1)[-1]
                if _SCRIPT_CLOSE.search(after):
                    in_script = False
                    script_start_idx = i
                    script_lines = [_SCRIPT_CLOSE.split(after, maxsplit=1)[0]]
                    flush_script()
                    script_lines = []
        else:
            if _SCRIPT_CLOSE.search(line):
                script_lines.append(_SCRIPT_CLOSE.split(line, maxsplit=1)[0])
                flush_script()
                script_lines = []
                in_script = False
            else:
                script_lines.append(line)

    if in_script:  # незакрытый <script> до конца файла — на всякий случай разбираем то, что есть
        flush_script()

    return results


_CSS_SELECTOR = re.compile(r"^\s*([^{}]+)\{\s*$")


def _extract_css(lines: list[str]) -> list[RawSymbol]:
    results: list[RawSymbol] = []
    for i, line in enumerate(lines):
        m = _CSS_SELECTOR.match(line)
        if m:
            selector = m.group(1).strip()
            if not selector or selector.startswith("@"):
                continue
            end = find_brace_block_end(lines, i)
            results.append((selector, "style", i + 1, end + 1, selector))
    return results


_EXTRACTORS = {
    ".py": _extract_python,
    ".js": _extract_js, ".ts": _extract_js, ".jsx": _extract_js, ".tsx": _extract_js,
    ".go": _extract_go,
    ".rs": _extract_rust,
    ".html": _extract_html,
    ".css": _extract_css,
}


def extract_symbols_from_file(relative_path: str, content: str) -> list[CodeSymbol]:
    """Прогоняет файл через regex-разбор под его расширение и превращает
    сырые совпадения в CodeSymbol с точными границами и обрезанным текстом."""
    extractor = _EXTRACTORS.get(Path(relative_path).suffix)
    if extractor is None:
        return []

    lines = content.splitlines()
    symbols: list[CodeSymbol] = []
    for name, kind, start, end, signature in extractor(lines):
        block_lines = lines[start - 1: min(end, start - 1 + MAX_SYMBOL_LINES)]
        block = "\n".join(block_lines)
        if len(block) > MAX_SYMBOL_CHARS:
            block = block[:MAX_SYMBOL_CHARS] + "\n…"
        symbols.append(CodeSymbol(
            id=str(uuid.uuid4()),
            file_path=relative_path,
            name=name,
            kind=kind,
            signature=signature.strip()[:200],
            start_line=start,
            end_line=end,
            text=block,
        ))
    return symbols


def build_symbols_from_project(root: Path, progress_callback=None) -> tuple[list[CodeSymbol], int]:
    """Сканирует проект и превращает файлы в список CodeSymbol."""
    files = collect_source_files(root)
    all_symbols: list[CodeSymbol] = []

    for i, file_path in enumerate(files):
        content = read_file_safely(file_path)
        if content is not None:
            relative_path = str(file_path.relative_to(root))
            all_symbols.extend(extract_symbols_from_file(relative_path, content))
        if progress_callback:
            progress_callback((i + 1) / max(len(files), 1))

    return all_symbols, len(files)


# ============================================================================
# Шаг 3: токенизация для BM25
# ============================================================================

def tokenize_for_bm25(text: str) -> list[str]:
    """
    Разбивает текст на токены для лексического поиска. Составные
    идентификаторы (camelCase, PascalCase, snake_case, kebab-case)
    дополнительно дробятся на смысловые части — это сильно повышает
    шанс совпадения, когда вопрос использует слово "переименование",
    а в коде идентификатор называется, например, "renameCard" или
    "rename-card".
    """
    raw_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
    tokens: list[str] = []
    for tok in raw_tokens:
        tokens.append(tok.lower())
        parts = re.findall(r"[A-ZА-Я]?[a-zа-я0-9]+|[A-ZА-Я]+(?![a-zа-я])", tok)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if len(p) > 1)
    return tokens


# ============================================================================
# Шаг 4: гибридный индекс символов (BM25 + эмбеддинги) и слияние через RRF
# ============================================================================

@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Кешируется на уровне процесса Streamlit — не перезагружается между поисками."""
    return SentenceTransformer(model_name)


class HybridSymbolIndex:
    """
    Объединяет лексический поиск (BM25) и семантический (эмбеддинги +
    ChromaDB) ПО СИМВОЛАМ (функции/методы/классы/обработчики/роуты/стили),
    а не по произвольным кускам текста. Результаты сливаются через
    Reciprocal Rank Fusion — каждый символ получает суммарный ранг по
    обоим спискам, сортировка по нему даёт финальный порядок кандидатов.

    Почему не один из двух: BM25 отлично ловит точные совпадения имён
    (handleClick, onCardRename), но ничего не знает о синонимах и
    перефразировках. Эмбеддинги — наоборот, хороши в перефразировках
    ("переименование карточки" → "renameCard"), но могут "размыть"
    точное совпадение редкого имени среди похожих по структуре символов.
    Вместе они компенсируют слабости друг друга.
    """

    def __init__(self, symbols: list[CodeSymbol], embedding_model: SentenceTransformer):
        self.symbols = symbols
        self.embedding_model = embedding_model
        self.id_to_symbol = {s.id: s for s in symbols}

        # --- Лексический индекс (BM25) ---
        tokenized_corpus = [tokenize_for_bm25(s.searchable_text) for s in symbols]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        # --- Семантический индекс (ChromaDB, in-memory) ---
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"code_symbols_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )
        if symbols:
            texts = [s.searchable_text for s in symbols]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            self.collection.add(
                ids=[s.id for s in symbols],
                embeddings=embeddings.tolist(),
                documents=texts,
            )

    def search(self, query: str, top_k: int) -> list[CodeSymbol]:
        """Возвращает top_k символов, отранжированных гибридным поиском (RRF)."""
        if not self.symbols or self.bm25 is None:
            return []

        pool_size = min(CANDIDATE_POOL_SIZE, len(self.symbols))

        # --- Кандидаты по BM25 ---
        bm25_scores = self.bm25.get_scores(tokenize_for_bm25(query))
        bm25_ranked_idx = sorted(range(len(self.symbols)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
        bm25_ranked_ids = [self.symbols[i].id for i in bm25_ranked_idx]

        # --- Кандидаты по эмбеддингам ---
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        semantic_result = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=pool_size)
        semantic_ranked_ids = semantic_result.get("ids", [[]])[0]

        # --- Reciprocal Rank Fusion: суммируем 1/(RRF_K + ранг) по обоим спискам ---
        rrf_scores: dict[str, float] = {}
        for rank, symbol_id in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[symbol_id] = rrf_scores.get(symbol_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, symbol_id in enumerate(semantic_ranked_ids, start=1):
            rrf_scores[symbol_id] = rrf_scores.get(symbol_id, 0.0) + 1.0 / (RRF_K + rank)

        best_ids = sorted(rrf_scores, key=lambda sid: rrf_scores[sid], reverse=True)[:top_k]
        return [self.id_to_symbol[sid] for sid in best_ids if sid in self.id_to_symbol]


# ============================================================================
# Шаг 5: один batched-запрос к Ollama — подтвердить и объяснить находку
# ============================================================================

def is_ollama_available() -> bool:
    """Быстрая проверка, что локальный сервер Ollama поднят."""
    try:
        requests.get(OLLAMA_BASE_URL, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def list_ollama_models() -> list[str]:
    """Модели, реально скачанные пользователем (`ollama pull ...`)."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return []


def pick_default_model(available_models: list[str]) -> int:
    """Индекс модели, заточенной под код, если такая уже скачана — иначе 0."""
    for preferred in PREFERRED_MODELS:
        if preferred in available_models:
            return available_models.index(preferred)
    return 0


def _extract_json_snippet(text: str) -> str:
    """Вырезает JSON-массив из текста ответа модели: даже с "format": "json"
    некоторые модели добавляют лишний текст вокруг — берём подстроку от
    первой '[' до последней ']'."""
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _coerce_to_analysis_list(parsed: object) -> list[dict] | None:
    """Приводит разобранный JSON к списку словарей независимо от формы,
    в которую его завернула модель (голый массив / {"results": [...]} /
    словарь с номерами-ключами)."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
        if parsed and all(isinstance(v, dict) for v in parsed.values()):
            items = []
            for key, value in parsed.items():
                item = dict(value)
                item.setdefault("index", int(key) if str(key).isdigit() else key)
                items.append(item)
            return items
    return None


def analyze_candidates_with_ollama_plain(question: str, candidates: list[CodeSymbol], model: str) -> list[dict]:
    """Запасной способ — БЕЗ принудительного JSON, построчным текстом вида
    "N: описание". Используется, если строгий JSON-режим не сработал."""
    numbered_blocks = [f"[{i}] ({c.kind}) {c.name} — {c.location}" for i, c in enumerate(candidates, start=1)]
    context_text = "\n".join(numbered_blocks)
    user_message = (
        f"Вопрос: {question}\n\nНайденные символы кода:\n{context_text}\n\n"
        "Для каждого номера одной строкой напиши, релевантен ли он вопросу "
        "и почему, в формате 'N: описание'. Если нерелевантен — не пиши "
        "строку для этого номера."
    )
    system_message = (
        "Ты — инструмент локализации кода в локальном приложении для чтения "
        "собственной кодовой базы пользователя на его личном компьютере."
    )
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        raise AnalysisError(f"сетевая ошибка запроса к Ollama: {exc}") from exc
    except ValueError as exc:
        raise AnalysisError(f"некорректный ответ от Ollama (не JSON): {exc}") from exc

    raw_text = payload.get("message", {}).get("content", "").strip()
    if not raw_text and "error" in payload:
        raise AnalysisError(f"Ollama вернула ошибку: {payload['error']}")
    if not raw_text:
        raise AnalysisError("Ollama вернула пустой ответ.")

    results: list[dict] = []
    for line in raw_text.splitlines():
        match = re.match(r"\s*\[?(\d+)\]?[.:)\-]\s*(.+)", line.strip())
        if match:
            description = match.group(2).strip()
            if description:
                results.append({"index": int(match.group(1)), "relevant": True, "score": 70, "reason": description})
    return results


def analyze_candidates_with_ollama(question: str, candidates: list[CodeSymbol], model: str) -> list[dict]:
    """
    ОДИН запрос к Ollama на весь набор кандидатов: модель получает вопрос и
    все найденные символы сразу, пронумерованные, и должна вернуть строгий
    JSON-массив — для каждого номера: релевантен ли он, насколько уверенно
    (0-100) и краткое объяснение. Путь к файлу и номера строк в финальном
    выводе ВСЕГДА берутся из Python-данных по номеру кандидата, а не из
    текста модели — модель не может ошибиться в локации, она только
    выбирает лучший вариант среди уже гарантированно верных кандидатов.

    Возвращает список {"index": int, "relevant": bool, "score": int, "reason": str}.
    """
    if not candidates:
        return []

    numbered_blocks = [
        f"[{i}] тип={c.kind} имя={c.name}\nсигнатура: {c.signature}\n{c.text}"
        for i, c in enumerate(candidates, start=1)
    ]
    context_text = "\n\n".join(numbered_blocks)

    system_message = (
        "Ты — инструмент локализации кода в локальном приложении для чтения "
        "собственной кодовой базы пользователя на его личном компьютере. "
        "Это обычная, безопасная задача.\n\n"
        "Тебе дан вопрос пользователя (он ищет, ГДЕ в коде определён или "
        "используется конкретный обработчик/функция/маршрут/стиль) и "
        "несколько пронумерованных кандидатов, уже найденных гибридным "
        "поиском. Для КАЖДОГО кандидата реши: действительно ли это то самое "
        "место, о котором спрашивает пользователь, и насколько ты уверен.\n\n"
        "Будь щедрым, но честным: если кандидат задействует ту же сущность "
        "(то же имя функции/переменной/id/класса/маршрута), что и в вопросе, "
        "или явно реализует описанное поведение — считай его релевантным.\n\n"
        "Ответь СТРОГО в виде JSON-массива и ничего, кроме него — без "
        "вступлений, пояснений и markdown-разметки. По одному объекту на "
        "каждый номер кандидата, в точности в этом формате:\n"
        '[{"index": 1, "relevant": true, "score": 90, "reason": "..."}, '
        '{"index": 2, "relevant": false, "score": 0, "reason": ""}]\n\n'
        "score — целое число 0-100 (насколько это именно то место). "
        "reason заполняй ТОЛЬКО если relevant=true — одна короткая фраза "
        "на русском языке (до 15 слов), что конкретно делает этот код "
        "применительно к вопросу. Если relevant=false — reason оставь "
        "пустой строкой."
    )
    user_message = f"Вопрос: {question}\n\nНайденные кандидаты:\n{context_text}"

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        raise AnalysisError(f"сетевая ошибка запроса к Ollama: {exc}") from exc
    except ValueError as exc:
        raise AnalysisError(f"некорректный ответ от Ollama (не JSON): {exc}") from exc

    raw_text = payload.get("message", {}).get("content", "").strip()
    if not raw_text and "error" in payload:
        raise AnalysisError(f"Ollama вернула ошибку: {payload['error']}")
    if not raw_text:
        raise AnalysisError("Ollama вернула пустой ответ.")

    try:
        parsed = json.loads(_extract_json_snippet(raw_text))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"не удалось разобрать JSON-ответ модели: {exc}") from exc

    analysis_list = _coerce_to_analysis_list(parsed)
    if analysis_list is None:
        raise AnalysisError(
            f"модель вернула JSON в неожиданной форме ({type(parsed).__name__}), "
            "не удалось привести к списку кандидатов."
        )
    return analysis_list


# ============================================================================
# Вспомогательные функции интерфейса
# ============================================================================

def pick_folder_native() -> str | None:
    """
    Открывает системное окно выбора папки через AppleScript, запущенный
    ОТДЕЛЬНЫМ процессом (Streamlit выполняет код в фоновом потоке, а GUI-
    диалоги на macOS обязаны работать в главном потоке). Поддерживается
    только на macOS.
    """
    if sys.platform != "darwin":
        return None

    apple_script = 'POSIX path of (choose folder with prompt "Выберите папку проекта")'
    try:
        result = subprocess.run(
            ["osascript", "-e", apple_script], capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


KIND_ICONS = {
    "function": "🔧", "method": "🔧", "class": "🏛️", "route": "🌐",
    "listener": "🎯", "handler-ref": "🎯", "inline-handler": "🎯",
    "style": "🎨",
}


def kind_icon(kind: str) -> str:
    for prefix, icon in KIND_ICONS.items():
        if kind.startswith(prefix):
            return icon
    return "📍"


def render_symbol_line(symbol: CodeSymbol, rank: int | None = None, reason: str | None = None,
                        score: int | None = None, highlight: bool = False) -> None:
    """Компактная строка с локацией найденного символа."""
    prefix = f"**#{rank}**  " if rank is not None else ""
    icon = kind_icon(symbol.kind)
    score_badge = f"  · уверенность {score}%" if score is not None else ""
    line = f"{prefix}{icon} **{symbol.name}** ({symbol.kind}) — `{symbol.location}`{score_badge}"
    if highlight:
        st.success(line)
    else:
        st.markdown(line)
    if reason:
        st.caption(reason)
    elif symbol.signature:
        preview = symbol.signature if len(symbol.signature) <= 100 else symbol.signature[:100] + "…"
        st.caption(f"`{preview}`")


def render_sidebar() -> tuple[bool, bool, str, int]:
    """Отрисовывает боковую панель и возвращает (index_button, use_generation, ollama_model, top_k)."""
    with st.sidebar:
        st.header("1. Индексация проекта")

        st.session_state.setdefault("project_path_input", "")

        def _handle_browse_click() -> None:
            picked = pick_folder_native()
            if picked:
                st.session_state["project_path_input"] = picked

        path_col, browse_col = st.columns([4, 1])
        with path_col:
            st.text_input(
                "Путь к папке проекта",
                key="project_path_input",
                placeholder="/Users/you/Downloads/my_project",
            )
        with browse_col:
            st.write("")
            st.button("📂", help="Выбрать папку через системный диалог", on_click=_handle_browse_click)

        index_button = st.button("📚 Проиндексировать", use_container_width=True)

        if st.session_state.index is not None:
            st.info(
                f"**Текущий индекс:**\n\n"
                f"📁 `{st.session_state.indexed_project}`\n\n"
                f"📄 файлов: {st.session_state.file_count}  \n"
                f"🧩 символов: {st.session_state.symbol_count}"
            )

        st.divider()
        st.header("2. Подтверждение через LLM")

        ollama_ready = is_ollama_available()
        use_generation = st.toggle(
            "Подтверждать находку через локальную LLM (Ollama)",
            value=ollama_ready,
            disabled=not ollama_ready,
            help="Требует установленной и запущенной Ollama (localhost:11434).",
        )

        available_models = list_ollama_models() if ollama_ready else []
        if available_models:
            ollama_model = st.selectbox(
                "Модель Ollama",
                options=available_models,
                index=pick_default_model(available_models),
                help="Список моделей, уже скачанных через `ollama pull`.",
            )
        else:
            ollama_model = st.text_input("Модель Ollama", value=FALLBACK_MODEL_NAME)

        if not ollama_ready:
            st.warning(
                "Ollama не обнаружена на localhost:11434. Поиск будет работать, "
                "но без подтверждения — только найденные кандидаты по гибридному поиску.\n\n"
                "Установка: https://ollama.com/download, затем `ollama pull qwen2.5-coder`"
            )
        elif not available_models:
            st.warning("Ollama запущена, но моделей не найдено. Выполните: `ollama pull qwen2.5-coder`")

        top_k = st.slider(
            "Сколько кандидатов рассматривать", min_value=3, max_value=15, value=DEFAULT_TOP_K,
        )
        st.caption(
            "Гибридный поиск (BM25 + эмбеддинги) находит лучших кандидатов среди "
            "уже распарсенных символов кода, модель лишь подтверждает лучший — "
            "точные номера строк она не придумывает."
        )

    return index_button, use_generation, ollama_model, top_k


def handle_indexing(project_path: str) -> None:
    """Обрабатывает нажатие кнопки «Проиндексировать»."""
    root = Path(project_path).expanduser().resolve()
    if not project_path:
        st.error("Укажите путь к проекту.")
        return
    if not root.exists():
        st.error(f"Путь не существует: {root}")
        return
    if not root.is_dir():
        st.error(f"Это не папка: {root}")
        return

    progress_bar = st.progress(0.0, text="Сканирование и разбор символов...")
    try:
        symbols, n_files = build_symbols_from_project(
            root, progress_callback=lambda p: progress_bar.progress(p * 0.5, text="Разбираю функции/классы/обработчики...")
        )
        if not symbols:
            st.warning(
                f"Не удалось найти ни одного символа (функции/классы/обработчики/роуты/стили) "
                f"в файлах с расширениями: {', '.join(sorted(CODE_EXTENSIONS))}."
            )
            return

        progress_bar.progress(0.6, text="Строю BM25-индекс и векторизую символы...")
        embedding_model = load_embedding_model()
        index = HybridSymbolIndex(symbols, embedding_model)
        progress_bar.progress(1.0, text="Готово!")

        st.session_state.index = index
        st.session_state.indexed_project = str(root)
        st.session_state.symbol_count = len(symbols)
        st.session_state.file_count = n_files
        st.success(f"Проиндексировано {n_files} файлов, найдено {len(symbols)} символов кода.")
    except Exception as exc:
        st.error(f"Ошибка при индексации: {exc}")


def handle_search(question: str, ollama_model: str, use_generation: bool, ollama_ready: bool, top_k: int) -> None:
    """Обрабатывает нажатие кнопки «Найти»: гибридный поиск + (опционально) подтверждение через LLM."""
    with st.spinner("Ищу подходящие символы (BM25 + эмбеддинги)..."):
        try:
            candidates = st.session_state.index.search(question, top_k=top_k)
        except Exception as exc:
            st.error(f"Ошибка поиска: {exc}")
            return

    if not candidates:
        st.warning("Ничего не найдено. Возможно, обработчик написан в стиле, который парсер не распознаёт.")
        return

    if not (use_generation and ollama_ready):
        st.subheader(f"📍 Найдено кандидатов: {len(candidates)}")
        st.caption("Включите подтверждение через LLM в боковой панели для проверки и объяснения.")
        for rank, symbol in enumerate(candidates, start=1):
            render_symbol_line(symbol, rank=rank, highlight=(rank == 1))
        return

    with st.spinner(f"Проверяю {len(candidates)} кандидатов одним запросом ({ollama_model})..."):
        try:
            analysis = analyze_candidates_with_ollama(question, candidates, ollama_model)
        except AnalysisError as exc_json:
            # Строгий JSON-режим не сработал — пробуем запасной вариант
            # обычным текстом, прежде чем совсем сдаться.
            try:
                analysis = analyze_candidates_with_ollama_plain(question, candidates, ollama_model)
            except AnalysisError as exc_plain:
                st.error(
                    f"⚠️ Не удалось получить ответ от Ollama.\n\n"
                    f"JSON-режим: {exc_json}\n\nТекстовый режим: {exc_plain}\n\n"
                    f"Частая причина — модель ещё не скачана: `ollama pull {ollama_model}`."
                )
                st.markdown("**Но вот что нашёл гибридный поиск (без подтверждения LLM):**")
                render_symbol_line(candidates[0], highlight=True)
                for rank, symbol in enumerate(candidates[1:], start=2):
                    render_symbol_line(symbol, rank=rank)
                return

    relevant: list[tuple[CodeSymbol, int, str]] = []  # (symbol, score, reason)
    for item in analysis:
        idx = item.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(candidates)):
            continue
        if not item.get("relevant"):
            continue
        try:
            score = int(item.get("score", 50))
        except (TypeError, ValueError):
            score = 50
        reason = str(item.get("reason", "")).strip()
        relevant.append((candidates[idx - 1], score, reason))

    st.subheader(f"📍 Где находится обработчик — найдено релевантных: {len(relevant)}")
    if relevant:
        # Сортируем по уверенности модели; при равном score — по исходному
        # рангу гибридного поиска (кто был выше в candidates, тот и раньше).
        candidate_rank = {id(symbol): i for i, symbol in enumerate(candidates)}
        relevant.sort(key=lambda t: (-t[1], candidate_rank.get(id(t[0]), 999)))

        if len({score for _, score, _ in relevant}) == 1 and len(relevant) > 1:
            st.caption(
                "⚠️ Все найденные варианты получили одинаковую уверенность — "
                "возможно, вопрос описывает сразу несколько связанных мест "
                "(например, сам обработчик события и функцию/метод, которую он вызывает). "
                "Посмотрите код каждого варианта ниже, чтобы выбрать нужный."
            )

        for rank, (symbol, score, reason) in enumerate(relevant, start=1):
            render_symbol_line(symbol, rank=rank, reason=reason, score=score, highlight=(rank == 1))
    else:
        st.info(
            "Среди найденных кандидатов ни один не показался модели явно релевантным. "
            "Попробуйте переформулировать вопрос ближе к терминам из кода, увеличить "
            "«Сколько кандидатов рассматривать», или выбрать модель, заточенную под код "
            "(например, qwen2.5-coder)."
        )
        st.markdown("**Но вот что нашёл гибридный поиск (без подтверждения LLM):**")
        for rank, symbol in enumerate(candidates, start=1):
            render_symbol_line(symbol, rank=rank)

    with st.expander("🔍 Показать всех кандидатов и код целиком"):
        for rank, symbol in enumerate(candidates, start=1):
            st.markdown(f"**#{rank}** {kind_icon(symbol.kind)} `{symbol.location}` — {symbol.name} ({symbol.kind})")
            st.code(symbol.text, language=None)


def handle_browse(filter_text: str) -> None:
    """Вкладка «Все определения»: просто список всех найденных символов,
    без вопроса и без LLM — как облегчённый ctags/symbol table."""
    symbols: list[CodeSymbol] = st.session_state.index.symbols
    if filter_text.strip():
        needle = filter_text.strip().lower()
        symbols = [s for s in symbols if needle in s.name.lower() or needle in s.file_path.lower()]

    st.caption(f"Показано {len(symbols)} из {st.session_state.symbol_count} символов.")
    for symbol in symbols[:300]:
        render_symbol_line(symbol)
    if len(symbols) > 300:
        st.caption(f"…и ещё {len(symbols) - 300}. Уточните фильтр.")


# ============================================================================
# Точка входа
# ============================================================================

def main() -> None:
    st.set_page_config(page_title="Handler Locator", page_icon="📍", layout="wide")
    st.title("📍 Handler Locator")
    st.caption(
        "Находит ТОЧНОЕ место в коде, где определён обработчик/функция/маршрут/стиль — "
        "символьный индекс + гибридный поиск + локальная LLM для подтверждения."
    )

    for key, default in [
        ("index", None), ("indexed_project", None), ("symbol_count", 0), ("file_count", 0),
    ]:
        st.session_state.setdefault(key, default)

    index_button, use_generation, ollama_model, top_k = render_sidebar()
    ollama_ready = is_ollama_available()

    if index_button:
        handle_indexing(st.session_state["project_path_input"])

    if st.session_state.index is None:
        st.info("👈 Сначала укажите путь к проекту и нажмите «Проиндексировать» в боковой панели.")
        return

    tab_search, tab_browse = st.tabs(["🔎 Найти обработчик", "🗂 Все определения"])

    with tab_search:
        question = st.text_input(
            "Где находится...",
            placeholder="Где обработчик клика по карточке? / Где определён роут /api/login?",
        )
        search_clicked = st.button("🔎 Найти", type="primary")

        if search_clicked and question.strip():
            handle_search(question, ollama_model, use_generation, ollama_ready, top_k)
        elif search_clicked:
            st.warning("Введите вопрос.")

    with tab_browse:
        filter_text = st.text_input("Фильтр по имени или пути файла", key="browse_filter")
        handle_browse(filter_text)


if __name__ == "__main__":
    main()
