"""
Модуль работы с векторным хранилищем ChromaDB.
Обрабатывает очистку текста, загрузку документов, семантический chunking,
метаданные источников и поиск по векторам с фильтрацией.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе ChromaDB для RAG-ассистента ООО «ХАЙТЕК МЕДИА»."""

    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./chroma_db"):
        """
        Инициализация векторного хранилища.

        Args:
            collection_name: имя коллекции в ChromaDB
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Инициализация ChromaDB клиента
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Локальная векторная модель ChromaDB (ONNX 384-dim) + проверка OpenAI
        self.default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key.startswith("sk-") and api_key != "sk-local-rag-key":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
            except Exception:
                self.openai_client = None

        # Получение или создание коллекции
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая коллекция '{collection_name}'")

    @staticmethod
    def clean_raw_text(text: str) -> str:
        """
        Предобработка и очистка исходного текста:
        - склейка слов с дефисным переносом строк (например, 'реагиро-\nвания' -> 'реагирования')
        - удаление колонтитулов ('Стр. X из Y', 'КОНФИДЕНЦИАЛЬНО')
        - нормализация пробелов и пустых строк
        """
        # Склейка дефисных переносов слов
        text = re.sub(r'([а-яА-Яa-zA-Z])-\s*\n\s*([а-яА-Яa-zA-Z])', r'\1\2', text)
        # Удаление технических разделителей и номеров страниц
        text = re.sub(r'Стр\.\s*\d+\s*из\s*\d+', '', text)
        text = re.sub(r'-{10,}', '', text)
        # Нормализация множественных переносов строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 550, overlap: int = 90) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.

        Стратегия:
        1. Приоритет смысловым блокам и абзацам (разделение по \n\n)
        2. Если абзац начинается с мета-заголовка '[Компания: ...]', он сохраняется как единый смысловой квант
        3. Разбиение слишком длинных абзацев по предложениям
        4. Сохранение контекста через overlap
        """
        text = self.clean_raw_text(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Если документ уже структурирован по тематическим блокам '[Компания: ...]'
        structured_blocks = [p for p in paragraphs if p.startswith('[Компания:')]
        if len(structured_blocks) >= 3 and len(structured_blocks) == len(paragraphs):
            return structured_blocks

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Получение текста для overlap из конца предыдущего чанка (по границам предложений)."""
        if len(text) <= overlap_size:
            return text

        overlap_candidate = text[-overlap_size:]
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0

        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)

        if best_start > 0:
            return overlap_candidate[best_start:].strip()

        return overlap_candidate.strip()

    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """Разбиение длинного абзаца на чанки по предложениям."""
        sentences = re.split(r'([.!?]+\s+)', paragraph)

        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])

        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        chunks = []
        current_chunk = ""

        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = f"{overlap_text} {sentence}" if overlap_text else sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def _extract_metadata(chunk: str, idx: int) -> Dict[str, Any]:
        """Извлекает метаданные раздела и файла-источника из заголовка чанка."""
        source = "docs.txt"
        section = f"Раздел {idx + 1}"

        src_match = re.search(r'Источник:\s*([^\]]+)\]', chunk)
        if src_match:
            source = src_match.group(1).strip()

        sec_match = re.search(r'Раздел:\s*([^|\]]+)', chunk)
        if sec_match:
            section = sec_match.group(1).strip()

        category = "general"
        if "1_hitech_services" in source:
            category = "services"
        elif "2_hitech_faq" in source:
            category = "faq"
        elif "3_hitech_regulations" in source:
            category = "sla"

        return {
            "source": source,
            "section": section,
            "category": category,
            "chunk_index": idx
        }

    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI API или локальную модель ChromaDB.
        """
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception:
                self.openai_client = None

        emb = self.default_ef([text])[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла в векторное хранилище ChromaDB.
        """
        if not os.path.exists(file_path):
            abs_candidate = Path(__file__).parent / file_path
            if abs_candidate.exists():
                file_path = str(abs_candidate)
            else:
                raise FileNotFoundError(f"Файл {file_path} не найден")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self._chunk_text(text)
        print(f"Текст очищен и разбит на {len(chunks)} смысловых чанков")

        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        documents = []
        ids = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            embedding = self._create_embedding(chunk)
            meta = self._extract_metadata(chunk, i)

            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)
            metadatas.append(meta)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Загружено {len(chunks)} чанков с метаданными в коллекцию '{self.collection_name}'")

    def search(self, query: str, top_k: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Гибридный семантический поиск релевантных документов по запросу (Dense ChromaDB + Keyword Boost).

        Args:
            query: текст запроса
            top_k: количество документов для возврата
            where: фильтр по метаданным (например, {"source": "3_hitech_regulations.txt"})

        Returns:
            список документов с метаданными и оценкой расстояния
        """
        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        query_embedding = self._create_embedding(query)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(max(top_k * 3, total_docs), total_docs)
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        q_tokens = set(re.findall(r'[а-яА-Яa-zA-Z0-9]{3,}', query.lower()))
        stopwords = {"какие", "какой", "какая", "какое", "сколько", "компании", "компания", "хайтек", "медиа", "ооо", "для", "при", "или", "как", "что"}
        q_keywords = {w[:5] for w in q_tokens if w not in stopwords}

        candidates = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                raw_dist = results['distances'][0][i] if 'distances' in results and results['distances'] else 0.3
                meta = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}

                doc_lower = doc_text.lower()
                kw_hits = sum(1 for kw in q_keywords if kw in doc_lower)
                adjusted_dist = max(0.01, float(raw_dist) - kw_hits * 0.08)

                candidates.append({
                    'id': results['ids'][0][i],
                    'text': doc_text,
                    'metadata': meta,
                    'source': meta.get('source', 'docs.txt'),
                    'section': meta.get('section', ''),
                    'distance': round(adjusted_dist, 4),
                    'similarity': round(1.0 - min(adjusted_dist, 0.99), 4)
                })

        candidates.sort(key=lambda x: x['distance'])
        return candidates[:top_k]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Получение статистики коллекции."""
        return {
            'name': self.collection_name,
            'count': self.collection.count(),
            'persist_directory': self.persist_directory,
            'sources': ["1_hitech_services.txt", "2_hitech_faq.txt", "3_hitech_regulations.txt"]
        }


if __name__ == "__main__":
    vector_store = VectorStore(collection_name="test_hitech_collection")
    vector_store.load_documents("data/docs.txt")

    results = vector_store.search("Какое время реакции техподдержки на критические инциденты?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. [{doc['source']} | {doc['section']}] (Similarity: {doc['similarity']})")
        print(f"   {doc['text'][:180]}...")

    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")
