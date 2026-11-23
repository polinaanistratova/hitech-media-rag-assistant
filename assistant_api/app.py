"""
Консольное приложение для взаимодействия с RAG-ассистентом ООО «ХАЙТЕК МЕДИА» (API Mode).
Объединяет векторный поиск ChromaDB, SQLite-кеширование, фильтрацию метаданных и память диалога.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def print_banner():
    """Вывод приветственного баннера."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║     RAG-Ассистент ООО «ХАЙТЕК МЕДИА» (API Mode: Gemini/OpenAI)   ║
║  ChromaDB + Семантический чанкинг + SQLite-кеш + Память + RAGAS  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Доступные команды:")
    print("  'stats'          — статистика векторной БД ChromaDB, кеша и памяти")
    print("  'clear'          — очистить SQLite-кеш ответов")
    print("  'forget'         — очистить краткосрочную память диалога")
    print("  'filter:services'— искать только по услугам (1_hitech_services.txt)")
    print("  'filter:faq'     — искать только по FAQ (2_hitech_faq.txt)")
    print("  'filter:sla'     — искать только по регламенту SLA (3_hitech_regulations.txt)")
    print("  'filter:off'     — отключить фильтр метаданных")
    print("  'exit' / 'quit'  — выход из программы\n")


def print_response(result: dict):
    """Форматированный вывод ответа с источниками и временем выполнения."""
    print(f"\n{'─'*68}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*68}")

    elapsed = result.get('elapsed_sec', 0.0)
    if result['from_cache']:
        print(f"💾 Источник ответа: SQLite КЕШ (время: {elapsed:.4f} сек.)")
        if result.get('cached_at'):
            print(f"   Сохранено в кеш: {result['cached_at']}")
    else:
        print(f"🌐 Источник ответа: {result.get('model', 'LLM')} (время: {elapsed:.4f} сек.)")
        print(f"   Извлечено релевантных чанков: {len(result.get('context_docs', []))}")

    print(f"\n💬 Ответ:\n{result['answer']}")

    context_docs = result.get('context_docs') or []
    if context_docs and isinstance(context_docs[0], dict):
        print(f"\n📚 Использованные чанки из ChromaDB:")
        for i, doc in enumerate(context_docs[:3], 1):
            src = doc.get('source', 'docs.txt')
            sec = doc.get('section', '')
            sim = doc.get('similarity', '')
            sim_str = f" | score={sim}" if sim != '' else ""
            preview = doc['text'][:140].replace('\n', ' ') + "..."
            print(f"   {i}. [{src} | {sec}{sim_str}]\n      {preview}")

    print(f"{'─'*68}\n")


def print_stats(pipeline: RAGPipeline):
    """Вывод статистики системы."""
    stats = pipeline.get_stats()

    print(f"\n{'═'*68}")
    print("📊 СТАТИСТИКА RAG-СИСТЕМЫ ООО «ХАЙТЕК МЕДИА»")
    print(f"{'═'*68}")

    print("\n🗄️  Векторное хранилище ChromaDB:")
    print(f"   Коллекция:        {stats['vector_store']['name']}")
    print(f"   Чанков в базе:    {stats['vector_store']['count']}")
    print(f"   Директория:       {stats['vector_store']['persist_directory']}")
    print(f"   Файлы-источники:  {', '.join(stats['vector_store'].get('sources', []))}")

    print("\n💾 SQLite-кеш запросов:")
    print(f"   Записей в кеше:   {stats['cache']['total_entries']}")
    print(f"   Размер БД:        {stats['cache']['db_size_mb']:.3f} MB")
    if stats['cache']['oldest_entry']:
        print(f"   Первая запись:    {stats['cache']['oldest_entry']}")
    if stats['cache']['newest_entry']:
        print(f"   Последняя запись: {stats['cache']['newest_entry']}")

    print("\n🧠 Краткосрочная память сессии:")
    print(f"   Сообщений в памяти: {stats.get('memory_messages', 0)}")
    print(f"   Имя пользователя:   {stats.get('user_name') or 'не указано'}")

    print(f"\n🤖 Модель: {stats['model']} (TOP_K={stats.get('top_k', 3)})")
    print(f"🌐 Режим:  {stats['mode']}")
    print(f"{'═'*68}\n")


def main():
    """Главная функция консольного приложения."""
    print_banner()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Ошибка: переменная окружения OPENAI_API_KEY / GEMINI_API_KEY не установлена в .env")
        sys.exit(1)

    try:
        print("🚀 Инициализация RAG-ассистента ООО «ХАЙТЕК МЕДИА»...\n")
        pipeline = RAGPipeline(
            collection_name="hitech_api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/docs.txt",
            model="gemini-2.5-flash / gpt-4o-mini"
        )
        print("\n✅ Система готова к работе!\n")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        sys.exit(1)

    active_filter = None

    while True:
        try:
            prompt_prefix = f" [фильтр: {active_filter}]" if active_filter else ""
            user_input = input(f"💭 Ваш вопрос{prompt_prefix}: ").strip()

            if user_input.lower() in ['exit', 'quit', 'q', 'выход']:
                print("\n👋 До свидания!")
                break

            if user_input.lower() == 'stats':
                print_stats(pipeline)
                continue

            if user_input.lower() == 'clear':
                pipeline.cache.clear()
                print("✅ SQLite-кеш ответов успешно очищен!\n")
                continue

            if user_input.lower() == 'forget':
                pipeline.clear_memory()
                print("✅ Краткосрочная память диалога очищена!\n")
                continue

            if user_input.lower().startswith('filter:'):
                f_val = user_input.split(':', 1)[1].strip().lower()
                if f_val in ['off', 'none', 'reset']:
                    active_filter = None
                    print("✅ Фильтр метаданных отключён (поиск по всей базе).\n")
                elif f_val in ['services', 'faq', 'sla']:
                    active_filter = {"category": f_val}
                    print(f"✅ Установлен фильтр метаданных: category = '{f_val}'\n")
                else:
                    print("⚠️ Доступные фильтры: filter:services, filter:faq, filter:sla, filter:off\n")
                continue

            if not user_input:
                print("⚠️ Пожалуйста, введите вопрос\n")
                continue

            result = pipeline.query(user_input, where=active_filter)
            print_response(result)

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
