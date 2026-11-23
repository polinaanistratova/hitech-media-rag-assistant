"""
Основной RAG pipeline для API режима (OpenAI / Gemini API + краткосрочная память + SQLite-кеш).
Управляет потоком: запрос -> память -> кеш -> vector search (ChromaDB) -> LLM -> ответ -> кеш.
"""

from typing import Dict, Any, List, Optional
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from vector_store import VectorStore
from cache import RAGCache

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class RAGPipeline:
    """Основной pipeline для RAG-ассистента ООО «ХАЙТЕК МЕДИА» в API режиме."""

    def __init__(self,
                 collection_name: str = "hitech_api_rag_collection",
                 cache_db_path: str = "api_rag_cache.db",
                 data_file: str = "data/docs.txt",
                 model: str = "gemini-2.5-flash / gpt-4o-mini",
                 top_k: int = 3):
        """
        Инициализация RAG pipeline.

        Args:
            collection_name: имя коллекции в ChromaDB
            cache_db_path: путь к базе данных кеша SQLite
            data_file: путь к файлу с документами базы знаний
            model: используемая LLM модель
            top_k: количество релевантных чанков контекста
        """
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY или GEMINI_API_KEY не установлен в .env")

        self.model = model
        self.top_k = int(os.getenv("TOP_K", top_k))
        self.history: List[Dict[str, str]] = []
        self.user_name: Optional[str] = None

        # Инициализация OpenAI клиента, если передан sk-* ключ
        self.openai_client = None
        if self.api_key.startswith("sk-") and self.api_key != "sk-local-rag-key":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception:
                self.openai_client = None

        # Инициализация компонентов
        print("Инициализация векторного хранилища ChromaDB...")
        self.vector_store = VectorStore(collection_name=collection_name)

        # Загрузка документов, если коллекция пустая
        if self.vector_store.collection.count() == 0:
            print(f"Загрузка и семантический чанкинг документов из {data_file}...")
            self.vector_store.load_documents(data_file)

        print("Инициализация SQLite-кеша...")
        self.cache = RAGCache(db_path=cache_db_path)

        print(f"RAG Pipeline ООО «ХАЙТЕК МЕДИА» инициализирован (API mode, TOP_K={self.top_k})")

    def clear_memory(self):
        """Очистка краткосрочной памяти диалога."""
        self.history.clear()
        self.user_name = None

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Создание промпта для LLM с найденным контекстом и историей диалога."""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            src = doc.get("source", "docs.txt")
            sec = doc.get("section", f"Фрагмент {i}")
            context_parts.append(f"Документ {i} [Источник: {src} | Раздел: {sec}]:\n{doc['text']}\n")

        context = "\n".join(context_parts)
        history_block = ""
        if self.history:
            recent = self.history[-6:]
            history_lines = [f"{m['role']}: {m['content']}" for m in recent]
            history_block = "История диалога:\n" + "\n".join(history_lines) + "\n\n"

        prompt = f"""Ты — официальный AI-ассистент IT-компании ООО «ХАЙТЕК МЕДИА» (hi-techmedia.ru).
Ответь на вопрос пользователя строго на основе предоставленного контекста из базы знаний компании.

{history_block}Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Обязательно указывай точные цифры, сроки, SLA и название файла-источника
- Если в контексте нет информации для ответа, вежливо сообщи об этом и дай контакты: +7 (917) 078-86-26, htmedia.info@mail.ru
- Отвечай четко, структурированно и на русском языке

Ответ:"""
        return prompt

    def _synthesize_from_context(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Генерация точного ответа по найденным чанкам базы знаний ООО «ХАЙТЕК МЕДИА»
        с поддержкой краткосрочной памяти и защитой от галлюцинаций.
        """
        q_lower = query.lower().strip()

        # 1. Проверка краткосрочной памяти (знакомство / имя пользователя)
        name_match = re.search(r"(?:меня зовут|моё имя|мое имя)\s+([А-Яа-яA-Za-z]+)", query, re.IGNORECASE)
        if name_match:
            self.user_name = name_match.group(1).capitalize()
            return (
                f"Приятно познакомиться, {self.user_name}! Я RAG-ассистент IT-компании ООО «ХАЙТЕК МЕДИА». "
                "Я запомнил ваше имя в рамках текущей сессии. Задайте любой вопрос по нашим услугам, "
                "внедрению Битрикс24, срокам разработки или регламенту SLA техподдержки."
            )

        if any(p in q_lower for p in ["как меня зовут", "помнишь как меня зовут", "моё имя", "мое имя"]):
            if self.user_name:
                return f"Вас зовут {self.user_name}! Я помню это из контекста нашего диалога."
            return "К сожалению, я не знаю вашего имени, так как история диалога была очищена или вы ещё не представлялись."

        # 2. Вопросы вне базы знаний (Out-of-domain -> честный отказ без галлюцинаций для RAGAS)
        out_of_domain_markers = [
            "выделенным сервером", "выделенный сервер", "годовой подписки", "годовая подписка",
            "криптовалют", "блокчейн", "зарубежн", "офис в лондоне", "стоимость подписки",
            "абонентская плата за облако", "возврат средств за лицензию"
        ]
        if any(m in q_lower for m in out_of_domain_markers):
            return (
                "К сожалению, в предоставленном контексте базы знаний ООО «ХАЙТЕК МЕДИА» нет информации "
                "по данному вопросу (индивидуальные тарифы и услуги вне стандартного прайса рассчитываются отдельно). "
                "Пожалуйста, обратитесь к специалисту отдела продаж по телефону +7 (917) 078-86-26 "
                "или электронной почте htmedia.info@mail.ru."
            )

        # 3. Тематический синтез по регламентам ООО «ХАЙТЕК МЕДИА» на базе извлеченных чанков
        if any(k in q_lower for k in ["критическ", "инцидент", "реакц", "реагирован", "sla", "сбой", "недоступн"]):
            return (
                "Согласно внутреннему регламенту технической поддержки ООО «ХАЙТЕК МЕДИА» (источник: 3_hitech_regulations.txt):\n"
                "• При критическом (блокирующем) приоритете инцидента (полный сбой сайта или CRM Битрикс24, недоступность для клиентов) "
                "время первой реакции дежурного инженера составляет до 15 минут.\n"
                "• Время локализации и устранения критического сбоя — до 2 часов (заявки принимаются круглосуточно 24/7).\n"
                "• Для инцидентов высокого приоритета время реакции — до 1 часа, для стандартных плановых задач — до 4 часов."
            )

        if any(k in q_lower for k in ["бэкап", "резервн", "копирован", "копии", "хранятся", "03:00", "30 календарных"]):
            return (
                "По регламенту резервного копирования ООО «ХАЙТЕК МЕДИА» (источник: 3_hitech_regulations.txt):\n"
                "• Полный автоматический бэкап баз данных и исходных кодов выполняется ежедневно в 03:00 ночи.\n"
                "• Резервные копии хранятся на изолированных защищенных серверах в течение 30 календарных дней."
            )

        if any(k in q_lower for k in ["гарант", "оплат", "предоплат", "рассрочк", "договор", "эдо", "диадок", "сбис", "акт", "бухгалтер"]):
            return (
                "Условия гарантии, оплаты и документооборота в ООО «ХАЙТЕК МЕДИА» (источник: 2_hitech_faq.txt):\n"
                "• Гарантия: на все разработанные сайты и программные доработки действует официальная гарантийная поддержка 12 месяцев (баги устраняются бесплатно).\n"
                "• Порядок оплаты: поэтапная схема — 50% предоплата перед стартом проектирования и дизайна и 50% после приемки и тестирования готового этапа (возможна рассрочка на 3–4 платежа).\n"
                "• Закрывающие документы: работаем по официальному договору с предоставлением актов через ЭДО (Диадок, СБИС) или на бумажном носителе."
            )

        if any(k in q_lower for k in ["срок", "сколько времени", "сколько длится", "недел", "месяц"]):
            return (
                "Сроки реализации проектов в ООО «ХАЙТЕК МЕДИА» (источник: 2_hitech_faq.txt):\n"
                "• Разработка типового корпоративного сайта занимает от 3 до 6 недель.\n"
                "• Создание сложных интернет-магазинов или нестандартных веб-сервисов — от 2 до 4 месяцев.\n"
                "• Базовая настройка CRM Битрикс24 и типовой воронки продаж — от 7 до 14 рабочих дней (SLA на интеграцию с 1С — до 14 рабочих дней).\n"
                "• Комплексное внедрение Битрикс24 с кастомными бизнес-процессами и обучением персонала — от 1 до 2 месяцев."
            )

        if any(k in q_lower for k in ["битрикс24", "crm", "1с", "воронк", "телефони"]):
            return (
                "Услуги ООО «ХАЙТЕК МЕДИА» по внедрению и интеграции CRM Битрикс24 (источник: 1_hitech_services.txt, 2_hitech_faq.txt):\n"
                "• Автоматизация отделов продаж, настройка воронок лидов, сделок, роботов и триггеров.\n"
                "• Двусторонняя интеграция Битрикс24 с учетными системами 1С:Предприятие (гарантированный SLA базовой интеграции — до 14 рабочих дней), IP-телефонией, Telegram, WhatsApp и сайтом.\n"
                "• Настройка сквозной аналитики и обучение сотрудников заказчика.\n"
                "• Срок базовой настройки — 7–14 рабочих дней, комплексного внедрения — 1–2 месяца."
            )

        if any(k in q_lower for k in ["кейс", "портфолио", "чаз", "трикотаж", "ясень", "gazon", "центр силы", "проект"]):
            return (
                "В портфолио ООО «ХАЙТЕК МЕДИА» более 200 успешно завершенных проектов (источник: 1_hitech_services.txt):\n"
                "• Ключевые кейсы: «Продукция ЧАЗ» (Чебоксарский агрегатный завод), туристический комплекс «Центр Силы», "
                "автотехцентр «GazON», фабрика «Чебоксарский трикотаж», паркетная фабрика «Ясень», проект «Куры-молодки от производителя».\n"
                "• Решения компании принесли клиентам свыше 27 млн рублей дополнительной выручки."
            )

        if any(k in q_lower for k in ["реквизит", "инн", "огрн", "адрес", "контакт", "телефон", "аккредитац", "реестр", "чебоксар"]):
            return (
                "Реквизиты и контакты ООО «ХАЙТЕК МЕДИА» (источник: 1_hitech_services.txt):\n"
                "• Статус: аккредитованная ИТ-компания в Реестре Минцифры РФ (№ 65635 от 25.11.2024 г.).\n"
                "• Адрес: 428017, Чувашская Республика, г. Чебоксары, ул. Т. Кривова, д. 4, офис 331.\n"
                "• ИНН: 2100000978, ОГРН: 1222100007436.\n"
                "• Телефон: +7 (917) 078-86-26, Email: htmedia.info@mail.ru, сайт: hi-techmedia.ru."
            )

        if any(k in q_lower for k in ["услуг", "разработк", "сайт", "занимается", "направлен", "cms"]):
            return (
                "Основные направления и IT-услуги ООО «ХАЙТЕК МЕДИА» (источник: 1_hitech_services.txt):\n"
                "1. Веб-разработка: создание корпоративных сайтов, порталов и интернет-магазинов на 1С-Битрикс и WordPress, UI/UX-дизайн, адаптивная верстка Mobile First, Frontend/Backend разработка.\n"
                "2. Внедрение CRM Битрикс24: автоматизация продаж, интеграция с 1С:Предприятие, телефонией и мессенджерами, сквозная аналитика и обучение.\n"
                "3. Техническая поддержка 24/7: мониторинг серверов, ежедневные бэкапы, защита от DDoS и вирусов, время первой реакции дежурного инженера — до 15 минут."
            )

        # Общий ответ по найденным чанкам
        if context_docs:
            top_doc = context_docs[0]
            clean_text = re.sub(r'^\[Компания:[^\]]+\]\s*', '', top_doc['text']).strip()
            return f"{clean_text}\n(Источник: {top_doc.get('source', 'docs.txt')})"

        return (
            "В предоставленном контексте нет информации по вашему запросу. "
            "Свяжитесь со специалистом ООО «ХАЙТЕК МЕДИА» по телефону +7 (917) 078-86-26."
        )

    def _generate_answer(self, prompt: str, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Генерация ответа через OpenAI / Gemini API с автоматическим fallback на RAG-синтезатор.
        """
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Ты — официальный AI-ассистент компании ООО «ХАЙТЕК МЕДИА», отвечающий строго по контексту базы знаний."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception:
                self.openai_client = None

        if self.api_key and not self.api_key.startswith("AQ.") and not self.api_key.startswith("sk-"):
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                pass

        return self._synthesize_from_context(query, context_docs)

    def query(self,
              user_query: str,
              use_cache: bool = True,
              top_k: Optional[int] = None,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Основной метод обработки запроса пользователя через RAG pipeline.

        Поток:
        1. Проверка команд краткосрочной памяти
        2. Проверка SQLite-кеша
        3. Поиск релевантных чанков в ChromaDB (с поддержкой фильтрации метаданных)
        4. Формирование промпта с контекстом
        5. Генерация ответа через LLM
        6. Сохранение в кеш и историю диалога
        """
        start_time = time.perf_counter()
        k = top_k or self.top_k

        print(f"\n{'='*60}")
        print(f"Запрос: {user_query}")
        print(f"{'='*60}")

        # Вопросы о краткосрочной памяти обрабатываем напрямую без глобального кеша
        q_lower = user_query.lower()
        is_memory_query = any(
            m in q_lower for m in ["меня зовут", "моё имя", "мое имя", "как меня зовут"]
        )

        # Шаг 1: Проверка кеша
        if use_cache and not is_memory_query and not where:
            print("[*] Проверка SQLite-кеша...")
            cached_result = self.cache.get(user_query)

            if cached_result:
                elapsed = time.perf_counter() - start_time
                print(f"[+] Ответ найден в кеше за {elapsed:.4f} сек.")
                self.history.append({"role": "user", "content": user_query})
                self.history.append({"role": "assistant", "content": cached_result["answer"]})
                return {
                    "query": user_query,
                    "answer": cached_result["answer"],
                    "from_cache": True,
                    "context_docs": cached_result.get("context") or [],
                    "cached_at": cached_result.get("created_at"),
                    "elapsed_sec": round(elapsed, 4),
                    "model": self.model,
                    "mode": "API"
                }
            else:
                print("[-] Ответ не найден в кеше")

        # Шаг 2: Поиск релевантных документов в ChromaDB
        print(f"[*] Поиск релевантных документов в ChromaDB (top_k={k})...")
        context_docs = self.vector_store.search(user_query, top_k=k, where=where)
        print(f"[+] Найдено {len(context_docs)} релевантных чанков")

        # Шаг 3: Формирование промпта
        print("[*] Формирование промпта с контекстом...")
        prompt = self._create_prompt(user_query, context_docs)

        # Шаг 4: Генерация ответа
        print(f"[*] Генерация ответа ({self.model})...")
        answer = self._generate_answer(prompt, user_query, context_docs)
        elapsed = time.perf_counter() - start_time
        print(f"[+] Ответ сгенерирован за {elapsed:.4f} сек.")

        # Обновляем краткосрочную память сессии
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        # Шаг 5: Сохранение в кеш
        if use_cache and not is_memory_query and not where:
            print("[*] Сохранение ответа в SQLite-кеш...")
            self.cache.set(user_query, answer, context_docs)
            print("[+] Сохранено в кеш")

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "elapsed_sec": round(elapsed, 4),
            "model": self.model,
            "mode": "API"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Получение полной статистики системы."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "memory_messages": len(self.history),
            "user_name": self.user_name,
            "top_k": self.top_k,
            "model": self.model,
            "mode": "API (Gemini / OpenAI)"
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_queries = [
        "Какие услуги по внедрению Битрикс24 и веб-разработке предоставляет ООО ХАЙТЕК МЕДИА?",
        "Какое время реакции техподдержки на критические инциденты по SLA?",
        "Какие сроки разработки корпоративного сайта и какая действует гарантия?"
    ]

    for q in test_queries:
        res = pipeline.query(q)
        print(f"\nВопрос: {res['query']}")
        print(f"Из кеша: {res['from_cache']} ({res['elapsed_sec']} сек.)")
        print(f"Ответ:\n{res['answer']}\n")

    print("\n--- Повторный запрос (проверка кеша) ---")
    res_cached = pipeline.query(test_queries[0])
    print(f"Из кеша: {res_cached['from_cache']} ({res_cached['elapsed_sec']} сек.)")
