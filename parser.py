import os
import sys
import asyncio
import json
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pydantic import BaseModel, Field
from telethon import TelegramClient
from ollama import Client as OllamaClient
from dotenv import load_dotenv
from db import save_vacancy, init_db, vacancy_exists, clear_db

load_dotenv()

# Схема для валидации данных от LLM
class VacancySchema(BaseModel):
    title: Optional[str] = Field(default=None, description="Название должности")
    company: Optional[str] = Field(default=None, description="Название компании")
    location: Optional[str] = Field(default=None, description="Город или страна")
    grade: Optional[str] = Field(default=None, description="Грейд: junior, middle, senior или lead")
    is_remote: Optional[bool] = Field(default=False, description="Удаленная ли работа")
    skills: Optional[List[str]] = Field(default_factory=list, description="Список ключевых навыков")
    contacts: Optional[Dict[str, Optional[str]]] = Field(default_factory=dict, description="Контакты: telegram, phone, email")
    is_vacancy: Optional[bool] = Field(default=True, description="Является ли сообщение вакансией")

# Инициализация клиентов
api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
api_hash = os.getenv("TELEGRAM_API_HASH")
raw_channels = os.getenv("TELEGRAM_CHANNELS", "").split(",")
ollama_client = OllamaClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """
Ты — эксперт по анализу ИТ-вакансий. Твоя задача — извлечь структурированную информацию из текста вакансии.
Отвечай СТРОГО в формате JSON, соответствующем следующей схеме:
{
    "title": "название должности или профессии (например, Python Developer)",
    "company": "название компании или null",
    "location": "город/страна или null",
    "grade": "junior, middle, senior или lead или null",
    "is_remote": true/false или null,
    "skills": ["skill1", "skill2"],
    "contacts": {"telegram": "@user", "email": "...", "phone": "..."},
    "is_vacancy": true
}
Если это НЕ вакансия (например: реклама курсов, резюме кандидата, анонс мероприятия, новость), верни {"is_vacancy": false}.
Если поле не найдено, используй null или пустой список/объект.
"""

def clean_channel_name(channel: str) -> str:
    c = channel.strip()
    c = c.replace("https://t.me/", "").replace("http://t.me/", "").replace("t.me/", "")
    c = c.lstrip("@").rstrip("/")
    return c

async def process_message(message_text: str):
    if not message_text or len(message_text.strip()) < 50:
        return None
    
    try:
        # Вызов Ollama в отдельном потоке, чтобы не блокировать event loop Telethon
        response = await asyncio.to_thread(
            ollama_client.chat,
            model=model_name,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Извлеки данные из этой вакансии:\n\n{message_text}"}
            ],
            format='json'
        )
        
        data = json.loads(response['message']['content'])
        if not data.get("is_vacancy", True):
            return None

        vacancy = VacancySchema(**data)
        vacancy_dict = vacancy.model_dump()
        
        # Гарантируем обязательные поля для БД
        if not vacancy_dict.get('title'):
            first_line = message_text.strip().split('\n')[0][:80]
            vacancy_dict['title'] = first_line or "IT Вакансия"
            
        if vacancy_dict.get('contacts') is None:
            vacancy_dict['contacts'] = {}
        if vacancy_dict.get('skills') is None:
            vacancy_dict['skills'] = []
        if vacancy_dict.get('is_remote') is None:
            vacancy_dict['is_remote'] = False
            
        vacancy_dict['raw_text'] = message_text
        return vacancy_dict
    except Exception as e:
        print(f"Ошибка при обработке через Ollama: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Сборщик вакансий из Telegram-каналов в базу PostgreSQL")
    parser.add_argument("--limit", type=int, default=None, help="Лимит сообщений на канал (по умолчанию 10, если не задан --since)")
    parser.add_argument("--since", type=str, default=None, help="Дата начала (с какого числа) в формате YYYY-MM-DD (например 2026-06-01)")
    parser.add_argument("--until", type=str, default=None, help="Дата окончания (по какое число) в формате YYYY-MM-DD (например 2026-08-31)")
    parser.add_argument("--channel", type=str, default=None, help="Собрать только из одного указанного канала")
    parser.add_argument("--clear-db", action="store_true", help="Очистить базу данных перед сбором (начать начисто)")
    
    args = parser.parse_args()
    
    # Если дата не указана и лимит не передан — берем дефолтный лимит 10
    limit = args.limit
    if limit is None and not args.since:
        limit = int(os.getenv("MESSAGES_LIMIT", "10"))
    
    since_date = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[Ошибка] Неверный формат даты --since: {args.since}. Используйте YYYY-MM-DD")
            return

    until_date = None
    if args.until:
        try:
            # Конец указанного дня: 23:59:59
            until_date = datetime.strptime(args.until + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[Ошибка] Неверный формат даты --until: {args.until}. Используйте YYYY-MM-DD")
            return

    init_db()

    if args.clear_db:
        clear_db()
    
    if args.channel:
        channels = [clean_channel_name(args.channel)]
    else:
        channels = [clean_channel_name(ch) for ch in raw_channels if clean_channel_name(ch)]
    
    dates_info = []
    if since_date:
        dates_info.append(f"с {args.since}")
    if until_date:
        dates_info.append(f"по {args.until}")
    limit_info = " ".join(dates_info) if dates_info else f"лимит {limit} сообщений на канал"
    
    print(f"Всего каналов: {len(channels)} -> {channels}")
    print(f"Режим сбора: {limit_info}\n")
    
    async with TelegramClient('session_name', api_id, api_hash) as client:
        for channel in channels:
            print(f"[Канал] Обработка: @{channel}...")
            try:
                entity = await client.get_entity(channel)
                count_saved = 0
                count_existing = 0
                count_skipped = 0
                
                async for message in client.iter_messages(entity, limit=limit):
                    # Проверка даты окончания (если сообщение новее, чем until — пропускаем)
                    if until_date and message.date and message.date > until_date:
                        continue

                    # Проверка даты начала (если сообщение старше, чем since — останавливаем чтение канала)
                    if since_date and message.date and message.date < since_date:
                        print(f"  [i] Достигнута граница даты ({message.date.strftime('%Y-%m-%d')} < {args.since}), завершаем канал.")
                        break
                        
                    # Проверка на наличие текста
                    if not message.text or len(message.text.strip()) < 50:
                        count_skipped += 1
                        continue
                        
                    # Быстрая проверка: если уже есть в БД — пропускаем БЕЗ вызова LLM
                    if vacancy_exists(message.text):
                        count_existing += 1
                        continue
                    
                    # Извлечение через Ollama
                    vacancy_data = await process_message(message.text)
                    if vacancy_data:
                        vacancy_data['channel'] = channel
                        vacancy_data['message_id'] = message.id
                        vacancy_data['published_at'] = message.date.replace(tzinfo=None) if message.date else None

                        vacancy_id = save_vacancy(vacancy_data)
                        if vacancy_id:
                            count_saved += 1
                            date_str = message.date.strftime('%d.%m.%Y') if message.date else 'н/д'
                            print(f"  [+] [{vacancy_id}] ({date_str}) {vacancy_data.get('title')} ({vacancy_data.get('company') or 'Компания не указана'})")
                        else:
                            count_existing += 1
                    else:
                        count_skipped += 1
                        
                print(f"Итог по @{channel}: новых сохранено: {count_saved}, уже было в базе: {count_existing}, пропущено: {count_skipped}.\n")
            except Exception as e:
                print(f"Ошибка при чтении канала @{channel}: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
