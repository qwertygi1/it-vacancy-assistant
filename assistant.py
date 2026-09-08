import os
import sys
import json
from typing import List, Optional
from pydantic import BaseModel, Field

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ollama import Client as OllamaClient
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from dotenv import load_dotenv
from db import search_vacancies, get_db_summary

load_dotenv()

console = Console()
ollama_client = OllamaClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

class UserIntent(BaseModel):
    intent: str = Field(default="chat", description="'search' или 'chat'")
    profession: Optional[str] = None
    grade: Optional[str] = None
    skills: List[str] = []
    is_remote: Optional[bool] = None

def build_system_prompt() -> str:
    summary = get_db_summary()
    total = summary.get("total", 0)
    top_skills = summary.get("top_skills", [])
    top_skills_str = ", ".join(top_skills) if top_skills else "Python, DevOps, Frontend, Аналитика"
    recent_titles = summary.get("recent_titles", [])
    recent_str = "\n".join([f"  • {t}" for t in recent_titles[:6]]) if recent_titles else "  • Пока нет вакансий"

    return f"""Ты — умный, дружелюбный и профессиональный ИТ-карьерный консультант и персональный помощник по поиску работы.
У тебя есть доступ к актуальной локальной базе данных вакансий с ИТ-каналов Казахстана.

[АКТУАЛЬНОЕ СОСТОЯНИЕ ТВОЕЙ БАЗЫ ДАННЫХ (обновляется в реальном времени)]:
- Всего вакансий в базе: {total}
- Популярный стек в базе: {top_skills_str}
- Примеры свежих вакансий:
{recent_str}

Твои правила:
1. Веди естественный, вежливый и живой диалог.
2. Если пользователь спрашивает о базе (например: "сколько вакансий?", "что у тебя в базе?", "какие есть специальности?"), используй актуальную информацию выше (в базе ровно {total} вакансий).
3. Если пользователь задает общие вопросы о карьере (составление резюме, подготовка к собеседованиям, выбор направления) — давай качественные экспертные советы.
4. Если переданы найденные вакансии — расскажи о них, выдели главное (стек, компанию, удаленку) и обязательно укажи контакты для отклика.
5. Отвечай кратко, емко и по делу, оформляя ответ аккуратными списками."""

NLU_PROMPT = """
Проанализируй сообщение пользователя и верни СТРОГО JSON:
{
    "intent": "search" или "chat",
    "profession": "роль в именительном падеже (например Python, DevOps, Аналитик, Дизайнер, Fullstack) или null",
    "grade": "junior / middle / senior / lead или null",
    "skills": ["навык1", "навык2"],
    "is_remote": true / false / null
}

Правила:
- intent: "search" — ТОЛЬКО если пользователь прямо ищет или просит показать вакансии по специальности/стеку/удаленке (например "ищу python", "есть вакансии для junior?", "покажи удаленку").
- intent: "chat" — если это приветствие, вопрос о количестве вакансий в базе ("сколько вакансий?", "что в базе?"), совет по резюме, вопрос о подготовке к собеседованию или продолжение диалога.

Примеры:
- "Привет, как дела?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null}
- "Сколько у тебя вакансий в базе?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null}
- "Что есть по Python разработчикам?" -> {"intent": "search", "profession": "Python", "grade": null, "skills": ["python"], "is_remote": null}
- "Ищу Middle DevOps на удаленку" -> {"intent": "search", "profession": "DevOps", "grade": "middle", "skills": ["devops"], "is_remote": true}
"""

def parse_user_intent(user_query: str) -> UserIntent:
    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': NLU_PROMPT},
                {'role': 'user', 'content': user_query}
            ],
            format='json'
        )
        data = json.loads(response['message']['content'])
        return UserIntent(**data)
    except Exception as e:
        console.print(f"[dim red]Ошибка NLU: {e}[/dim red]")
        return UserIntent(intent="chat")

def format_vacancies_context(vacancies: list) -> str:
    if not vacancies:
        return "По данному запросу подходящих вакансий в базе не найдено."

    lines = []
    for v in vacancies:
        company = v.get('company') or "Компания не указана"
        location = v.get('location') or "Не указано"
        grade = v.get('grade') or "Не указан"
        skills = ", ".join(v.get('skills') or []) or "Не указаны"
        remote = "Да" if v.get('is_remote') else "Нет"
        contacts_dict = v.get('contacts') or {}
        contacts_str = ", ".join([f"{k}: {val}" for k, val in contacts_dict.items() if val]) or "Не указаны"
        lines.append(
            f"- **{v['title']}** в {company} ({location})\n"
            f"  Грейд: {grade} | Удаленка: {remote}\n"
            f"  Стек: {skills}\n"
            f"  Контакты: {contacts_str}"
        )
    return "\n".join(lines)

def run_chat_turn(user_query: str, history: list) -> str:
    # 1. Формируем актуальный системный промпт со свежей статистикой базы
    system_prompt = build_system_prompt()
    
    # 2. Определяем намерение (search vs chat)
    intent_data = parse_user_intent(user_query)
    
    extra_context = ""
    if intent_data.intent == "search":
        params = {
            "profession": intent_data.profession,
            "grade": intent_data.grade,
            "skills": intent_data.skills,
            "is_remote": intent_data.is_remote
        }
        results = search_vacancies(params)
        console.print(f"[dim blue]Найдено вакансий в БД: {len(results)}[/dim blue]")
        
        if results:
            vacancies_text = format_vacancies_context(results)
            extra_context = f"\n\n[РЕЗУЛЬТАТЫ ПОИСКА В БАЗЕ ДАННЫХ]:\n{vacancies_text}\n(Используй эти данные, чтобы подробно ответить пользователю и дать контакты)."
        else:
            extra_context = "\n\n[РЕЗУЛЬТАТЫ ПОИСКА В БАЗЕ ДАННЫХ]: По заданным фильтрам точных совпадений не найдено. Сообщи об этом и предложи посмотреть смежные направления."
    
    # 3. Собираем сообщения для модели с историей диалога
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(history[-6:])  # последние 6 реплик для контекста
    
    current_message = user_query + extra_context if extra_context else user_query
    messages.append({'role': 'user', 'content': current_message})
    
    # 4. Стриминг ответа
    try:
        response = ollama_client.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in response:
            content = chunk['message']['content']
            full_response += content
            console.print(content, end="")
        console.print()
        
        # Сохраняем в историю диалога
        history.append({'role': 'user', 'content': user_query})
        history.append({'role': 'assistant', 'content': full_response})
        return full_response
    except Exception as e:
        err = f"Ошибка генерации: {e}"
        console.print(f"[red]{err}[/red]")
        return err

def main():
    console.print(Panel.fit(
        "[bold cyan]IT Vacancy Assistant[/bold cyan]\n[italic]Умный карьерный консультант по ИТ-рынку Казахстана[/italic]",
        border_style="cyan"
    ))
    
    # Показываем текущее состояние базы при входе
    summary = get_db_summary()
    console.print(f"[green]База данных подключена. Доступно вакансий: {summary['total']}[/green]\n")
    
    history = []
    
    while True:
        try:
            user_input = Prompt.ask("[bold green]Вы[/bold green] (или 'exit' для выхода)")
        except (KeyboardInterrupt, EOFError):
            break
            
        if not user_input.strip():
            continue
            
        if user_input.lower() in ['exit', 'quit', 'выход']:
            console.print("[yellow]До свидания! Удачи в поиске работы![/yellow]")
            break
            
        console.print("\n[bold magenta]Ассистент:[/bold magenta]")
        run_chat_turn(user_input, history)
        console.print()

if __name__ == "__main__":
    main()
