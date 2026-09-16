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
from db import search_vacancies, get_db_summary, get_market_analytics

load_dotenv()

console = Console()
ollama_client = OllamaClient(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

class UserIntent(BaseModel):
    intent: str = Field(default="chat", description="'search', 'analytics' или 'chat'")
    profession: Optional[str] = None
    grade: Optional[str] = None
    skills: List[str] = []
    is_remote: Optional[bool] = None
    keyword: Optional[str] = None

def build_system_prompt() -> str:
    summary = get_db_summary()
    total = summary.get("total", 0)
    date_range = summary.get("date_range_str", "нет данных")
    top_skills = summary.get("top_skills", [])
    top_skills_str = ", ".join(top_skills) if top_skills else "Python, DevOps, Frontend, Аналитика"
    recent_titles = summary.get("recent_titles", [])
    recent_str = "\n".join([f"  • {t}" for t in recent_titles[:6]]) if recent_titles else "  • Пока нет вакансий"

    return f"""Ты — умный, дружелюбный и профессиональный ИТ-карьерный консультант и персональный помощник по поиску работы.
У тебя есть доступ к актуальной локальной базе данных вакансий с ИТ-каналов Казахстана.

[АКТУАЛЬНОЕ СОСТОЯНИЕ ТВОЕЙ БАЗЫ ДАННЫХ (обновляется в реальном времени)]:
- Всего вакансий в базе: {total}
- Временной диапазон вакансий: {date_range}
- Популярный стек в базе: {top_skills_str}
- Примеры свежих вакансий:
{recent_str}

Твои правила:
1. Веди естественный, вежливый и живой диалог.
2. Если пользователь спрашивает о базе или датах (например: "сколько вакансий?", "за какую дату у тебя вакансии?", "за какой период база?"), используй актуальную информацию выше (в базе {total} вакансий, охватывают период: {date_range}).
3. Если пользователь задает вопросы о востребованности, частоте или трендах — опирайся строго на переданные цифры аналитики, называй точные числа и проценты.
4. Если переданы найденные вакансии — расскажи о них, выдели главное (дату публикации, стек, компанию, удаленку), укажи контакты и прямую ссылку на пост в Telegram (если она есть).
5. Отвечай кратко, емко и по делу, оформляя ответ аккуратными списками."""

NLU_PROMPT = """
Проанализируй сообщение пользователя и верни СТРОГО JSON:
{
    "intent": "search" | "analytics" | "chat",
    "profession": "роль в именительном падеже или null",
    "grade": "junior / middle / senior / lead или null",
    "skills": ["навык1", "навык2"],
    "is_remote": true / false / null,
    "keyword": "ключевое слово для анализа или null"
}

Правила классификации:
- intent: "analytics" — если вопрос касается ЧАСТОТЫ, ВОСТРЕБОВАННОСТИ, РЕЙТИНГА, КОЛИЧЕСТВА ПУБЛИКАЦИЙ, ТРЕНДОВ или СТАТИСТИКИ (например: "какая вакансия попадается чаще всего?", "что самое востребованное?", "сколько раз тебе попадались эти вакансии?", "какой стек самый частый?", "топ профессий", "статистика по рынку", "какие тренды по python", "какие еще есть популярные вакансии").
- intent: "search" — если пользователь прямо просит НАЙТИ/ПОКАЗАТЬ конкретные вакансии для отклика со ссылками и контактами (например: "найди вакансии python", "покажи удаленку", "есть вакансии для junior?", "вакансии за август").
- intent: "chat" — если это приветствие, вопрос о датах базы ("за какую дату вакансии?"), совет по резюме или продолжение разговора.

ВАЖНО ДЛЯ АНАЛИТИКИ:
- Если пользователь просит ОБЩИЙ рейтинг или топ ("какие еще есть популярные вакансии?", "топ вакансий вообще", "что еще востребовано?") — ставь "keyword": null (чтобы анализировать ВСЮ базу).
- НЕ превращай разговорные реплики и реакции ("это важно", "понятно", "подметил об удаленке") в ключевые слова для поиска. Если вопрос общий, ставь "keyword": null.

Примеры:
- "Привет, как дела?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "За какую дату у тебя вакансии?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Какая вакансия попадается чаще всего?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Это важно то что ты подметил об удаленке, какие еще есть топ самых популярных вакансий?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Что самое востребованное на рынке?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Сколько раз тебе попадались эти вакансии?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Какие навыки сейчас в топе?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Какие тренды по Python?" -> {"intent": "analytics", "profession": "Python", "grade": null, "skills": ["python"], "is_remote": null, "keyword": "python"}
- "Что есть по Python разработчикам?" -> {"intent": "search", "profession": "Python", "grade": null, "skills": ["python"], "is_remote": null, "keyword": null}
- "Ищу Middle DevOps на удаленку" -> {"intent": "search", "profession": "DevOps", "grade": "middle", "skills": ["devops"], "is_remote": true, "keyword": null}
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
        
        date_str = v.get('published_at').strftime('%d.%m.%Y') if v.get('published_at') else "Не указана"
        channel = v.get('channel')
        msg_id = v.get('message_id')
        link_str = f"https://t.me/{channel}/{msg_id}" if channel and msg_id else None
        link_line = f"\n  Ссылка: {link_str}" if link_str else ""

        lines.append(
            f"- **{v['title']}** в {company} ({location})\n"
            f"  Дата публикации: {date_str} | Грейд: {grade} | Удаленка: {remote}\n"
            f"  Стек: {skills}\n"
            f"  Контакты: {contacts_str}"
            f"{link_line}"
        )
    return "\n".join(lines)

def run_chat_turn(user_query: str, history: list) -> str:
    # 1. Формируем актуальный системный промпт со свежей статистикой базы
    system_prompt = build_system_prompt()
    
    # 2. Определяем намерение (search vs analytics vs chat)
    intent_data = parse_user_intent(user_query)
    
    extra_context = ""
    if intent_data.intent == "analytics":
        keyword = intent_data.keyword or intent_data.profession
        if keyword:
            stopwords_k = {'удаленка', 'удаленке', 'вакансия', 'вакансии', 'популярных', 'еще', 'рынок', 'рынка', 'самых'}
            if keyword.lower().strip() in stopwords_k:
                keyword = None

        stats = get_market_analytics(keyword=keyword)
        total_a = stats.get("total", 0)
        console.print(f"[dim green]Сформирована аналитика рынка по {total_a} вакансиям[/dim green]")
        
        roles_lines = [
            f"  {i+1}. {r['title']} — {r['count']} раз ({r['pct']}%)"
            for i, r in enumerate(stats.get("top_roles", [])[:12])
        ]
        skills_lines = [
            f"  • {s['skill']} — {s['count']} упоминаний ({s['pct']}%)"
            for s in stats.get("top_skills", [])[:15]
        ]
        comp_lines = [
            f"  • {c['company']} — {c['count']} вакансий"
            for c in stats.get("top_companies", [])[:8]
        ]

        roles_text = "\n".join(roles_lines) if roles_lines else "  Данных нет"
        skills_text = "\n".join(skills_lines) if skills_lines else "  Данных нет"
        comp_text = "\n".join(comp_lines) if comp_lines else "  Данных нет"

        extra_context = f"""

[ТОЧНАЯ СТАТИСТИКА И АНАЛИТИКА ИЗ БАЗЫ ДАННЫХ]:
Всего проанализировано вакансий: {total_a}
Период анализа: {stats.get('date_range_str')}
Удаленка: {stats.get('remote_pct')}% ({stats.get('remote_cnt')} вакансий из {total_a})

ТОП САМЫХ ЧАСТО ПУБЛИКУЕМЫХ ДОЛЖНОСТЕЙ:
{roles_text}

ТОП САМЫХ ВОСТРЕБОВАННЫХ НАВЫКОВ (СТЕК):
{skills_text}

ТОП НАНИМАЮЩИХ КОМПАНИЙ:
{comp_text}

(ОЧЕНЬ ВАЖНАЯ ИНСТРУКЦИЯ: Отвечай на вопрос пользователя строго по этим статистическим данным. Обязательно называй точные цифры и количество упоминаний. Не выдумывай вакансии от себя.)
"""
    elif intent_data.intent == "search":
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
