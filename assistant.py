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

    return f"""Ты — опытный, доброжелательный и внимательный IT-карьерный консультант и ментор.
У тебя есть доступ к базе данных реальных IT-вакансий Казахстана (актуальные публикации с ведущих каналов: Астана, Алматы, удаленка).

[ОБЗОР РЫНКА И ТВОЕЙ БАЗЫ ДАННЫХ]:
- Всего вакансий в базе: {total} (период публикаций: {date_range})
- Ключевые направления на рынке (реальная картина спроса):
  1. Разработка ПО (Backend, Frontend, Fullstack, Mobile) — ~32% (абсолютный лидер рынка!)
  2. Данные и Аналитика (Data/BI/ML, Системный/Бизнес-анализ) — ~23%
  3. Менеджмент и Продукт (Project/Product Manager, Team Lead) — ~15%
  4. Инфраструктура и Поддержка (DevOps, SysAdmin, Support, Helpdesk) — ~12%
  5. Тестирование (QA Manual/Auto) — ~7%
  6. Дизайн (UI/UX) — ~4%
- География: Астана (~36%) и Алматы (~34%) — два ключевых IT-хаба (вместе более 70% всех вакансий страны), Удаленная работа — ~15%, другие города — ~15%.
- Самый востребованный стек: {top_skills_str}
- Примеры недавних вакансий:
{recent_str}

Твои принципы общения (стиль IT-ментора):
1. Будь живым, вовлеченным и чутким собеседником. Общайся на равных, дружелюбно, как опытный коллега и наставник. Не будь сухим калькулятором!
2. Концептуальные вопросы и суть профессий:
   - Если пользователь спрашивает "Кто такой X?", "Чем занимается Y?", "Что делает DevOps/QA/Сисадмин?", "Стоит ли учить Python?":
   - Объясни суть простым, человеческим и образным языком (какую пользу приносит специалист бизнесу, чем наполнен его день, с кем взаимодействует).
   - Приведи 1-2 наглядных примера или полезный совет от ментора.
   - КАТЕГОРИЧЕСКИ НЕ выгружай сухие статистические таблицы или списки вакансий с номерами и процентами, если тебя прямо не просили о цифрах!
3. Рассуждения и разговоры о рынке:
   - Если пользователь высказывает гипотезы или мысли (например, "Все вакансии наверное в Астане и Алматы, да?"):
   - Поддержи беседу, подтверди или скорректируй мысль реальными фактами из базы (да, действительно, более 70% рынка сосредоточено в Астане и Алматы, плюс 15% удаленка), объясни специфику рынка Казахстана.
4. Оптическая иллюзия названий вакансий:
   - Если заходит речь о востребованности: помни, что программисты (разработчики) занимают первое место по реальному спросу (>32-50%), но из-за десятков разных названий ("Backend Developer", "Python разработчик", "Frontend", "iOS") в единичных названиях может лидировать "Project Manager" или "Системный аналитик". Объясняй это пользователю, опираясь на направления!
5. Если в контексте переданы результаты поиска вакансий — презентуй их аккуратно, выделяя стек, вилку, компанию и прямую ссылку на Telegram.
6. Отвечай структурированно, красиво и емко."""

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
- intent: "chat" — ОБЯЗАТЕЛЬНО выбирай "chat" для:
  1. Вопросов о сути профессии или технологии ("кто такой системный администратор?", "кто такие аналитики?", "чем занимается проджект-менеджер?", "что делает devops?", "расскажи про QA").
  2. Рассуждений, мнений и бесед о рынке ("все вакансии наверное в Астане и Алматы да?", "правда ли что удаленка уходит?", "почему так мало вакансий?").
  3. Вопросов о базе, периоде, приветствий, советов по резюме ("привет", "за какую дату база?").
- intent: "analytics" — если вопрос касается СТАТИСТИКИ, ЧАСТОТЫ, РЕЙТИНГА, ТОПА, ВОСТРЕБОВАННОСТИ или НАПРАВЛЕНИЙ ("какая вакансия самая частая?", "топ направлений", "что востребовано на рынке?", "сколько вакансий на системного администратора?").
  * Если вопрос ОБЩИЙ по рынку ("топ вакансий", "что самое востребованное?", "какие направления популярны?", "самые частые вакансии") — ставь keyword: null.
  * Если вопрос про конкретную специальность ("сколько вакансий на Java?") — ставь keyword: "Java".
- intent: "search" — если пользователь прямо просит НАЙТИ/ПОКАЗАТЬ вакансии для отклика ("найди python разработчик", "покажи вакансии в Алматы").

Поле intent ОБЯЗАТЕЛЬНО должно быть строкой: "chat", "analytics" или "search" (никогда не null!).

Примеры:
- "Привет, как дела?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Кто такие системные администраторы?" -> {"intent": "chat", "profession": "системный администратор", "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Чем занимается проджект менеджер?" -> {"intent": "chat", "profession": "проджект менеджер", "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Все вакансии наверное в Астане и Алматы да?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "За какую дату у тебя вакансии?" -> {"intent": "chat", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Какая вакансия попадается чаще всего?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Какие самые частые направления вакансий?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "А че реально самые частые вакансий это проджект менеджеры?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Что самое востребованное на рынке?" -> {"intent": "analytics", "profession": null, "grade": null, "skills": [], "is_remote": null, "keyword": null}
- "Сколько вакансий на системного аналитика?" -> {"intent": "analytics", "profession": "системный аналитик", "grade": null, "skills": [], "is_remote": null, "keyword": "системный аналитик"}
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
        if not data.get("intent") or data["intent"] not in ["search", "analytics", "chat"]:
            data["intent"] = "chat"
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
            stopwords_k = {
                'удаленка', 'удаленке', 'вакансия', 'вакансии', 'популярных', 'популярные',
                'еще', 'рынок', 'рынка', 'самых', 'частые', 'частых', 'направления', 'направлений',
                'востребованные', 'востребованности', 'топ', 'профессий', 'профессии'
            }
            if keyword.lower().strip() in stopwords_k:
                keyword = None

        stats = get_market_analytics(keyword=keyword)
        total_a = stats.get("total", 0)
        console.print(f"[dim green]Сформирована аналитика рынка по {total_a} вакансиям[/dim green]")
        
        cat_lines = [
            f"  {i+1}. {c['category']} — {c['count']} вакансий ({c['pct']}%)"
            for i, c in enumerate(stats.get("categories", []))
        ]
        loc_lines = [
            f"  • {l['city']} — {l['count']} вакансий ({l['pct']}%)"
            for l in stats.get("locations", [])
        ]
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

        cat_text = "\n".join(cat_lines) if cat_lines else "  Данных нет"
        loc_text = "\n".join(loc_lines) if loc_lines else "  Данных нет"
        roles_text = "\n".join(roles_lines) if roles_lines else "  Данных нет"
        skills_text = "\n".join(skills_lines) if skills_lines else "  Данных нет"
        comp_text = "\n".join(comp_lines) if comp_lines else "  Данных нет"

        extra_context = f"""

[ТОЧНАЯ АНАЛИТИКА ИЗ БАЗЫ ДАННЫХ ДЛЯ ОТВЕТА]:
Всего проанализировано вакансий: {total_a}
Период анализа: {stats.get('date_range_str')}
Удаленная работа: {stats.get('remote_pct')}% ({stats.get('remote_cnt')} вакансий из {total_a})

РАСПРЕДЕЛЕНИЕ ПО СФЕРАМ / НАПРАВЛЕНИЯМ (РЕАЛЬНЫЙ СПРОС):
{cat_text}

ГЕОГРАФИЯ (ГОРОДА И РЕГИОНЫ):
{loc_text}

ТОП ЧАСТЫХ НАЗВАНИЙ ДОЛЖНОСТЕЙ В ОБЪЯВЛЕНИЯХ:
{roles_text}

ТОП САМЫХ ВОСТРЕБОВАННЫХ НАВЫКОВ (СТЕК):
{skills_text}

ТОП НАНИМАЮЩИХ КОМПАНИЙ:
{comp_text}

(ИНСТРУКЦИЯ КАРЬЕРНОМУ МЕНТОРУ:
1. При вопросах о востребованности и самых популярных профессиях/направлениях ориентируйся В ПЕРВУЮ ОЧЕРЕДЬ на "РАСПРЕДЕЛЕНИЕ ПО СФЕРАМ/НАПРАВЛЕНИЯМ", где лидирует Разработка ПО.
2. Не вываливай сухую сплошную простыню таблиц, если пользователь спрашивает общие выводы — объясни тенденции простыми живыми словами и подкрепи ключевыми цифрами.
3. Опирайся строго на предоставленные факты из базы.)
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
