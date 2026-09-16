import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "vacancies_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vacancies (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            company TEXT,
            location TEXT,
            grade TEXT,
            is_remote BOOLEAN DEFAULT FALSE,
            skills JSONB DEFAULT '[]',
            contacts JSONB DEFAULT '{}',
            raw_text TEXT NOT NULL,
            channel TEXT,
            message_id BIGINT,
            published_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title, company, raw_text)
        );
        ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS channel TEXT;
        ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS message_id BIGINT;
        ALTER TABLE vacancies ADD COLUMN IF NOT EXISTS published_at TIMESTAMP;
        CREATE INDEX IF NOT EXISTS idx_vacancies_published_at ON vacancies (published_at DESC NULLS LAST);
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized with date and channel support.")

def clear_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE vacancies RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()
    print("Database cleared (0 vacancies).")

def get_db_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*), MIN(published_at), MAX(published_at) FROM vacancies;")
        row = cur.fetchone()
        total = row[0] or 0
        min_date = row[1]
        max_date = row[2]

        if min_date and max_date:
            date_range_str = f"с {min_date.strftime('%d.%m.%Y')} по {max_date.strftime('%d.%m.%Y')}"
        elif min_date:
            date_range_str = f"с {min_date.strftime('%d.%m.%Y')}"
        else:
            date_range_str = "даты публикаций еще не зафиксированы"

        cur.execute("SELECT title, company, published_at FROM vacancies ORDER BY published_at DESC NULLS LAST, created_at DESC LIMIT 8;")
        rows = cur.fetchall()
        recent = []
        for r in rows:
            date_part = f" [{r[2].strftime('%d.%m')}]" if r[2] else ""
            comp_part = f" ({r[1]})" if r[1] else ""
            recent.append(f"{r[0]}{comp_part}{date_part}")

        cur.execute("""
            SELECT skill FROM (
                SELECT jsonb_array_elements_text(skills) as skill 
                FROM vacancies
            ) s 
            WHERE length(skill) > 2
            GROUP BY skill 
            ORDER BY count(*) DESC 
            LIMIT 7;
        """)
        top_skills = [r[0] for r in cur.fetchall()]

        return {
            "total": total,
            "date_range_str": date_range_str,
            "min_date": min_date,
            "max_date": max_date,
            "recent_titles": recent,
            "top_skills": top_skills
        }
    except Exception as e:
        print(f"Error fetching db summary: {e}")
        return {"total": 0, "date_range_str": "нет данных", "recent_titles": [], "top_skills": []}
    finally:
        cur.close()
        conn.close()

def get_market_analytics(since=None, until=None, keyword=None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        where_clauses = ["1=1"]
        args = []

        if since:
            where_clauses.append("published_at >= %s")
            args.append(since)
        if until:
            where_clauses.append("published_at <= %s")
            args.append(until)
        if keyword:
            where_clauses.append("(title ILIKE %s OR raw_text ILIKE %s)")
            args.append(f"%{keyword}%")
            args.append(f"%{keyword}%")

        where_sql = " AND ".join(where_clauses)

        # 1. Всего вакансий
        cur.execute(f"SELECT COUNT(*) FROM vacancies WHERE {where_sql};", args)
        total = cur.fetchone()[0] or 0
        if total == 0:
            return {"total": 0, "top_roles": [], "top_skills": [], "top_companies": []}

        # 2. Период
        cur.execute(f"SELECT MIN(published_at), MAX(published_at) FROM vacancies WHERE {where_sql};", args)
        drow = cur.fetchone()
        min_date = drow[0]
        max_date = drow[1]
        date_str = f"с {min_date.strftime('%d.%m.%Y')} по {max_date.strftime('%d.%m.%Y')}" if min_date and max_date else "нет данных"

        # 3. Топ-15 самых частых позиций
        cur.execute(f"""
            SELECT title, COUNT(*) as cnt 
            FROM vacancies 
            WHERE {where_sql} AND length(title) > 2
            GROUP BY title 
            ORDER BY cnt DESC 
            LIMIT 15;
        """, args)
        top_roles = [
            {"title": r[0], "count": r[1], "pct": round(r[1] / total * 100, 1)}
            for r in cur.fetchall()
        ]

        # 4. Топ-20 самых востребованных навыков
        cur.execute(f"""
            SELECT skill, COUNT(*) as cnt
            FROM (
                SELECT jsonb_array_elements_text(skills) as skill 
                FROM vacancies 
                WHERE {where_sql}
            ) s
            WHERE length(skill) > 1
            GROUP BY skill
            ORDER BY cnt DESC
            LIMIT 20;
        """, args)
        top_skills = [
            {"skill": r[0], "count": r[1], "pct": round(r[1] / total * 100, 1)}
            for r in cur.fetchall()
        ]

        # 5. Топ-10 нанимающих компаний
        cur.execute(f"""
            SELECT company, COUNT(*) as cnt
            FROM vacancies
            WHERE {where_sql} AND company IS NOT NULL AND company != '' AND company NOT ILIKE '%%не указана%%'
            GROUP BY company
            ORDER BY cnt DESC
            LIMIT 10;
        """, args)
        top_companies = [
            {"company": r[0], "count": r[1]}
            for r in cur.fetchall()
        ]

        # 6. Удаленка vs Офис
        cur.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE is_remote = true) as remote_cnt,
                COUNT(*) FILTER (WHERE is_remote = false) as office_cnt
            FROM vacancies
            WHERE {where_sql};
        """, args)
        rem_row = cur.fetchone()
        remote_cnt = rem_row[0] or 0
        office_cnt = rem_row[1] or 0

        return {
            "total": total,
            "date_range_str": date_str,
            "top_roles": top_roles,
            "top_skills": top_skills,
            "top_companies": top_companies,
            "remote_cnt": remote_cnt,
            "office_cnt": office_cnt,
            "remote_pct": round(remote_cnt / total * 100, 1) if total else 0
        }
    except Exception as e:
        print(f"Error getting market analytics: {e}")
        return {"total": 0}
    finally:
        cur.close()
        conn.close()

def vacancy_exists(raw_text: str) -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM vacancies WHERE raw_text = %s LIMIT 1;", (raw_text,))
        return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking vacancy existence: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def save_vacancy(data):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        skills_normalized = [s.strip().lower() for s in data.get('skills', []) if isinstance(s, str) and s.strip()]
        cur.execute("""
            INSERT INTO vacancies (title, company, location, grade, is_remote, skills, contacts, raw_text, channel, message_id, published_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (title, company, raw_text) DO NOTHING
            RETURNING id;
        """, (
            data.get('title') or 'Вакансия',
            data.get('company'),
            data.get('location'),
            data.get('grade'),
            bool(data.get('is_remote', False)),
            json.dumps(skills_normalized),
            json.dumps(data.get('contacts') or {}),
            data.get('raw_text'),
            data.get('channel'),
            data.get('message_id'),
            data.get('published_at')
        ))
        result = cur.fetchone()
        conn.commit()
        return result[0] if result else None
    except Exception as e:
        print(f"Error saving vacancy: {e}")
        conn.rollback()
        return None
    finally:
        cur.close()
        conn.close()

def search_vacancies(params):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = "SELECT * FROM vacancies WHERE 1=1"
    args = []
    
    if params.get('grade'):
        query += " AND grade ILIKE %s"
        args.append(f"%{params['grade']}%")
    
    if params.get('is_remote') is not None:
        query += " AND is_remote = %s"
        args.append(params['is_remote'])
        
    if params.get('skills'):
        for skill in params['skills']:
            if isinstance(skill, str) and skill.strip():
                clean_skill = skill.strip().lower()
                query += " AND (skills @> %s OR raw_text ILIKE %s)"
                args.append(json.dumps([clean_skill]))
                args.append(f"%{clean_skill}%")

    if params.get('profession'):
        stopwords = {'работа', 'вакансия', 'ищу', 'нужна', 'нужен', 'разработчиком', 'специалист'}
        words = [w.strip() for w in params['profession'].split() if len(w.strip()) > 2 and w.strip().lower() not in stopwords]
        if words:
            word_clauses = []
            for w in words:
                word_clauses.append("(title ILIKE %s OR raw_text ILIKE %s)")
                args.append(f"%{w}%")
                args.append(f"%{w}%")
            query += f" AND ({' OR '.join(word_clauses)})"

    if params.get('since'):
        query += " AND published_at >= %s"
        args.append(params['since'])

    if params.get('until'):
        query += " AND published_at <= %s"
        args.append(params['until'])

    query += " ORDER BY published_at DESC NULLS LAST, created_at DESC LIMIT 15"
    
    cur.execute(query, args)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

if __name__ == "__main__":
    init_db()
