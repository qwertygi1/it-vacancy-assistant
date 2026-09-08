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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(title, company, raw_text)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized.")

def get_db_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM vacancies;")
        total = cur.fetchone()[0]

        cur.execute("SELECT title, company FROM vacancies ORDER BY created_at DESC LIMIT 8;")
        rows = cur.fetchall()
        recent = [f"{r[0]}" + (f" ({r[1]})" if r[1] else "") for r in rows]

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
            "recent_titles": recent,
            "top_skills": top_skills
        }
    except Exception as e:
        print(f"Error fetching db summary: {e}")
        return {"total": 0, "recent_titles": [], "top_skills": []}
    finally:
        cur.close()
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
            INSERT INTO vacancies (title, company, location, grade, is_remote, skills, contacts, raw_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
            data.get('raw_text')
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

    query += " ORDER BY created_at DESC LIMIT 10"
    
    cur.execute(query, args)
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

if __name__ == "__main__":
    init_db()
