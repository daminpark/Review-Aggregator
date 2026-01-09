import sqlite3
import json
import pandas as pd
import os

DB_FILE = "reviews.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def init_db():
    """Initializes the database tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()
    
    # Listings Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS listings (
            entity_id TEXT PRIMARY KEY,
            nickname TEXT
        )
    ''')
    
    # Reviews Table
    # review_id is the primary key (reviewsReceived[X].review.reviewId)
    # entity_id matches listings.entity_id
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            entity_id TEXT,
            submitted_at DATETIME,
            comment TEXT,
            private_feedback TEXT,
            full_text TEXT,
            positive_points TEXT, -- Stored as JSON string
            negative_points TEXT, -- Stored as JSON string
            rating_overall INTEGER,
            rating_cleanliness INTEGER,
            rating_communication INTEGER,
            rating_checkin INTEGER,
            rating_accuracy INTEGER,
            rating_value INTEGER,
            rating_location INTEGER,
            comment_english TEXT,
            private_feedback_english TEXT
        )
    ''')
    
    # Config Table (for API Key etc)
    c.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # Messages Table (for Guest/Host communication analysis)
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT,
            sender_id TEXT,
            text TEXT,
            timestamp DATETIME,
            analysis_json TEXT, -- Full analysis
            category TEXT,      -- Extracted category (e.g. Complaint, Question)
            action_required INTEGER DEFAULT 0, -- 0 or 1
            text_english TEXT,   -- Translated to English
            is_ambiguous INTEGER DEFAULT 0, -- 0 or 1
            ambiguity_reason TEXT           -- Reason if ambiguous
        )
    ''')
    
    # MIGRATION: Check if text_english exists in messages (for existing DBs)
    try:
        c.execute("SELECT text_english FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding text_english to messages table...")
        c.execute("ALTER TABLE messages ADD COLUMN text_english TEXT")

    # MIGRATION: Check if is_ambiguous exists
    try:
        c.execute("SELECT is_ambiguous FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding ambiguity columns to messages table...")
        c.execute("ALTER TABLE messages ADD COLUMN is_ambiguous INTEGER DEFAULT 0")
        c.execute("ALTER TABLE messages ADD COLUMN ambiguity_reason TEXT")
    
    # Phase 2: Thread Incidents Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS thread_incidents (
            thread_id TEXT PRIMARY KEY,
            category TEXT,
            severity_score INTEGER,
            summary TEXT,
            message_count INTEGER,
            status TEXT DEFAULT 'Open'
        )
    ''')
    
    
    # Phase 2b: Hierarchical Categorization Columns
    try:
        c.execute("SELECT topic FROM messages LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating DB: Adding topic/sub_issue columns...")
        c.execute("ALTER TABLE messages ADD COLUMN topic TEXT")
        c.execute("ALTER TABLE messages ADD COLUMN sub_issue TEXT")
        # Enhance Phase 2: Automation Type
        try:
             c.execute("SELECT automation_type FROM messages LIMIT 1")
        except sqlite3.OperationalError:
             c.execute("ALTER TABLE messages ADD COLUMN automation_type TEXT")
        
    
    # Phase 3: Solution Proposals Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS solution_proposals (
            sub_issue TEXT PRIMARY KEY,
            topic TEXT,
            analysis_json TEXT,
            automation_score INTEGER,
            estimated_savings INTEGER
        )
    ''')

    conn.commit()
    conn.close()

def clear_db():
    """Deletes all data from the database, but PRESERVES the API Key."""
    # 1. Save API Key
    api_key = get_api_key()
    
    # 2. Delete DB File
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    # 3. Re-init DB
    init_db()
    
    # 4. Restore API Key
    if api_key:
        set_api_key(api_key)

def clear_message_analysis():
    """Clears ONLY the analysis columns for messages, resetting them to processed=False state."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE messages 
        SET analysis_json = NULL, category = NULL, action_required = 0, is_ambiguous = 0, ambiguity_reason = NULL
    ''')
    conn.commit()
    conn.close()

def clear_message_translations():
    """Clears ONLY the translation column for messages."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE messages SET text_english = NULL")
    conn.commit()
    conn.close()

def upsert_listings(listings_data):
    """
    Upserts listings into the database.
    listings_data: List of dicts with {'entity_id', 'nickname'}
    """
    if not listings_data:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    for item in listings_data:
        c.execute('''
            INSERT INTO listings (entity_id, nickname)
            VALUES (?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET nickname=excluded.nickname
        ''', (item['entity_id'], item['nickname']))
        
    conn.commit()
    conn.close()

def insert_reviews_ignore_duplicates(reviews_df):
    """
    Inserts reviews from a dataframe. Skips if review_id already exists.
    Returns the count of NEW reviews added.
    """
    if reviews_df.empty:
        return 0
        
    conn = get_connection()
    c = conn.cursor()
    
    new_count = 0
    
    # Expected columns in reviews_df:
    # review_id, entity_id, submittedAt, comment, privateFeedback, 
    # rating_cleanliness, etc.
    
    for _, row in reviews_df.iterrows():
        try:
            # Check existence first (or use INSERT OR IGNORE)
            # INSERT OR IGNORE is cleaner
            
            # Prepare JSON fields - ensure they are None/Null initially if not present
            # But here we are inserting raw reviews, so analysis hasn't happened yet.
            
            # Map DF columns to DB columns
            review_id = str(row.get('review_id', ''))
            if not review_id:
                continue # Skip invalid
                
            c.execute('''
                INSERT OR IGNORE INTO reviews (
                    review_id, entity_id, submitted_at, comment, private_feedback, full_text,
                    rating_overall, rating_cleanliness, rating_communication, 
                    rating_checkin, rating_accuracy, rating_value, rating_location
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                review_id,
                str(row.get('entityId', 'Unknown')),
                row.get('submittedAt').to_pydatetime() if pd.notnull(row.get('submittedAt')) else None,
                row.get('comment'),
                row.get('privateFeedback'),
                row.get('full_text', ''), # Might be constructed later, or empty now
                row.get('rating_overall'),
                row.get('rating_cleanliness'),
                row.get('rating_communication'),
                row.get('rating_checkin'),
                row.get('rating_accuracy'),
                row.get('rating_value'),
                row.get('rating_location')
            ))
            
            if c.rowcount > 0:
                new_count += 1
                
        except Exception as e:
            print(f"Error inserting review: {e}")
            
    conn.commit()
    conn.close()
    return new_count

def get_unprocessed_reviews():
    """Returns a list of reviews (dict) that have no positive_points analysis yet."""
    conn = get_connection()
    # Use pandas to read for easier handling
    query = "SELECT review_id, full_text, comment, private_feedback FROM reviews WHERE positive_points IS NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # If full_text is empty, construct it on the fly?
    # Better to ensure full_text is populated during insert or now.
    if not df.empty:
        if 'full_text' not in df.columns or df['full_text'].isna().all():
             df['full_text'] = df['comment'].fillna('').astype(str) + " " + df['private_feedback'].fillna('').astype(str)
    
    return df

def update_review_analysis(review_id, pos_points, neg_points, comment_en=None, private_en=None):
    """Updates a single review with analysis results including translations."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        UPDATE reviews 
        SET positive_points = ?, negative_points = ?, comment_english = ?, private_feedback_english = ?
        WHERE review_id = ?
    ''', (json.dumps(pos_points), json.dumps(neg_points), comment_en, private_en, review_id))
    
    conn.commit()
    conn.close()

def get_untranslated_reviews():
    """Returns reviews that have no comment_english (or NULL) but have text."""
    conn = get_connection()
    # Fetch reviews where comment_english IS NULL
    # We check 'comment' and 'private_feedback'
    query = """
        SELECT review_id, comment, private_feedback 
        FROM reviews 
        WHERE (comment_english IS NULL OR comment_english = '') 
          AND ( (comment IS NOT NULL AND comment != '') OR (private_feedback IS NOT NULL AND private_feedback != '') )
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_review_translation_only(review_id, comment_en, private_en):
    """Updates just the translation fields of a review."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE reviews 
        SET comment_english = ?, private_feedback_english = ?
        WHERE review_id = ?
    ''', (comment_en, private_en, review_id))
    conn.commit()
    conn.close()

def update_review_translations_bulk(updates):
    """
    Updates multiple review translations at once.
    updates: List of dicts {'id': review_id, 'comment_english': ..., 'private_feedback_english': ...}
    """
    if not updates:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    # Pre-process
    params = [(x.get('comment_english'), x.get('private_feedback_english'), x.get('id')) for x in updates]
    
    c.executemany('''
        UPDATE reviews 
        SET comment_english = ?, private_feedback_english = ?
        WHERE review_id = ?
    ''', params)
    
    conn.commit()
    conn.close()

def get_api_key():
    """Retrieves the API key from the config table."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key = 'gemini_api_key'")
    result = c.fetchone()
    conn.close()
    return result[0] if result else ""

def set_api_key(key):
    """Upserts the API key into the config table."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO config (key, value) VALUES ('gemini_api_key', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    ''', (key,))
    conn.commit()
    conn.close()

def get_all_reviews_with_listings():
    """
    Fetches all reviews JOINed with listings to get nicknames.
    Returns a DataFrame.
    """
    conn = get_connection()
    query = '''
        SELECT 
            r.*, 
            COALESCE(l.nickname, r.entity_id) as listing_name
        FROM reviews r
        LEFT JOIN listings l ON r.entity_id = l.entity_id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Parse JSON columns back to lists
    if 'positive_points' in df.columns:
        df['PositivePoints'] = df['positive_points'].apply(lambda x: json.loads(x) if x else [])
    if 'negative_points' in df.columns:
        df['NegativePoints'] = df['negative_points'].apply(lambda x: json.loads(x) if x else [])
        
    # Convert submitted_at back to datetime
    if 'submitted_at' in df.columns:
        df['submittedAt'] = pd.to_datetime(df['submitted_at'], format='mixed')
        
    df['entityId'] = df['entity_id'] # alias for compatibility
    
    return df

def delete_listings(entity_ids):
    """Deletes listings and their reviews from the database."""
    if not entity_ids:
        return 0
        
    conn = get_connection()
    c = conn.cursor()
    
    # Create placeholders
    placeholders = ','.join('?' for _ in entity_ids)
    
    # Delete reviews first
    c.execute(f"DELETE FROM reviews WHERE entity_id IN ({placeholders})", entity_ids)
    reviews_deleted = c.rowcount
    
    # Delete listings
    c.execute(f"DELETE FROM listings WHERE entity_id IN ({placeholders})", entity_ids)
    listings_deleted = c.rowcount
    
    conn.commit()
    conn.close()
    
    return listings_deleted, reviews_deleted

def delete_review(review_id):
    """Deletes a single review from the database."""
    if not review_id:
        return False
        
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM reviews WHERE review_id = ?", (review_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def insert_messages_ignore_duplicates(messages_df):
    """
    Inserts messages from a dataframe. Skips if message_id already exists.
    Returns the count of NEW messages added.
    """
    if messages_df.empty:
        return 0
        
    conn = get_connection()
    c = conn.cursor()
    
    new_count = 0
    
    for _, row in messages_df.iterrows():
        try:
            message_id = str(row.get('message_id', ''))
            if not message_id:
                continue
                
            c.execute('''
                INSERT OR IGNORE INTO messages (
                    message_id, thread_id, sender_id, text, timestamp, text_english
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                message_id,
                str(row.get('thread_id', '')),
                str(row.get('sender_id', '')),
                row.get('text', ''),
                row.get('timestamp').to_pydatetime() if pd.notnull(row.get('timestamp')) else None,
                row.get('text_english', None) # Support re-ingestion of translations
            ))
            
            if c.rowcount > 0:
                new_count += 1
        except Exception as e:
            print(f"Error inserting message: {e}")
            
    conn.commit()
    conn.close()
    return new_count

def get_unprocessed_messages():
    """Returns messages that have no analysis yet."""
    conn = get_connection()
    query = "SELECT * FROM messages WHERE analysis_json IS NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_message_analysis(message_id, analysis_data):
    """Updates a message with analysis results."""
    conn = get_connection()
    c = conn.cursor()
    
    category = analysis_data.get('category', 'Uncategorized')
    action = 1 if analysis_data.get('action_required') else 0
    
    c.execute('''
        UPDATE messages 
        SET analysis_json = ?, category = ?, action_required = ?
        WHERE message_id = ?
    ''', (json.dumps(analysis_data), category, action, message_id))
    
    conn.commit()
    conn.commit()
    conn.close()

def update_message_actionability_bulk(updates):
    """
    Updates actionability and ambiguity status for a batch of messages.
    updates: List of dicts {
        'id': message_id, 
        'action_required': bool, 
        'category': str, 
        'is_ambiguous': bool, 
        'ambiguity_reason': str,
        'summary': str
    }
    """
    if not updates:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    # Pre-process
    params = []
    for x in updates:
        # Construct analysis structure locally so we store "something" in analysis_json 
        # even if it's just the initial classification.
        # Although, user might want to do "full" analysis later. 
        # For now, we update the columns primarily.
        
        # We also create a small JSON for the analysis_json field so it's not NULL
        # This prevents it from showing up in get_unprocessed_messages() if that function checks for NULL.
        # Wait, if we only do actionability check now, do we consider it "processed"?
        # The user wants a 2-step flow. 
        # If we set analysis_json now, get_unprocessed_messages will skip it.
        # Maybe we should keep analysis_json as NULL until the "categorization" phase?
        # BUT, the current goal is to "classify actionable vs non-actionable".
        # If non-actionable, we are done. If actionable, we might want detail.
        # Let's populate analysis_json with what we have.
        
        analysis_data = {
            'action_required': x.get('action_required'),
            'category': x.get('category'),
            'summary': x.get('summary'),
            'is_ambiguous': x.get('is_ambiguous'),
            'ambiguity_reason': x.get('ambiguity_reason')
        }
        
        params.append((
            json.dumps(analysis_data),
            x.get('category'),
            1 if x.get('action_required') else 0,
            1 if x.get('is_ambiguous') else 0,
            x.get('ambiguity_reason'),
            x.get('id')
        ))
    
    c.executemany('''
        UPDATE messages 
        SET analysis_json = ?, category = ?, action_required = ?, is_ambiguous = ?, ambiguity_reason = ?
        WHERE message_id = ?
    ''', params)
    
    conn.commit()
    conn.close()

def update_message_activity(message_id, is_actionable=False, is_ambiguous=False):
    """Updates a single message's activity status (manual override)."""
    conn = get_connection()
    c = conn.cursor()
    
    # If explicitly setting actionable/not, update category too for clarity
    cat = "Actionable" if is_actionable else "Non-Actionable"
    if is_ambiguous:
        cat = "Ambiguous"
        
    c.execute('''
        UPDATE messages 
        SET action_required = ?, is_ambiguous = ?, category = ?
        WHERE message_id = ?
    ''', (1 if is_actionable else 0, 1 if is_ambiguous else 0, cat, message_id))
    
    conn.commit()
    conn.close()

def get_untranslated_messages():
    """Returns messages that have no text_english (or NULL)."""
    conn = get_connection()
    # Fetch messages where text_english IS NULL
    query = "SELECT message_id, text FROM messages WHERE text_english IS NULL AND text IS NOT NULL AND text != ''"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_message_translation(message_id, text_english):
    """Updates a message with its english translation."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE messages SET text_english = ? WHERE message_id = ?", (text_english, message_id))
    conn.commit()
    conn.close()

def update_message_translations_bulk(updates):
    """
    Updates multiple message translations at once.
    updates: List of dicts/tuples {'id': message_id, 'text_english': text}
    """
    if not updates:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    # Pre-process into list of tuples
    params = [(x.get('text_english'), x.get('id')) for x in updates]
    
    c.executemany("UPDATE messages SET text_english = ? WHERE message_id = ?", params)
    
    conn.commit()
    conn.close()
    
def get_all_messages():
    """Fetches all messages."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM messages", conn)
    conn.close()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return df

def upsert_thread_incidents(incidents):
    """
    Upserts a list of incident dicts.
    incidents: List of {'thread_id', 'category', 'severity_score', 'summary', 'message_count'}
    """
    if not incidents:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    params = [(
        x['category'], 
        x['severity_score'], 
        x['summary'], 
        x['message_count'], 
        x['thread_id']
    ) for x in incidents]
    
    c.executemany('''
        INSERT INTO thread_incidents (category, severity_score, summary, message_count, thread_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET 
            category=excluded.category,
            severity_score=excluded.severity_score,
            summary=excluded.summary,
            message_count=excluded.message_count
    ''', params)
    
    conn.commit()
    conn.close()

def get_thread_incidents():
    """Fetches all thread incidents."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM thread_incidents ORDER BY severity_score DESC", conn)
    conn.close()
    return df

def clear_thread_incidents():
    """Clears all thread incidents."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM thread_incidents")
    conn.commit()
    conn.close()

def update_message_hierarchy_bulk(updates):
    """
    Bulk updates topic and sub_issue for messages.
    updates: List of {'id': message_id, 'topic': ..., 'sub_issue': ...}
    """
    if not updates:
        return
        
    conn = get_connection()
    c = conn.cursor()
    
    params = [(x.get('topic'), x.get('sub_issue'), x.get('automation_type'), x.get('id')) for x in updates]
    
    c.executemany("UPDATE messages SET topic = ?, sub_issue = ?, automation_type = ? WHERE message_id = ?", params)
    
    conn.commit()
    conn.close()

def clear_message_hierarchy():
    """Resets hierarchical categorization."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE messages SET topic = NULL, sub_issue = NULL, automation_type = NULL")
    conn.commit()
    conn.close()

def upsert_solution_proposal(proposal):
    """
    Upserts a single solution proposal.
    proposal: dict with sub_issue, topic, analysis_json, automation_score, estimated_savings
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO solution_proposals (sub_issue, topic, analysis_json, automation_score, estimated_savings)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(sub_issue) DO UPDATE SET 
            analysis_json=excluded.analysis_json,
            automation_score=excluded.automation_score,
            estimated_savings=excluded.estimated_savings,
            topic=excluded.topic
    ''', (
        proposal['sub_issue'], 
        proposal['topic'], 
        json.dumps(proposal['analysis_json']) if isinstance(proposal['analysis_json'], dict) else proposal['analysis_json'],
        proposal['automation_score'], 
        proposal['estimated_savings']
    ))
    
    conn.commit()
    conn.close()

def get_solution_proposals():
    """Fetches all solution proposals."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM solution_proposals ORDER BY estimated_savings DESC", conn)
    conn.close()
    return df

def clear_solution_proposals():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM solution_proposals")
    conn.commit()
    conn.close()
