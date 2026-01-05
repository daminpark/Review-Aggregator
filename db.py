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
    
    conn.commit()
    conn.close()

def clear_db():
    """Deletes all data from the database."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    init_db()

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
