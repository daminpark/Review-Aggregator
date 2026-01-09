import sys
from unittest.mock import MagicMock, PropertyMock
# Mock google.genai
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()

import pandas as pd
from utils import translate_messages_batch
import db
from langdetect import detect

def test_translation_logic():
    print("\n--- Testing Translation Batch Logic ---")
    
    # Mock Data
    df = pd.DataFrame([
        {'message_id': '1', 'text': 'Bonjour'},
        {'message_id': '2', 'text': 'Hello'}
    ])
    
    # Mock Gemini Client and Response
    # Be careful with how utils imports genai
    # utils.py does: from google import genai
    
    # We need to make sure genai.Client returns our mock_client
    mock_genai = sys.modules['google'].genai
    mock_client = mock_genai.Client.return_value
    
    # Mock Response Text for Batch
    mock_response_json = [
        {'id': '1', 'text_english': 'Hello'},
        {'id': '2', 'text_english': 'Hello'}
    ]
    
    mock_response = MagicMock()
    # Configure the 'text' property
    type(mock_response).text = PropertyMock(return_value=str(mock_response_json).replace("'", '"'))
    
    mock_client.models.generate_content.return_value = mock_response
    
    # Run Generator
    generator = translate_messages_batch(df, "fake_key")
    
    results = []
    for batch in generator:
        results.extend(batch)
        
    print(f"Results: {results}")
    
    assert len(results) == 2
    assert results[0]['id'] == '1'
    assert results[0]['text_english'] == 'Hello'
    
    print("SUCCESS: mock translation batch processed.")
    
    print("\n--- Testing Review Translation Batch Logic ---")
    from utils import translate_reviews_batch
    
    # Mock Review Data
    df_rev = pd.DataFrame([
        {'review_id': 'r1', 'comment': 'Hola', 'private_feedback': 'Gracias'},
        {'review_id': 'r2', 'comment': 'Bonjour', 'private_feedback': ''}
    ])
    
    # Mock Review Response
    mock_rev_json = [
        {'id': 'r1', 'comment_english': 'Hello', 'private_feedback_english': 'Thanks'},
        {'id': 'r2', 'comment_english': 'Hello', 'private_feedback_english': ''}
    ]
    mock_rev_response = MagicMock()
    type(mock_rev_response).text = PropertyMock(return_value=str(mock_rev_json).replace("'", '"'))
    mock_client.models.generate_content.return_value = mock_rev_response
    
    rev_gen = translate_reviews_batch(df_rev, "fake_key")
    rev_results = []
    for batch in rev_gen:
        rev_results.extend(batch)
        
    assert len(rev_results) == 2
    assert rev_results[0]['comment_english'] == 'Hello'
    print("SUCCESS: mock review translation processed.")
    
    # Verify DB column exists (sanity check on migration logic)
    print("\n--- Testing DB Column Existence ---")
    db.init_db()
    conn = db.get_connection()
    try:
        df_db = pd.read_sql_query("SELECT text_english FROM messages LIMIT 1", conn)
        print("Column 'text_english' exists.")
    except Exception as e:
        print(f"FAILED: text_english column missing. {e}")
        raise e
    finally:
        conn.close()
    
    # Verify CSV Logic for Messages
    print("\n--- Testing Message CSV Export Logic ---")
    
    # Simulate data with text_english
    msg_df = pd.DataFrame([{
        'message_id': '100', 'message_id': '100', 'text': 'Hola', 'text_english': 'Hello'
    }])
    
    # Logic from app.py
    export_df = msg_df.copy()
    if 'text_english' not in export_df.columns:
         export_df['text_english'] = ""
            
    export_df.rename(columns={
        'message_id': 'Message ID',
        'text_english': 'Message Content (English)', 
        'text': 'Message Content'
    }, inplace=True)
    
    if 'Message Content (English)' in export_df.columns:
        print("SUCCESS: 'Message Content (English)' column present in export.")
    else:
        print("FAILED: 'Message Content (English)' missing.")
        
    print("\n--- Testing Message CSV Import Logic ---")
    from utils import parse_csv_messages
    # Create temp csv
    export_df.to_csv("temp_test_msg.csv", index=False)
    # Parse back
    parsed = parse_csv_messages("temp_test_msg.csv")
    
    # Debug print
    if parsed.empty:
        print("DEBUG: Parsed DF is empty. Columns found in CSV:", pd.read_csv("temp_test_msg.csv").columns.tolist())
    
    if 'text_english' in parsed.columns:
        print("SUCCESS: Parsed CSV contains 'text_english'.")
    else:
        print("FAILED: Parsed CSV missing 'text_english'. Parse cols:", parsed.columns.tolist())
        
    import os
    if os.path.exists("temp_test_msg.csv"):
        os.remove("temp_test_msg.csv")

if __name__ == "__main__":
    test_translation_logic()
