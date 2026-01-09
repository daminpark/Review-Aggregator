import sys
from unittest.mock import MagicMock
# Mock google.genai to avoid import errors in test env
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()

import pandas as pd
from utils import parse_csv_reviews, parse_csv_messages
import os

def test_csv_ingestion():
    print("\n--- Testing CSV Ingestion ---")
    
    # 1. Test Reviews CSV
    rev_csv = "test_reviews.csv"
    df_rev = pd.DataFrame({
        'Review ID': ['REV_001'],
        'Entity ID': ['ENT_1'],
        'Listing Nickname': ['My Rental'],
        'Date': ['2025-01-01'],
        'Overall Rating': [5],
        'Public Review': ['Great stay!'],
        'Private Note': ['Thanks!'],
        # Optional columns can be missing without crashing, but let's include mapped ones
        'Cleanliness': [5] 
    })
    df_rev.to_csv(rev_csv, index=False)
    
    parsed_rev = parse_csv_reviews(rev_csv)
    print("Parsed Reviews:")
    print(parsed_rev.head(1).to_string())
    
    assert not parsed_rev.empty, "Review CSV parsing failed"
    assert parsed_rev.iloc[0]['review_id'] == 'REV_001', "Review ID mismatch"
    assert parsed_rev.iloc[0]['rating_overall'] == 5, "Rating mismatch"
    
    # 2. Test Messages CSV
    msg_csv = "test_messages.csv"
    df_msg = pd.DataFrame({
        'Message ID': ['MSG_001'],
        'Thread ID': ['TH_001'],
        'Sender ID': ['USER_1'],
        'Message Content': ['Hello'],
        'Time': ['2025-01-01 12:00:00']
    })
    df_msg.to_csv(msg_csv, index=False)
    
    parsed_msg = parse_csv_messages(msg_csv)
    print("\nParsed Messages:")
    print(parsed_msg.head(1).to_string())
    
    assert not parsed_msg.empty, "Message CSV parsing failed"
    assert parsed_msg.iloc[0]['message_id'] == 'MSG_001', "Message ID mismatch"
    assert parsed_msg.iloc[0]['text'] == 'Hello', "Text mismatch"
    
    print("\nSUCCESS: CSV Ingestion Verified.")
    
    # Cleanup
    if os.path.exists(rev_csv): os.remove(rev_csv)
    if os.path.exists(msg_csv): os.remove(msg_csv)

if __name__ == "__main__":
    test_csv_ingestion()
