from utils import parse_messages
import pandas as pd
import os

file_path = "vbrfiona.xlsx"

print(f"--- Parsing {file_path} ---")
try:
    with open(file_path, "rb") as f:
        df, detected_hosts = parse_messages(f)
        
    print(f"Total Messages: {len(df)}")
    print(f"Detected Hosts: {detected_hosts}")
    
    if not df.empty:
        print("\n--- Sender IDs ---")
        senders = df['sender_id'].unique().tolist()
        print(f"Unique Sender IDs (first 10): {senders[:10]}")
        print(f"Sender ID Type: {type(senders[0])} (example: {senders[0]})")
        
        print("\n--- Simulating Filter ---")
        # Simulate common user selections like '1', '2', '3' and detected hosts
        filter_list = detected_hosts + ['1', '2', '3']
        print(f"Filter List: {filter_list}")
        
        # Check explicit type matching
        matching = df[df['sender_id'].isin(filter_list)]
        print(f"Messages Matching Filter: {len(matching)}")
        
        non_matching = df[~df['sender_id'].isin(filter_list)]
        print(f"Messages Remaining (Guest): {len(non_matching)}")
        
        # Check if type mismatch is possible
        print("\n--- Type Check ---")
        for h in detected_hosts:
            print(f"Host '{h}' in df?: {h in df['sender_id'].values}")
            
except Exception as e:
    print(f"Error: {e}")
