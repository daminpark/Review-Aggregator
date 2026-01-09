import pandas as pd
import re
from datetime import datetime
from datetime import datetime
import json
import time
import concurrent.futures
# New SDK import
from google import genai
from google.genai import types

def parse_excel_data(file):
    """
    Parses the 'flattened' Excel file where Column A has keys like 'reviewsReceived[0].comment'
    and Column B has values.
    Returns a pandas DataFrame of structured reviews.
    """
    try:
        # Load the excel file, specifically the "Reviews" sheet
        # Assuming no header in the raw key-value format
        df_raw = pd.read_excel(file, sheet_name='Reviews', header=None, engine='openpyxl')
    except Exception as e:
        # Fallback: Start with user warning or try to inspect available sheets?
        # For now, if 'Reviews' sheet is missing or other error, return empty.
        # Streamlit app will show "No valid data".
        return pd.DataFrame()

    # Expected columns: 0 is Key, 1 is Value
    if df_raw.shape[1] < 2:
        return pd.DataFrame()

    reviews = {}
    
    # Regex to capture index and attribute
    # Example: reviewsReceived[0].comment -> index=0, attribute=comment
    # Example: reviewsReceived[0].reviewCategoryRatings[0].ratingCategory -> nested
    pattern = re.compile(r"reviewsReceived\[(\d+)\]\.(.+)")

    for index, row in df_raw.iterrows():
        key = str(row[0]).strip() # Clean whitespace
        value = row[1]
        
        match = pattern.match(key)
        if match:
            idx = int(match.group(1))
            attr = match.group(2).strip()
            
            if idx not in reviews:
                reviews[idx] = {}
            
            reviews[idx][attr] = value

    # Process nested structures (Category Ratings) before creating DataFrame
    # Format: reviewCategoryRatings[0].ratingCategory = 'CLEANLINESS'
    #         reviewCategoryRatings[0].ratingV2 = 5
    
    processed_data = []
    
    for idx, review in reviews.items():
        # Flat dict for this review
        item = {}
        
        # Temp storage for categories
        # cat_map = { '0': {'category': 'CLEANLINESS', 'score': 5} }
        cat_map = {}
        
        for k, v in review.items():
            # Handle standard fields
            if k == 'review.comment' or k == 'review.text':
                item['comment'] = v
            elif k == 'review.privateFeedback':
                item['privateFeedback'] = v
            elif k == 'review.submittedAt':
                item['submittedAt'] = v
            elif k == 'review.createdAt':
                item['createdAt'] = v
            elif k == 'review.reviewId':
                item['review_id'] = str(v)
            elif k == 'review.entityId':
                item['entityId'] = str(v)
            elif k == 'review.rating':
                item['rating_overall'] = v
            
            # Handle Category Ratings
            # reviewCategoryRatings[0].ratingCategory
            cat_match = re.match(r"reviewCategoryRatings\[(\d+)\]\.(ratingCategory|ratingV2)", k)
            if cat_match:
                c_idx = cat_match.group(1)
                c_attr = cat_match.group(2)
                
                if c_idx not in cat_map:
                    cat_map[c_idx] = {}
                cat_map[c_idx][c_attr] = v
            else:
                # Capture all other fields straight up, just in case
                # Simplify keys: review.comment -> comment
                clean_key = k.replace('review.', '')
                if clean_key not in item:
                    item[clean_key] = v

        # Flatten category ratings into item
        for c_idx, c_data in cat_map.items():
            cat_name = c_data.get('ratingCategory')
            score = c_data.get('ratingV2')
            
            # Standardize names
            if cat_name and score is not None:
                if cat_name == 'CLEANLINESS':
                    item['rating_cleanliness'] = score
                elif cat_name == 'COMMUNICATION':
                    item['rating_communication'] = score
                elif cat_name == 'CHECKIN':
                    item['rating_checkin'] = score
                elif cat_name == 'ACCURACY':
                    item['rating_accuracy'] = score
                elif cat_name == 'VALUE':
                    item['rating_value'] = score
                elif cat_name == 'LOCATION':
                    item['rating_location'] = score

        processed_data.append(item)
    
    if not processed_data:
        return pd.DataFrame()

    df = pd.DataFrame(processed_data)
    
    # Ensure review_id exists (critical for DB)
    if 'review_id' not in df.columns:
        # Fallback: Generate specific hash or use index if desperate?
        # User said "review.id" exists. If not, we can't dedup reliably.
        # Let's hope it's there. 
        # Create a fallback ID? entityId + submittedAt
        pass

    # Post-processing: Convert dates
    date_cols = ['submittedAt', 'createdAt']
    for col in date_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')

    # Robust EntityId Extraction (Backups)
    if 'entityId' not in df.columns:
         candidates = ['entityId', 'listingId', 'listing.id']
         for cand in candidates:
             if cand in df.columns and df[cand].notna().any():
                 df.loc[:, 'entityId'] = df[cand]
                 break
         if 'entityId' not in df.columns:
             df.loc[:, 'entityId'] = 'Unknown'

    return df

def parse_listings(file):
    """
    Parses the 'Listings' sheet to map IDs to Nicknames.
    Returns a list of dicts: [{'entity_id': '...', 'nickname': '...'}]
    """
    try:
        df_raw = pd.read_excel(file, sheet_name='Listings', header=None, engine='openpyxl')
    except:
        return []

    if df_raw.shape[1] < 2:
        return []

    listings = {}
    pattern = re.compile(r"listings\[(\d+)\]\.(.+)")

    for index, row in df_raw.iterrows():
        key = str(row[0]).strip()
        value = row[1]
        
        match = pattern.match(key)
        if match:
            idx = int(match.group(1))
            attr = match.group(2).strip()
            
            if idx not in listings:
                listings[idx] = {}
            
            listings[idx][attr] = value
            
    # Extract ID and Nickname
    result = []
    for idx, data in listings.items():
        # User said: listings[Y].id, listings[Y].nickname
        e_id = data.get('id') or data.get('listing.id')
        nickname = data.get('nickname') or data.get('listing.nickname') or data.get('name')
        
        if e_id:
            result.append({
                'entity_id': str(e_id),
                'nickname': str(nickname) if nickname else f"Listing {e_id}"
            })
            
    return result






def _analyze_reviews_worker(batch, api_key, model_name):
    """Worker for threaded review analysis."""
    import json
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    # Needs to return items with 'review_id' to map back for DB
    # The batch has 'review_id' column
    
    for _, row in batch.iterrows():
        rid = row['review_id']
        c_text = str(row.get('comment', '')) if pd.notnull(row.get('comment')) else ""
        pf_key = 'privateFeedback' if 'privateFeedback' in batch.columns else 'private_feedback'
        p_text = str(row.get(pf_key, '')) if pd.notnull(row.get(pf_key)) else ""
        
        prompt_items.append({
            "id": rid, 
            "comment": c_text,
            "private_feedback": p_text
        })
        
    prompt = f"""
    You are an expert hospitality analyst. Analyze the following reviews.
    
    Tasks for EACH item:
    1. **Translation**: If 'comment' or 'private_feedback' is NOT in English, translate it to English. If it is already in English, return it exactly as is.
    2. **Analysis**: Extract 'PositivePoints' (strengths) and 'NegativePoints' (issues) from BOTH the 'comment' and 'private_feedback'. Use **Standardized Labels** (e.g., 'Excellent Location', 'Noise Complaint').
    
    Input Data:
    {json.dumps(prompt_items)}
    
    Return ONLY a valid JSON List of Objects. Format:
    [
        {{
            "id": "REVIEW_ID", 
            "PositivePoints": ["Label 1", ...], 
            "NegativePoints": ["Label 1", ...],
            "CommentEnglish": "Translated comment...",
            "PrivateFeedbackEnglish": "Translated private feedback..."
        }},
        ...
    ]
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        return data
    except Exception as e:
        print(f"Batch review analysis failed: {e}")
        return []

def analyze_reviews_batch_generator(df, api_key, model_name='gemini-3-flash-preview', batch_size=20):
    """
    Analyzes reviews using Parallel Execution.
    Yields batches of analysis results (list of dicts).
    """
    import concurrent.futures
    import time
    
    if df.empty:
        return

    # Create batches
    # Ensure columns exist
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {
            executor.submit(_analyze_reviews_worker, b, api_key, model_name): b 
            for b in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                data = future.result()
                if data:
                    yield data
            except Exception as e:
                print(f"Parallel review batch failed: {e}")



def clean_reviews_data(df_reviews, listings_list):
    """
    Cleans review data:
    1. Maps entityId to Listing Nickname.
    2. Standardizes columns.
    Returns cleaned DataFrame.
    """
    if df_reviews.empty:
        return df_reviews
        
    # Create Map: ID -> Nickname
    # Handle both string and int IDs just in case
    id_map = {}
    for l in listings_list:
        eid = str(l.get('entity_id', ''))
        nick = l.get('nickname', '')
        if eid:
            id_map[eid] = nick
            
    # Function to apply map
    def get_nickname(row):
        eid = str(row.get('entityId', ''))
        return id_map.get(eid, f"Unknown ({eid})")
        
    df_reviews['Listing Nickname'] = df_reviews.apply(get_nickname, axis=1)
    
    # Rename/Reorder for final CSV
    # We want: Listing Nickname, Date, Rating, Localized Review, Private Note
    
    # Standardize Column Names for typical CSV export
    cols_map = {
        'submittedAt': 'Date',
        'rating_overall': 'Overall Rating',
        'comment': 'Public Review',
        'privateFeedback': 'Private Note',
        'rating_cleanliness': 'Cleanliness',
        'rating_accuracy': 'Accuracy',
        'rating_checkin': 'Check-in',
        'rating_communication': 'Communication',
        'rating_location': 'Location',
        'rating_value': 'Value'
    }
    
    # Add mapped columns if they exist
    for c, new_c in cols_map.items():
        if c in df_reviews.columns:
            df_reviews[new_c] = df_reviews[c]
            
    # Ensure all target columns exist (fill NaN)
    required_csv_cols = ['Listing Nickname', 'Date', 'Overall Rating', 'Public Review', 'Private Note']
    for c in required_csv_cols:
        if c not in df_reviews.columns:
            df_reviews[c] = ""
            
    return df_reviews[required_csv_cols + [c for c in df_reviews.columns if c not in required_csv_cols]]

def parse_messages(file):
    """
    Parses 'Messages' sheet. 
    Returns: (DataFrame of GUEST ONLY messages, List of host_ids encountered)
    
    STRICT FILTERING:
    - Participants at indices 1, 2, 3 in 'messageThreadParticipants' are HOSTS.
    - We exclude any message sent by these accounts.
    - We include participant 0 (Principal Guest) and 4+ (Other Guests).
    """
    try:
        df_raw = pd.read_excel(file, sheet_name='Messages', header=None, engine='openpyxl')
    except:
        return pd.DataFrame(), []

    if df_raw.shape[1] < 2:
        return pd.DataFrame(), []

    threads = {}
    thread_hosts = {} # t_idx -> set(host_account_ids)
    
    # Regex 1: Messages
    # messageThreads[0].messagesAndContents[0]...
    msg_pattern = re.compile(r"messageThreads\[(\d+)\]\.messagesAndContents\[(\d+)\]\.(.+)")
    
    # Regex 2: Participants
    # messageThreads[0].messageThreadParticipants[1].accountId
    part_pattern = re.compile(r"messageThreads\[(\d+)\]\.messageThreadParticipants\[(\d+)\]\.accountId")

    for index, row in df_raw.iterrows():
        key = str(row[0]).strip()
        value = row[1]
        
        # Check Participants (Identify Hosts)
        p_match = part_pattern.match(key)
        if p_match:
            t_idx = int(p_match.group(1))
            p_idx = int(p_match.group(2))
            
            # STRICT RULE: 1, 2, 3 are Hosts
            if p_idx in [1, 2, 3]:
                if t_idx not in thread_hosts:
                    thread_hosts[t_idx] = set()
                thread_hosts[t_idx].add(str(value))
            continue

        # Check Messages
        m_match = msg_pattern.match(key)
        if m_match:
            t_idx = int(m_match.group(1))
            m_idx = int(m_match.group(2))
            attr = m_match.group(3)
            
            if t_idx not in threads:
                threads[t_idx] = {'messages': {}}
            if m_idx not in threads[t_idx]['messages']:
                threads[t_idx]['messages'][m_idx] = {}
                
            threads[t_idx]['messages'][m_idx][attr] = value

    items = []
    
    for t_idx, t_data in threads.items():
        thread_id = f"thread_{t_idx}"
        hosts_in_thread = thread_hosts.get(t_idx, set())
        
        for m_idx, m_data in t_data['messages'].items():
            # Get Sender
            sender_id = m_data.get('message.accountId') or m_data.get('message.authorId') or m_data.get('message.userId')
            sender_id = str(sender_id).strip() if sender_id else 'Unknown'
            
            # STRICT FILTER: Skip if Host
            if sender_id in hosts_in_thread:
                continue

            # Try multiple paths for text
            text = m_data.get('messageContent.textAndReferenceContent.text')
            if not text:
                text = m_data.get('messageContent.textContent.body')
            if not text:
                text = m_data.get('message.text') # Fallback
            
            if not text:
                continue 
                
            msg_id = m_data.get('message.id') or f"{t_idx}_{m_idx}"
            ts = m_data.get('message.createdAt')
            
            items.append({
                'message_id': str(msg_id),
                'thread_id': str(thread_id),
                'sender_id': sender_id,
                'text': str(text),
                'timestamp': ts
            })
            
    df = pd.DataFrame(items)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    # Flatten host IDs for reporting purposes (optional)
    all_hosts = set()
    for s in thread_hosts.values():
        all_hosts.update(s)
        
    return df, list(all_hosts)

def parse_csv_reviews(file):
    """
    Parses a CLEANED reviews CSV.
    Expected columns: Listing Nickname, Date, Overall Rating, Public Review, Private Note, Review ID, Entity ID
    Maps them back to the internal structure for DB insertion.
    """
    try:
        df = pd.read_csv(file)
    except Exception:
        return pd.DataFrame()
        
    # Check for critical identifier columns
    if 'Review ID' not in df.columns:
        return pd.DataFrame() # Can't ingest without ID
        
    # Map back to internal storage names
    # Internal: review_id, entityId, submittedAt, comment, privateFeedback, rating_overall...
    
    mapping = {
        'Review ID': 'review_id',
        'Entity ID': 'entity_id', # raw ID
        'Listing Nickname': 'listing_name', # Just for context, not DB core
        'Date': 'submittedAt',
        'Overall Rating': 'rating_overall',
        'Public Review': 'comment',
        'Private Note': 'privateFeedback',
        'Cleanliness': 'rating_cleanliness',
        'Accuracy': 'rating_accuracy',
        'Check-in': 'rating_checkin',
        'Communication': 'rating_communication',
        'Location': 'rating_location',
        'Value': 'rating_value',
        # Analysis columns might be present if we exported them, but DB update usually happens via AI.
        # If imports contain analysis, we could parse that too, but let's stick to raw data for now
        # unless user asks for full restore.
    }
    
    renamed_df = df.rename(columns=mapping)
    
    # Ensure entityId alias exists if entity_id is present
    if 'entity_id' in renamed_df.columns:
        renamed_df['entityId'] = renamed_df['entity_id']
    else:
        # Fallback if Entity ID is missing but we want to import? 
        # We need entityId for keys.
        renamed_df['entityId'] = 'Unknown'

    # Convert Date
    if 'submittedAt' in renamed_df.columns:
        renamed_df['submittedAt'] = pd.to_datetime(renamed_df['submittedAt'], errors='coerce')

    return renamed_df

def parse_csv_messages(file):
    """
    Parses a CLEANED messages CSV.
    Expected columns including: Message ID, Thread ID, Sender ID, Message Content, Time
    """
    try:
        df = pd.read_csv(file)
    except:
        return pd.DataFrame()
        
    if 'Message ID' not in df.columns:
        return pd.DataFrame()
        
    mapping = {
        'Message ID': 'message_id',
        'Thread ID': 'thread_id',
        'Sender ID': 'sender_id',
        'Message Content': 'text',
        'Message Content (English)': 'text_english',
        'Time': 'timestamp'
    }
    
    renamed_df = df.rename(columns=mapping)
    
    if 'timestamp' in renamed_df.columns:
        renamed_df['timestamp'] = pd.to_datetime(renamed_df['timestamp'], errors='coerce')
        
    return renamed_df

def analyze_messages_with_gemini(df, api_key, model_name='gemini-3-flash-preview', host_ids=None):
    """
    Analyzes messages to determine category and actionability.
    Yields batches of results for checkpointing.
    """
    if df.empty:
        return
        
    client = genai.Client(api_key=api_key)
    
    # Batch size increased to 50 for speed
    batch_size = 50
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        prompt_items = []
        current_batch_results = []
        
        for _, row in batch.iterrows():
            mid = row['message_id']
            sid = str(row['sender_id'])
            # Use Translated Text if available, else original
            text = row.get('text_english') 
            if not text or pd.isna(text):
                text = row['text']
            
            # Skip Host Messages
            if host_ids and sid in host_ids:
                current_batch_results.append({
                    'message_id': mid,
                    'category': 'Host Sent',
                    'action_required': False,
                    'summary': 'Message sent by host.'
                })
                continue
                
            prompt_items.append({
                'id': mid,
                'text': text
            })
            
        if prompt_items:
            prompt = f"""
            Analyze these hospitality guest messages.
            
            For each message, determine:
            1. **Category**: Use highly specific format 'Type: Detail'.
               - **Check-in/Out Requests**: 'Request: Early Check-in', 'Request: Drop Bags Early', 'Request: Late Check-out', 'Request: Store Bags Late'.
               - **Amenities**: 'Question: Washing Machine', 'Question: Dryer', 'Question: Iron/Board', 'Question: Hairdryer', 'Question: Kitchen/Cooking', 'Question: TV/Streaming'.
               - **Access/Parking**: 'Question: Parking Availability', 'Question: Parking Instructions', 'Question: Keys/Access Code', 'Question: Directions/Location'.
               - **Supplies**: 'Request: Extra Towels', 'Request: Extra Bedding', 'Request: Toiletries', 'Request: Toilet Paper'.
               - **Issues/Complaints**: 'Complaint: Cleanliness', 'Complaint: Noise', 'Complaint: Temperature (Hot/Cold)', 'Complaint: Odor', 'Complaint: Wifi Connection', 'Complaint: Broken Appliance'.
               - **Logistics**: 'Info: Flight Details', 'Info: Arrival Time', 'Info: Guest Count Change'.
               - **General**: 'Greetings/Thanks', 'Review Notification'.
               
               *Instruction*: Be specific. Do not use generic 'Question: Amenities' if it can be 'Question: Washing Machine'.
            2. **ActionRequired**: Boolean (True if host reply needed).
            3. **Summary**: Very brief summary.
            
            Input:
            {json.dumps(prompt_items)}
            
            Return JSON List:
            [{{ "id": "...", "category": "...", "action_required": true/false, "summary": "..." }}]
            """
            
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                data = json.loads(resp.text)
                for item in data:
                    current_batch_results.append({
                        'message_id': item.get('id'),
                        'category': item.get('category'),
                        'action_required': item.get('action_required'),
                        'summary': item.get('summary')
                    })
            except Exception as e:
                print(f"Message analysis error: {e}")
                
        # Yield what we have for this batch (skipped + analyzed)
        if current_batch_results:
            yield current_batch_results
            time.sleep(1) # Rate limit safety

def _classify_actionability_worker(batch, api_key, model_name):
    """Worker for threaded actionability classification."""
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    # Store IDs for safe mapping
    batch_mapping = {}
    
    for _, row in batch.iterrows():
        mid = row['message_id']
        text = str(row.get('text_english') or row.get('text', ''))
        
        prompt_items.append({
            'id': mid,
            'text': text
        })
        
    if not prompt_items:
        return []
        
    prompt = f"""
    You are a strictly logical hospitality assistant.
    Task: Analyze each message and determine if the HOST needs to take action or reply.
    
    Classifications (Choose ONE):
    1. **Actionable**: The guest is asking a question, making a request, reporting an issue, or complaining. A reply is REQUIRED.
    2. **Non-Actionable**: simple thanks, "ok", "great", "checked in", system notifications, or pure chatter. NO reply needed.
    3. **Ambiguous**: You are unsure. The message is too short, context is missing, or it's unclear if it's a question.
    
    Input:
    {json.dumps(prompt_items)}
    
    Return JSON List:
    [
        {{
            "id": "...", 
            "action_required": true/false, 
            "is_ambiguous": true/false,
            "category": "Actionable" / "Non-Actionable" / "Ambiguous",
            "summary": "Verified tag for the issue (e.g. 'wifi_issue', 'checkin_query', 'simple_thanks')",
            "ambiguity_reason": "Explanation if ambiguous, else null"
        }}
    ]
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"Classification error: {e}")
        return []

def classify_actionability_batch(df, api_key, model_name='gemini-3-flash-preview', host_ids=None):
    """
    Classifies messages as Actionable, Non-Actionable, or Ambiguous using Parallel Execution.
    """
    if df.empty:
        return
        
    # Filter out host messages first if list provided
    if host_ids:
        # We can yield these immediately as Non-Actionable
        host_msgs = df[df['sender_id'].astype(str).isin(host_ids)]
        if not host_msgs.empty:
             results = []
             for _, row in host_msgs.iterrows():
                 results.append({
                     'id': row['message_id'],
                     'action_required': False,
                     'is_ambiguous': False,
                     'category': 'Host Sent',
                     'summary': 'Sent by host',
                     'ambiguity_reason': None
                 })
             yield results
             
        # Process the rest
        df = df[~df['sender_id'].astype(str).isin(host_ids)]
    
    if df.empty:
        return
        
    batch_size = 50 
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {
            executor.submit(_classify_actionability_worker, b, api_key, model_name): b 
            for b in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                data = future.result()
                if data:
                    yield data
            except Exception as e:
                print(f"Parallel classification failed: {e}")

def _translate_message_batch_worker(batch, api_key, model_name):
    """Worker for threaded execution."""
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    for _, row in batch.iterrows():
        text = str(row['text'])
        mid = row['message_id']
        
        # Skip if empty
        if not text.strip():
             continue
            
        # Add to AI queue
        prompt_items.append({
            'id': mid,
            'text': text
        })
        
    if not prompt_items:
        return []
        
    prompt = f"""
    You are a translator.
    Task: For each message:
    1. Detect the language.
    2. If it is NOT English, translate it to English.
    3. If it IS English, return it exactly as is.
    
    Input:
    {json.dumps(prompt_items)}
    
    Return JSON List:
    [{{ "id": "...", "text_english": "..." }}]
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"Translation error: {e}")
        return []

def translate_messages_batch(df, api_key, model_name='gemini-3-flash-preview'):
    """
    Translates non-English messages to English using Parallel Execution.
    """
    if df.empty:
        return
        
    batch_size = 100 # Optimistize larger batches since we filter locally
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    # Run 10 workers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {
            executor.submit(_translate_message_batch_worker, b, api_key, model_name): b 
            for b in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                data = future.result()
                if data:
                    yield data
            except Exception as e:
                print(f"Parallel batch failed: {e}") 

def _translate_review_batch_worker(batch, api_key, model_name):
    """Worker for threaded review translation."""
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    for _, row in batch.iterrows():
        c_text = str(row.get('comment', ''))
        p_text = str(row.get('private_feedback', ''))
        
        item = {
            'id': row['review_id'],
            'comment': c_text,
            'private_feedback': p_text
        }
        prompt_items.append(item)
        
    if not prompt_items:
        return []
        
    prompt = f"""
    You are a translator.
    Task: For each review:
    1. Detect the language for 'comment' and 'private_feedback'.
    2. If NOT English, translate to English.
    3. If IS English or empty, return exactly as is.
    
    Input:
    {json.dumps(prompt_items)}
    
    Return JSON List:
    [{{ "id": "...", "comment_english": "...", "private_feedback_english": "..." }}]
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"Review Translation error: {e}")
        return []

def translate_reviews_batch(df, api_key, model_name='gemini-3-flash-preview'):
    """
    Translates review batches in parallel with optimizations.
    """
    if df.empty:
        return
        
    batch_size = 50 # Increased
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {
            executor.submit(_translate_review_batch_worker, b, api_key, model_name): b 
            for b in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                data = future.result()
                if data:
                    yield data
            except Exception as e:
                print(f"Parallel review batch failed: {e}")

def aggregate_threads(df):
    """
    Groups ACTIONABLE messages by thread_id to form 'Incidents'.
    Calculates severity based on message count and keywords.
    Returns a DataFrame of incidents.
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure summary column exists
    if 'summary' not in df.columns:
        if 'analysis_json' in df.columns:
            def extract_sum(x):
                try: return json.loads(x).get('summary') if x else None
                except: return None
            df['summary'] = df['analysis_json'].apply(extract_sum)
        else:
             df['summary'] = None

    # Filter for actionable OR ambiguous (just in case)
    # Actually, we should only group actionable ones, OR if a thread has at least one actionable msg.
    # Group by Thread ID
    
    # 1. Get all messages for threads that have AT LEAST ONE actionable message
    actionable_threads = df[df['action_required'] == 1]['thread_id'].unique()
    relevant_df = df[df['thread_id'].isin(actionable_threads)]
    
    if relevant_df.empty:
        return pd.DataFrame()
        
    incidents = []
    
    grouped = relevant_df.groupby('thread_id')
    
    for tid, group in grouped:
        # Sort by time
        group = group.sort_values('timestamp')
        
        # Calculate Severity
        # Base: 10 pts per actionable message
        actionable_count = len(group[group['action_required'] == 1])
        base_score = actionable_count * 10
        
        # Keyword Boost
        joined_text = " ".join(group['text_english'].fillna('').astype(str) + " " + group['text'].fillna('').astype(str)).lower()
        urgent_keywords = ['emergency', 'urgent', 'now', 'asap', 'broken', 'locked out', 'code', 'please help']
        boost = 0
        if any(k in joined_text for k in urgent_keywords):
            boost = 50
            
        final_score = base_score + boost
        
        # Determine Dominant Category
        # Take the most frequent category among actionable messages
        cats = group[group['action_required'] == 1]['category']
        if not cats.empty:
            dominant_cat = cats.mode().iloc[0]
        else:
            dominant_cat = "General Issue"
            
        # Create Summary
        # Simple concatenation of actionable summaries for now
        # Limit to first 3 to avoid huge blocks
        summaries = group[group['action_required'] == 1]['summary'].dropna().unique().tolist()
        final_summary = " -> ".join(summaries[:3])
        if len(summaries) > 3:
            final_summary += f" (+{len(summaries)-3} more)"
            
        incidents.append({
            'thread_id': tid,
            'category': dominant_cat,
            'severity_score': final_score,
            'summary': final_summary or "No details",
            'message_count': actionable_count
        })
        
    return pd.DataFrame(incidents)

def classify_topics_hierarchical_batch(df, api_key, model_name='gemini-3-flash-preview'):
    """
    Topic + Sub-Issue Classification.
    Standardized tagging for statistical analysis.
    """
    if df.empty:
        return

    # Define Topics List for Prompt
    topics_list = "Wifi, Parking, Check-in, Access, Amenities, Noise, Cleanliness, Luggage, Other"
    
    batch_size = 50 
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_batch = {
            executor.submit(_classify_hierarchy_worker, batch, api_key, model_name, topics_list): batch 
            for batch in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                results = future.result()
                if results:
                    yield results
            except Exception as e:
                print(f"Hierarchy Processing Error: {e}")

def _classify_hierarchy_worker(batch, api_key, model_name, topics_list):
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    for _, row in batch.iterrows():
        # Use existing analysis summary if possible, or text
        # Summary is often better as it's already distilled
        text_content = row.get('text_english') or row.get('text')
        
        # If we have a summary from Phase 1, use it too
        analysis_summary = None
        if 'analysis_json' in row and row['analysis_json']:
            try: analysis_summary = json.loads(row['analysis_json']).get('summary')
            except: pass
            
        final_input = f"Message: {text_content}"
        if analysis_summary:
            final_input += f" | Context: {analysis_summary}"
            
        prompt_items.append({
            'id': row['message_id'],
            'content': final_input
        })
        
    prompt = f"""
    You are classifying guest messages for operational analytics.
    
    Step 1: Assign a TOPIC from this exact list: [{topics_list}].
    Step 2: Assign a SUB-ISSUE. 
       - MUST be a short, snake_case tag (2-4 words).
       - MUST be standardized (e.g., use 'weak_signal', not 'signal is weak').
    Step 3: Assign an AUTOMATION TYPE:
       - 'Chatbot-Knowledge': Solvable by static info (e.g., Wifi password, Check-in time).
       - 'Chatbot-Action': Requires system action/API (e.g., Late checkout, Early check-in code).
       - 'Physical': Requires human presence (e.g., Broken item, Cleanliness, Missing supplies).
       - 'Complaint': Feedback/Refund request (Requires host review).
    
    Input:
    {json.dumps(prompt_items)}
    
    Return JSON List:
    [
        {{
            "id": "...",
            "topic": "Wifi", 
            "sub_issue": "weak_signal",
            "automation_type": "Physical"
        }}
    ]
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"Hierarchy Worker Error: {e}")
        return []

def analyze_solution_batch(issues_list, api_key, model_name='gemini-3-flash-preview'):
    """
    Analyzes a list of sub-issues to generate solution proposals.
    issues_list: List of dicts {'sub_issue': ..., 'topic': ..., 'sample_messages': [text...]}
    """
    client = genai.Client(api_key=api_key)
    
    # We analyze each issue individually to better focus
    # But we can thread it
    results = []
    
    for issue_data in issues_list:
        sub = issue_data['sub_issue']
        topic = issue_data['topic']
        samples = issue_data['sample_messages'][:30] # Limit context
        
        prompt = f"""
        You are an Operational Solution Architect.
        Task: Analyze these 30 guest messages about '{sub}' (Topic: {topic}).
        
        1. Diagnosis: What is the root cause?
        2. Automation Score (0-100): Can this be solved by a digital chatbot/guide?
           - 100: Purely digital info (e.g., Wifi Password).
           - 0: Purely physical (e.g., Broken Chair).
        3. Solution Plan: 1-sentence action plan (e.g., "Add Wifi Card" or "Integrate Smart Lock API").
        4. Estimated Savings: If implemented, what % of these messages would vanish?
        
        Input Messages (Sample):
        {json.dumps(samples)}
        
        Return JSON:
        {{
            "root_cause": "...",
            "automation_score": 85,
            "solution_plan": "...",
            "savings_percent": 0.8
        }}
        """
        
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            analysis = json.loads(resp.text)
            
            # Calculate messages saved (based on sample or passed in total)
            # For now, we store the % and calculate total later in UI
            
            proposal = {
                'sub_issue': sub,
                'topic': topic,
                'analysis_json': analysis,
                'automation_score': analysis.get('automation_score', 0),
                'estimated_savings': int(analysis.get('savings_percent', 0) * 100) # Store as integer 0-100 representation
            }
            upsert_solution_proposal(proposal)
            results.append(proposal)
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error Analyzing {sub}: {e}")
            
    return results

def generate_roadmap_summary(proposals_df, api_key, model_name='gemini-3-flash-preview'):
    """
    Meta-analysis to generate a Priority Roadmap.
    """
    client = genai.Client(api_key=api_key)
    
    # Convert DF to simplified JSON
    data = []
    for _, row in proposals_df.iterrows():
        try:
            details = json.loads(row['analysis_json'])
        except:
            details = {}
            
        data.append({
            'issue': row['sub_issue'],
            'score': row['automation_score'],
            'plan': details.get('solution_plan')
        })
        
    prompt = f"""
    You are a CTO building a Strategic Automation Roadmap.
    Input: List of potential projects with Automation Scores.
    
    Task:
    1. Select the Top 3 High-Impact / High-Ease projects.
    2. Explain WHY they are the winners.
    3. Suggest an order of implementation.
    
    Data:
    {json.dumps(data)}
    
    Output Format: Markdown. Use headers and bullet points. Be concise or executive.
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"Error plotting roadmap: {e}"

def generate_category_strategy_report(df, category_name, api_key, model_name='gemini-3-flash-preview'):
    """
    Generates a high-level strategic report for a specific automation category.
    df: DataFrame of messages in this category.
    category_name: 'Chatbot-Knowledge', 'Chatbot-Action', 'Physical', or 'Complaint'.
    """
    if df.empty:
        return "No data available for this category."
        
    client = genai.Client(api_key=api_key)
    
    # 1. Get Top 10 Sub-Issues
    top_issues = df['sub_issue'].value_counts().head(10)
    
    # 2. Sample Context (3 msgs per top issue)
    context_samples = []
    for issue, count in top_issues.items():
        samples = df[df['sub_issue'] == issue].head(3)['text'].tolist()
        context_samples.append({
            'issue': issue,
            'count': count,
            'samples': samples
        })
        
    prompt = f"""
    You are an Operations Strategy Consultant.
    
    Target Category: **{category_name}**
    
    Data: I have grouped user messages into this category. Here are the Top 10 specific issues driving volume, with samples:
    {json.dumps(context_samples, default=str)}
    
    Task: Write a generic, high-level Strategic Recommendation Report (Markdown).
    
    Structure:
    1. **Executive Summary**: 1 sentence on the biggest pain point.
    2. **Top 3 Strategic Actions**:
       - Example for Knowledge: "Update Welcome Guide with Wifi instructions."
       - Example for Action: "Integrate Smart Lock API for auto-codes."
       - Example for Physical: "Replace old router in Unit 4."
    3. **Impact Estimation**: "Solving these top 3 would reduce volume by X%."
    
    Keep it concise, professional, and actionable.
    """
    
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return resp.text
    except Exception as e:
        return f"Error generating strategy: {e}"


def _extract_review_issues_worker(batch, api_key, model_name):
    """Worker for threaded Review Intelligence extraction."""
    import json
    from google import genai
    
    client = genai.Client(api_key=api_key)
    prompt_items = []
    
    for _, row in batch.iterrows():
        # Context: Negative Points list + Full Comment
        neg_points = row.get('NegativePoints', [])
        if isinstance(neg_points, str):
            try: neg_points = json.loads(neg_points)
            except: neg_points = [str(neg_points)]
        
        context_text = f"Issues: {neg_points} | Full Review: {row.get('comment')}"
        
        prompt_items.append({
            'id': row['review_id'],
            'content': context_text
        })
        
    if not prompt_items:
        return []
        
    prompt = f"""
    You are an Airbnb Quality Analyst.
    
    Task: Analyze these reviews and extract specific, atomic issues into a structured list.
    
    Categories to use:
    - 'Cleanliness': Dirt, dust, hair, smells, laundry.
    - 'Maintenance': Broken items, wifi, appliances, leaks.
    - 'Accuracy': Photos don't match, missing amenities.
    - 'Communication': Slow host, rude, bad info.
    - 'Check-in': ID issues, codes, location.
    - 'Value': Price, noise (if external).
    
    Sub-Issue Rules:
    - Snake_case (e.g., 'hair_on_bed', 'wifi_slow', 'ac_broken').
    - Be specific but standardized.
    
    Input:
    {json.dumps(prompt_items)}
    
    Return JSON List:
    [
        {{
            "review_id": "...", 
            "category": "Cleanliness",
            "sub_issue": "dirty_floor",
            "snippet": "floor was sticky"
        }},
        ...
    ]
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        
        batch_results = json.loads(response.text)
        return batch_results
                
    except Exception as e:
        print(f"Extraction Error: {e}")
        return []

def extract_review_issues_batch(df, api_key, model_name='gemini-3-flash-preview', batch_size=20):
    """
    Analyzes reviews with negative feedback using Parallel Execution.
    Yields batches of results.
    """
    import concurrent.futures
    
    if df.empty:
        return

    # Create batches
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_batch = {
            executor.submit(_extract_review_issues_worker, b, api_key, model_name): b 
            for b in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                data = future.result()
                if data:
                    yield data
            except Exception as e:
                print(f"Parallel review extraction failed: {e}")
