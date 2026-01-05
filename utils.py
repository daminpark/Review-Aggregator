import pandas as pd
import re
from datetime import datetime
from datetime import datetime

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



import json
import time
# New SDK import
from google import genai
from google.genai import types

def analyze_with_gemini(df, api_key, model_name='gemini-3-flash-preview'):
    """
    Analyzes reviews using Google Gemini API.
    Batches reviews to reduce API calls.
    Returns df with 'PositivePoints' and 'NegativePoints'.
    """
    if df.empty:
        return df
        
    # Configure Client
    client = genai.Client(api_key=api_key)
    
    # Model name is passed as argument.
    # We trust the app.py to provide a valid one.

    # Prepare data
    df['full_text'] = df['comment'].fillna('').astype(str)
    pf_col = 'privateFeedback' if 'privateFeedback' in df.columns else ('private_feedback' if 'private_feedback' in df.columns else None)
    if pf_col:
        df['full_text'] = df['full_text'] + " " + df[pf_col].fillna('').astype(str)
    
    # Create result columns
    df['PositivePoints'] = [[] for _ in range(len(df))]
    df['NegativePoints'] = [[] for _ in range(len(df))]
    
    # Batch processing
    # Reduced to 5 to avoid timeouts with complex models/large payloads
    batch_size = 5
    
    # We iterate by index chunks
    num_reviews = len(df)
    
    # Create result columns
    df['PositivePoints'] = [[] for _ in range(num_reviews)]
    df['NegativePoints'] = [[] for _ in range(num_reviews)]
    df['CommentEnglish'] = [None for _ in range(num_reviews)]
    df['PrivateFeedbackEnglish'] = [None for _ in range(num_reviews)]

    for i in range(0, num_reviews, batch_size):
        # Get slice of dataframe
        batch_df = df.iloc[i : i + batch_size]
        indices = batch_df.index.tolist()
        
        # Construct Prompt
        prompt_items = []
        for local_idx, (df_idx, row) in enumerate(batch_df.iterrows()):
            # Use original 'comment' and 'privateFeedback' (or whatever column is present)
            # We must be careful about column names.
            # In analyze_with_gemini, df usually comes from unfiltered or parsed data.
            # db.get_unprocessed_reviews returned columns: review_id, full_text, comment, private_feedback
            # parse_excel_data returns: comment, privateFeedback
            
            # Helper to get text safely
            c_text = str(row.get('comment', '')) if pd.notnull(row.get('comment')) else ""
            
            # private feedback might be 'privateFeedback' (raw) or 'private_feedback' (db)
            pf_key = 'privateFeedback' if 'privateFeedback' in df.columns else 'private_feedback'
            p_text = str(row.get(pf_key, '')) if pd.notnull(row.get(pf_key)) else ""
            
            prompt_items.append({
                "id": local_idx, 
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
                "id": 0, 
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
            
            response_text = response.text
            response_data = json.loads(response_text)
            
            # Map back to DataFrame
            for item in response_data:
                local_id = item.get('id')
                if local_id is not None and 0 <= local_id < len(indices):
                    df_idx = indices[local_id]
                    df.at[df_idx, 'PositivePoints'] = item.get('PositivePoints', [])
                    df.at[df_idx, 'NegativePoints'] = item.get('NegativePoints', [])
                    df.at[df_idx, 'CommentEnglish'] = item.get('CommentEnglish')
                    df.at[df_idx, 'PrivateFeedbackEnglish'] = item.get('PrivateFeedbackEnglish')
                    
        except Exception as e:
            print(f"Batch analysis failed: {e}")
            time.sleep(1) 
            
    return df
