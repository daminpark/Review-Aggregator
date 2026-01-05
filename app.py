from utils import parse_excel_data, analyze_with_gemini, parse_listings
import streamlit as st
import pandas as pd
import db
import time
import json

st.set_page_config(page_title="Review Aggregator", page_icon="📊", layout="wide")

# Initialize DB
db.init_db()

st.title("📊 Review Aggregator & Analyzer")
st.markdown("Persistent Database Version: Upload files to add to your collection. Duplicates are automatically skipped.")

# Sidebar for controls
st.sidebar.header("Configuration")

# Load saved key
saved_api_key = db.get_api_key()
api_key = st.sidebar.text_input("Gemini API Key", value=saved_api_key, type="password", help="Enter your Google Gemini API key to enable AI-powered analysis.")

# Save key if changed
if api_key and api_key != saved_api_key:
    db.set_api_key(api_key)
    st.toast("API Key saved!", icon="💾")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if st.sidebar.button("🗑️ Clear All Data", type="primary"):
    db.clear_db()
    st.cache_data.clear()
    if 'processed_files' in st.session_state:
        st.session_state.processed_files = set()
    st.rerun()

# Manage Data
with st.sidebar.expander("Manage Data"):
    st.caption("Delete specific listings (and their reviews).")
    try:
        conn_temp = db.get_connection()
        listings_temp = pd.read_sql_query("SELECT entity_id, nickname FROM listings", conn_temp)
        conn_temp.close()
        
        if not listings_temp.empty:
            listings_temp['display'] = listings_temp['nickname'].fillna(listings_temp['entity_id'])
            options_map = dict(zip(listings_temp['display'], listings_temp['entity_id']))
            to_delete_names = st.multiselect("Select Listings", options=listings_temp['display'].tolist())
            
            if st.button("Delete Selected"):
                if to_delete_names:
                    ids_to_del = [options_map[n] for n in to_delete_names]
                    l_del, r_del = db.delete_listings(ids_to_del)
                    st.success(f"Deleted {l_del} listings and {r_del} reviews.")
                    time.sleep(1)
                    st.rerun()
    except Exception:
        pass

st.sidebar.divider()
model_choice = st.sidebar.selectbox(
    "AI Model", 
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"], 
    index=0,
    help="Flash is faster. Pro is deeper. Use Gemini 3 for best results."
)

# --- Handle Upload ---
if uploaded_file:
    # Use session state to track processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
        
    file_id = uploaded_file.name 
    
    if file_id not in st.session_state.processed_files:
        with st.spinner("Processing file..."):
            # 1. Parse & Update Listings
            try:
                listings_data = parse_listings(uploaded_file)
                if listings_data:
                    db.upsert_listings(listings_data)
            except Exception as e:
                st.error(f"Error parsing listings: {e}")

            # 2. Parse & Insert Reviews
            try:
                uploaded_file.seek(0)
                df_new = parse_excel_data(uploaded_file)
                
                if not df_new.empty:
                    new_count = db.insert_reviews_ignore_duplicates(df_new)
                    if new_count > 0:
                        st.toast(f"✅ Added {new_count} new reviews!", icon="📥")
                    else:
                        st.toast("⚠️ No new reviews found (all duplicates).", icon="ℹ️")
                else:
                    st.warning("Could not extract reviews from file.")
                    
                st.session_state.processed_files.add(file_id)

            except Exception as e:
                st.error(f"Error processing reviews: {e}")

# --- Load Main Data from DB ---
df = db.get_all_reviews_with_listings()

if df.empty:
    st.info("Database is empty. Please upload an Airbnb Excel file to begin.")
else:
    # --- Analysis Logic (Incremental & Re-analysis) ---
    unprocessed_df = db.get_unprocessed_reviews()
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        if not unprocessed_df.empty:
            count = len(unprocessed_df)
            st.info(f"💡 {count} new reviews.")
            
            if api_key:
                if st.button(f"✨ Analyze New ({model_choice})"):
                    with st.spinner(f"Analyzing..."):
                        try:
                            analyzed_df = analyze_with_gemini(unprocessed_df, api_key, model_name=model_choice)
                            
                            # Update DB
                            success_count = 0
                            for _, row in analyzed_df.iterrows():
                                if 'PositivePoints' in row and 'NegativePoints' in row:
                                    pos = row['PositivePoints'] if isinstance(row['PositivePoints'], list) else []
                                    neg = row['NegativePoints'] if isinstance(row['NegativePoints'], list) else []
                                    c_en = row.get('CommentEnglish')
                                    p_en = row.get('PrivateFeedbackEnglish')
                                    
                                    db.update_review_analysis(row['review_id'], pos, neg, c_en, p_en)
                                    success_count += 1
                            
                            st.success(f"Updated {success_count} reviews.")
                            time.sleep(1)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
        else:
             st.success("All reviews analyzed.")

    with col_b:
        with st.expander("🔄 Re-Analysis"):
            st.write("run analysis on ALL reviews again using the selected model.")
            st.warning("Consumes API quota.")
            only_issues = st.checkbox("Only re-analyze reviews with issues?", value=True, help="Skip reviews that currently have no negative points.")
            
            if st.button("Re-categorize All"):
                 if not api_key:
                     st.error("No API Key")
                 else:
                     with st.spinner("Re-analyzing reviews..."):
                         all_df = db.get_all_reviews_with_listings()
                         
                         if only_issues:
                             # Filter for items where NegativePoints is not empty
                             # Use apply because list in parsing might be weird
                             def has_issues(x):
                                 try:
                                     return isinstance(x, list) and len(x) > 0
                                 except:
                                     return False
                             
                             if 'NegativePoints' in all_df.columns:
                                 mask = all_df['NegativePoints'].apply(has_issues)
                                 target_df = all_df[mask]
                             else:
                                 target_df = all_df
                         else:
                             target_df = all_df
                             
                         if target_df.empty:
                             st.warning("No reviews matched criteria.")
                         else:
                             st.info(f"Sending {len(target_df)} reviews to Gemini...")
                             analyzed_df = analyze_with_gemini(target_df, api_key, model_name=model_choice)
                             
                             success_count = 0
                             for _, row in analyzed_df.iterrows():
                                pos = row.get('PositivePoints', [])
                                neg = row.get('NegativePoints', [])
                                c_en = row.get('CommentEnglish')
                                p_en = row.get('PrivateFeedbackEnglish')
                                db.update_review_analysis(row['review_id'], pos, neg, c_en, p_en)
                                success_count += 1
                             
                             st.success(f"Re-analysis complete for {success_count} reviews.")
                             time.sleep(1)
                             st.rerun()

    # --- Pre-processing for Display ---
    # FILTERING
    # Date Filter
    if 'submittedAt' in df.columns and df['submittedAt'].notna().any():
        min_ts = df['submittedAt'].min()
        max_ts = df['submittedAt'].max()
        MinDate = min_ts.date() if pd.notnull(min_ts) else None
        MaxDate = max_ts.date() if pd.notnull(max_ts) else None
        
        if MinDate and MaxDate:
            st.sidebar.subheader("Filters")
            date_range = st.sidebar.date_input("Date Range", [MinDate, MaxDate])
            if len(date_range) == 2:
                start_d, end_d = date_range
                mask = (df['submittedAt'].dt.date >= start_d) & (df['submittedAt'].dt.date <= end_d)
                df = df.loc[mask]

    # Nickname Filter
    selected_listings = []
    if 'listing_name' in df.columns:
        options = sorted(df['listing_name'].dropna().unique().tolist())
        selected_listings = st.sidebar.multiselect("Filter by Listing", options)
        if selected_listings:
             df = df[df['listing_name'].isin(selected_listings)]
    
    # --- Portfolio Report Feature ---
    # Show this in sidebar or main area? Let's put it in "Analysis Insights" header area
    
    st.divider()
    
    # Calculate Average Ratings
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Reviews", len(df))
    m2.metric("Unique Listings", df['entityId'].nunique())
    if 'rating_overall' in df.columns:
        m3.metric("Avg Rating", f"{df['rating_overall'].mean():.2f}")
    if 'rating_cleanliness' in df.columns:
        m4.metric("Avg Cleanliness", f"{df['rating_cleanliness'].mean():.2f}")
    if 'rating_value' in df.columns:
        m5.metric("Avg Value", f"{df['rating_value'].mean():.2f}")

    # --- AI Insights ---
    st.subheader("Analysis Insights")
    
    # Report Generation Button
    if api_key:
        report_label = "📝 Generate Portfolio Report" if not selected_listings else f"📝 Generate Report for {len(selected_listings)} Listing(s)"
        if st.button(report_label):
            with st.spinner("Generating detailed report from negative feedback..."):
                # Collect ALL negative points associated with listing name
                # Create a structure: Listing -> [Issues]
                
                report_data = {}
                # Iterate rows
                for _, row in df.iterrows():
                    lname = str(row.get('listing_name', 'Unknown'))
                    issues = row.get('NegativePoints', [])
                    if isinstance(issues, list) and issues:
                        if lname not in report_data:
                            report_data[lname] = []
                        report_data[lname].extend(issues)
                
                # Check if we have data
                if not report_data:
                    st.info("No negative issues found to report on.")
                else:
                    # Construct Prompt
                    # We might have too much data. Limit context?
                    # Let's dump the JSON.
                    
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    
                    prompt = f"""
                    You are a Senior Property Manager. 
                    I will provide a JSON object where Keys are Listing Names and Values are lists of reported issues (Negative Feedback).
                    
                    Data:
                    {json.dumps(report_data)}
                    
                    TASK:
                    1. **Per-Listing Breakdown**: For EACH listing, categorize the issues by Room (Kitchen, Bedroom, Bath) or System (HVAC, WiFi). Summarize the key pain points.
                    2. **Overall Portfolio Summary**: Identify systemic issues affecting multiple properties.
                    3. **Action Plan**: Prioritized checklist for the maintenance team.
                    
                    Output valid Markdown.
                    """
                    
                    try:
                        resp = client.models.generate_content(model=model_choice, contents=prompt)
                        st.markdown("---")
                        st.markdown("## 📋 Generated Report")
                        st.markdown(resp.text)
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")

    col_issues, col_strengths = st.columns(2)
    selected_topic = None
    
    def get_exploded_series(dataframe, col):
        if col not in dataframe.columns:
             return pd.Series(dtype='object')
        s = dataframe.explode(col)[col]
        return s.dropna()

    with col_issues:
        st.write("### 🚩 Key Issues")
        neg_series = get_exploded_series(df, 'NegativePoints')
        if not neg_series.empty:
            top_issues = neg_series.value_counts().head(15).reset_index()
            top_issues.columns = ["Topic", "Count"]
            event_issues = st.dataframe(top_issues, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch", key="issues_table")
            if len(event_issues.selection.rows) > 0:
                 selected_topic = top_issues.iloc[event_issues.selection.rows[0]]["Topic"]
        else:
            st.info("No issues detected yet.")

    with col_strengths:
        st.write("### ✅ Top Strengths")
        pos_series = get_exploded_series(df, 'PositivePoints')
        if not pos_series.empty:
            top_strengths = pos_series.value_counts().head(15).reset_index()
            top_strengths.columns = ["Topic", "Count"]
            event_strengths = st.dataframe(top_strengths, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch", key="strengths_table")
            if len(event_strengths.selection.rows) > 0:
                 selected_topic = top_strengths.iloc[event_strengths.selection.rows[0]]["Topic"]
        else:
            st.info("No strengths detected yet.")

    # --- Review Explorer ---
    st.divider()
    st.subheader("Review Explorer")
    
    df_display = df.copy()
    
    if selected_topic:
        st.info(f"🔍 Filtering for: **{selected_topic}**")
        def has_topic(row, topic):
            p = row.get('PositivePoints', [])
            n = row.get('NegativePoints', [])
            return (topic in p) or (topic in n)
        mask = df_display.apply(lambda x: has_topic(x, selected_topic), axis=1)
        df_display = df_display[mask]
        st.caption("Deselect to clear.")
    else:
        search = st.text_input("Search Reviews", placeholder="Type keywords...")
        if search:
            mask = df_display['comment'].astype(str).str.contains(search, case=False, na=False)
            if 'privateFeedback' in df_display.columns:
                mask |= df_display['privateFeedback'].astype(str).str.contains(search, case=False, na=False)
            df_display = df_display[mask]
    
    # Translation Display Logic
    df_display['DisplayComment'] = df_display['comment']
    if 'comment_english' in df_display.columns:
        df_display['DisplayComment'] = df_display['comment_english'].fillna(df_display['comment'])

    df_display['DisplayPrivate'] = df_display.get('private_feedback', '')
    if 'private_feedback_english' in df_display.columns:
        df_display['DisplayPrivate'] = df_display['private_feedback_english'].fillna(df_display.get('private_feedback', ''))

    possible_cols = [
        'submittedAt', 'listing_name', 'rating_overall', 
        'DisplayComment', 'DisplayPrivate',
        'rating_cleanliness', 'rating_accuracy', 'rating_checkin', 
        'rating_communication', 'rating_location', 'rating_value',
        'PositivePoints', 'NegativePoints'
    ]
    cols = [c for c in possible_cols if c in df_display.columns]

    event = st.dataframe(
        df_display[cols],
        column_config={
            "submittedAt": st.column_config.DatetimeColumn("Date", format="D MMM YYYY"),
            "listing_name": "Listing",
            "rating_overall": st.column_config.NumberColumn("Rating", format="%d ⭐"),
            "DisplayComment": "Comment (Eng)",
            "DisplayPrivate": "Private Feedback (Eng)",
            "rating_cleanliness": st.column_config.NumberColumn("Cleanliness", format="%d"),
            "rating_accuracy": st.column_config.NumberColumn("Accuracy", format="%d"),
            "rating_checkin": st.column_config.NumberColumn("Check-in", format="%d"),
            "rating_communication": st.column_config.NumberColumn("Comm.", format="%d"),
            "rating_location": st.column_config.NumberColumn("Location", format="%d"),
            "rating_value": st.column_config.NumberColumn("Value", format="%d"),
            "PositivePoints": st.column_config.ListColumn("Strengths"),
            "NegativePoints": st.column_config.ListColumn("Issues"),
        },
        height=500,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    if len(event.selection.rows) > 0:
        row_idx = event.selection.rows[0]
        # Get the actual row from the displayed dataframe
        selected_row = df_display.iloc[row_idx]
        review_id_to_delete = selected_row['review_id']
        review_excerpt = str(selected_row.get('comment', ''))[:50]
        
        st.warning(f"Selected Review: '{review_excerpt}...'")
        if st.button("🗑️ Delete Selected Review", type="primary"):
            if db.delete_review(review_id_to_delete):
                st.toast("Review deleted successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to delete review.")
