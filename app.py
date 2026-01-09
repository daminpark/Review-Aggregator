from utils import parse_excel_data, analyze_with_gemini, parse_listings, parse_messages, analyze_messages_with_gemini, clean_reviews_data, parse_csv_reviews, parse_csv_messages, translate_messages_batch, translate_reviews_batch, classify_actionability_batch, aggregate_threads, classify_topics_hierarchical_batch, generate_category_strategy_report
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
api_key = st.sidebar.text_input("Gemini API Key", value=saved_api_key, type="password", help="Enter your Google Gemini API key to allow AI analysis.")

# Save key if changed
if api_key and api_key != saved_api_key:
    db.set_api_key(api_key)
    st.toast("API Key saved!", icon="💾")

uploaded_file = st.sidebar.file_uploader("Upload File (Excel or Cleaned CSV)", type=["xlsx", "csv"])

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
            listings_temp['display'] = listings_temp['nickname'].fillna(listings_temp['entity_id']).astype(str)
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
    ["gemini-3-flash-preview", "gemini-3-pro-preview"], 
    index=0,
    help="Gemini 3 Flash is fast. Gemini 3 Pro is deeper."
)

# Host Configuration (Auto-handled now)
# We filter index 1,2,3 automatically.
# Keeping a simple display of stats.
with st.sidebar.expander("Database Stats"):
    st.caption("Overview of stored data.")
    try:
        c_stats = db.get_connection()
        r_count = pd.read_sql_query("SELECT COUNT(*) as c FROM reviews", c_stats)['c'][0]
        m_count = pd.read_sql_query("SELECT COUNT(*) as c FROM messages", c_stats)['c'][0]
        l_count = pd.read_sql_query("SELECT COUNT(*) as c FROM listings", c_stats)['c'][0]
        c_stats.close()
        st.write(f"**Listings:** {l_count}")
        st.write(f"**Reviews:** {r_count}")
        st.write(f"**Guest Msgs:** {m_count}")
        
    except:
        pass


# --- Handle Upload ---
if uploaded_file:
    # Use session state to track processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
        
    file_id = uploaded_file.name 
    
    if file_id not in st.session_state.processed_files:
        with st.spinner("Processing file..."):
            
            # --- CSV HANDLING ---
            if file_id.endswith('.csv'):
                try:
                    uploaded_file.seek(0)
                    # Detect CSV Type by columns
                    try:
                        peek = pd.read_csv(uploaded_file, nrows=1)
                        uploaded_file.seek(0) # Reset after peek
                    except:
                        peek = pd.DataFrame()
                        
                    if 'Review ID' in peek.columns and 'Public Review' in peek.columns:
                         # It's a Reviews CSV
                         df_new_rev = parse_csv_reviews(uploaded_file)
                         if not df_new_rev.empty:
                             # This handles "ingestion" of cleaned data.
                             # If IDs match, it might be ignored or duped depending on DB implementation.
                             # Our DB insert uses Ignore Duplicates on ID.
                             count = db.insert_reviews_ignore_duplicates(df_new_rev)
                             st.toast(f"✅ Loaded {count} reviews from CSV!", icon="📥")
                    
                    elif 'Message ID' in peek.columns and 'Message Content' in peek.columns:
                        # It's a Messages CSV
                        df_new_msg = parse_csv_messages(uploaded_file)
                        if not df_new_msg.empty:
                            count = db.insert_messages_ignore_duplicates(df_new_msg)
                            st.toast(f"✅ Loaded {count} messages from CSV!", icon="💬")
                    else:
                        st.error("Unknown CSV format. Please use exported files.")
                        
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")
            
            # --- EXCEL HANDLING ---
            else:
                # 1. Parse & Update Listings
                try:
                    listings_data = parse_listings(uploaded_file)
                    if listings_data:
                        db.upsert_listings(listings_data)
                except Exception as e:
                    # st.error(f"Error parsing listings: {e}") 
                    pass # might not exist

                # 2. Parse & Insert Reviews
                try:
                    uploaded_file.seek(0)
                    df_new = parse_excel_data(uploaded_file)
                    
                    if not df_new.empty:
                        new_count = db.insert_reviews_ignore_duplicates(df_new)
                        if new_count > 0:
                            st.toast(f"✅ Added {new_count} new reviews!", icon="📥")
                    
                    # 3. Parse & Insert Messages
                    uploaded_file.seek(0)
                    df_msgs, detected_hosts = parse_messages(uploaded_file)
                    
                    if not df_msgs.empty:
                         msg_count = db.insert_messages_ignore_duplicates(df_msgs)
                         if msg_count > 0:
                             st.toast(f"✅ Added {msg_count} new messages!", icon="💬")
                    
                    # Auto-save detected hosts (Log only, since filtering is upstream now)
                    # if detected_hosts:
                    #     st.toast(f"Filtered {len(detected_hosts)} host accounts.")

                    st.session_state.processed_files.add(file_id)
                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing file: {e}")

# --- TABS ---
tab_clean, tab_reviews, tab_messages = st.tabs(["🧹 Data Cleaning", "Reviews", "Message Analysis"])

with tab_clean:
    st.header("Data Cleaning & Export")
    st.markdown("Use this tab to verify and download your cleaned datasets.")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("Cleaned Reviews")
        df_rev = db.get_all_reviews_with_listings()
        if not df_rev.empty:
            st.metric("Total Reviews (Deduplicated)", len(df_rev))
            
            # Prepare for Export
            # Map columns
            export_df = df_rev.copy()
            # Ensure 'listing_name' is used as 'Listing Nickname'
            export_df.rename(columns={
                'review_id': 'Review ID', # CRITICAL for re-import
                'entityId': 'Entity ID',   # CRITICAL for re-import
                'listing_name': 'Listing Nickname',
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
            }, inplace=True)
            
            clean_cols = [
                'Review ID', 'Entity ID', 'Listing Nickname', 'Date', 'Overall Rating', 'Public Review', 'Private Note',
                'Cleanliness', 'Accuracy', 'Check-in', 'Communication', 'Location', 'Value',
                'comment_english', 'private_feedback_english'
            ]
            final_cols = [c for c in clean_cols if c in export_df.columns]
            
            csv_rev = export_df[final_cols].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download cleaned_reviews.csv",
                data=csv_rev,
                file_name="cleaned_reviews.csv",
                mime="text/csv",
                type="primary"
            )
            
            with st.expander("Preview Reviews Data"):
                st.dataframe(export_df[final_cols].head(50))
        else:
            st.info("No reviews data.")

    with col_c2:
        st.subheader("Cleaned Messages")
        df_msg = db.get_all_messages()
        if not df_msg.empty:
            # Stats
            st.metric("Total Guest Messages (Deduplicated)", len(df_msg))
            st.caption("Host messages (Participants 1, 2, 3) have been removed.")
            
            # Prepare Export
            export_msg = df_msg.copy()
            
            # Ensure text_english exists for CSV structure even if empty
            if 'text_english' not in export_msg.columns:
                 export_msg['text_english'] = ""
            
            # Rename for clarity
            export_msg.rename(columns={
                'message_id': 'Message ID',
                'thread_id': 'Thread ID',
                'sender_id': 'Sender ID',
                'text': 'Message Content',
                'text_english': 'Message Content (English)',
                'timestamp': 'Time',
                'category': 'Category',
                'action_required': 'Action Required',
                'analysis_json': 'Analysis Details'
            }, inplace=True)
            
            target_cols = ['Message ID', 'Thread ID', 'Sender ID', 'Message Content', 'Message Content (English)', 'Time', 'Category', 'Action Required', 'Analysis Details']
            final_m_cols = [c for c in target_cols if c in export_msg.columns]
            
            csv_msg = export_msg[final_m_cols].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download cleaned_messages.csv",
                data=csv_msg,
                file_name="cleaned_messages.csv",
                mime="text/csv",
                type="primary"
            )
            
            with st.expander("Preview Messages Data"):
                st.dataframe(export_msg[final_m_cols].head(50))
        else:
            st.info("No messages data.")
            
    st.divider()

with tab_reviews:
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

            # --- Review Translation Button ---
            untranslated_revs = db.get_untranslated_reviews()
            if not untranslated_revs.empty and api_key:
                st.divider()
                if st.button(f"🌐 Translate {len(untranslated_revs)} Reviews (English)"):
                    st.info("Translating reviews to English... (Resumable)")
                    
                    with st.spinner("Translating..."):
                        tr_progress = st.progress(0)
                        tr_status = st.empty()
                        
                        total_tr = len(untranslated_revs)
                        processed_tr = 0
                        
                        # Use Gemini Flash for speed
                        translator = translate_reviews_batch(untranslated_revs, api_key, model_name="gemini-3-flash-preview")
                        
                        for batch_data in translator:
                            if batch_data:
                                # Batch Save
                                db.update_review_translations_bulk(batch_data)
                                
                                processed_tr += len(batch_data)
                                prog = min(processed_tr / total_tr, 1.0)
                                tr_progress.progress(prog)
                                tr_status.text(f"Translated {processed_tr}/{total_tr} reviews...")
                                
                        tr_progress.empty()
                        tr_status.empty()
                        st.success("Translation Complete!")
                        time.sleep(1)
                        st.rerun()

        with col_b:
            with st.expander("🔄 Re-Analysis"):
                st.write("run analysis on ALL reviews again using the selected model.")
                st.warning("Consumes API quota.")
                only_issues = st.checkbox("Only reviews with issues?", value=True)
                
                if st.button("Re-categorize All"):
                     if not api_key:
                         st.error("No API Key")
                     else:
                         with st.spinner("Re-analyzing reviews..."):
                             all_df = db.get_all_reviews_with_listings()
                             if only_issues:
                                 def has_issues(x):
                                     try: return isinstance(x, list) and len(x) > 0
                                     except: return False
                                 if 'NegativePoints' in all_df.columns:
                                     mask = all_df['NegativePoints'].apply(has_issues)
                                     target_df = all_df[mask]
                                 else: target_df = all_df
                             else:
                                 target_df = all_df
                                 
                             if target_df.empty:
                                 st.warning("No reviews matched criteria.")
                             else:
                                 analyzed_df = analyze_with_gemini(target_df, api_key, model_name=model_choice)
                                 count = 0
                                 for _, row in analyzed_df.iterrows():
                                    pos = row.get('PositivePoints', [])
                                    neg = row.get('NegativePoints', [])
                                    c_en = row.get('CommentEnglish')
                                    p_en = row.get('PrivateFeedbackEnglish')
                                    db.update_review_analysis(row['review_id'], pos, neg, c_en, p_en)
                                    count += 1
                                 st.success(f"Done: {count} reviews.")
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
                st.sidebar.subheader("Review Filters")
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
        st.divider()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Reviews", len(df))
        m2.metric("Unique Listings", df['entityId'].nunique())
        if 'rating_overall' in df.columns: m3.metric("Avg Rating", f"{df['rating_overall'].mean():.2f}")
        if 'rating_cleanliness' in df.columns: m4.metric("Avg Cleanliness", f"{df['rating_cleanliness'].mean():.2f}")
        if 'rating_value' in df.columns: m5.metric("Avg Value", f"{df['rating_value'].mean():.2f}")

        # --- AI Insights ---
        st.subheader("Analysis Insights")
        
        if api_key:
            report_label = "📝 Generate Portfolio Report" if not selected_listings else f"📝 Generate Report for {len(selected_listings)} Listing(s)"
            if st.button(report_label):
                with st.spinner("Generating detailed report..."):
                    report_data = {}
                    for _, row in df.iterrows():
                        lname = str(row.get('listing_name', 'Unknown'))
                        issues = row.get('NegativePoints', [])
                        if isinstance(issues, list) and issues:
                            if lname not in report_data: report_data[lname] = []
                            report_data[lname].extend(issues)
                    
                    if not report_data:
                        st.info("No negative issues found.")
                    else:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
                        You are a Senior Property Manager. 
                        Data: {json.dumps(report_data)}
                        TASK: Per-Listing Breakdown (Room/System), Overall Portfolio Summary, Action Plan.
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
            if col not in dataframe.columns: return pd.Series(dtype='object')
            s = dataframe.explode(col)[col]
            return s.dropna()

        with col_issues:
            st.write("### 🚩 Key Issues")
            neg_series = get_exploded_series(df, 'NegativePoints')
            if not neg_series.empty:
                top_issues = neg_series.value_counts().reset_index()
                top_issues.columns = ["Topic", "Count"]
                event_issues = st.dataframe(top_issues, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch", key="issues_table")
                if len(event_issues.selection.rows) > 0:
                     selected_topic = top_issues.iloc[event_issues.selection.rows[0]]["Topic"]
            else: st.info("No issues detected yet.")

        with col_strengths:
            st.write("### ✅ Top Strengths")
            pos_series = get_exploded_series(df, 'PositivePoints')
            if not pos_series.empty:
                top_strengths = pos_series.value_counts().reset_index()
                top_strengths.columns = ["Topic", "Count"]
                event_strengths = st.dataframe(top_strengths, on_select="rerun", selection_mode="single-row", hide_index=True, width="stretch", key="strengths_table")
                if len(event_strengths.selection.rows) > 0:
                     selected_topic = top_strengths.iloc[event_strengths.selection.rows[0]]["Topic"]
            else: st.info("No strengths detected yet.")

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
        else:
            search = st.text_input("Search Reviews", placeholder="Type keywords...")
            if search:
                mask = df_display['comment'].astype(str).str.contains(search, case=False, na=False)
                if 'privateFeedback' in df_display.columns:
                    mask |= df_display['privateFeedback'].astype(str).str.contains(search, case=False, na=False)
                df_display = df_display[mask]
        
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
            selected_row = df_display.iloc[row_idx]
            review_id_to_delete = selected_row['review_id']
            st.warning("Selected Review...")
            if st.button("🗑️ Delete Selected Review", type="primary"):
                if db.delete_review(review_id_to_delete):
                    st.toast("Updated!")
                    time.sleep(1)
                    st.rerun()

with tab_messages:
    st.header("💬 Message Triage")
    st.caption("Identify messages requiring attention.")
    
    # Use live selection from sidebar if available for reactive updates
    if 'host_selection' in locals():
        host_list = host_selection
    else:
        host_list = [] # No longer needed since we filter at ingestion
        # But if we want to support extra filtering later, we can add it back.
        # For now, default to empty since DB only has guests.
    
    msgs_df = db.get_all_messages()
    
    if msgs_df.empty:
        st.info("No messages found. Please upload a file containing 'Messages' sheet.")
    else:
        # Filter metrics
        # Guest messages = not in host_list
        guest_msgs = msgs_df[~msgs_df['sender_id'].isin(host_list)]
        
        total_guest = len(guest_msgs)
        action_m = len(guest_msgs[guest_msgs['action_required'] == 1]) if 'action_required' in guest_msgs.columns else 0
        
        c1, c2 = st.columns(2)
        c1.metric("Guest Messages", total_guest)
        c2.metric("Action Required", action_m)
        
        st.divider()
        
        # Category Statistics
        if 'category' in guest_msgs.columns and not guest_msgs['category'].isna().all():
            st.subheader("📊 Issue Breakdown")
            cat_counts = guest_msgs['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            
            # Simple Bar Chart
            st.bar_chart(cat_counts.set_index('Category'), horizontal=True)
        
        st.divider()
        
        # --- TRANSLATION SECTION (Step 0) ---
        untranslated_msgs = db.get_untranslated_messages()
        # Filter guests (check against already loaded guest_msgs IDs)
        valid_guest_ids = set(guest_msgs['message_id'].tolist())
        target_untranslated = untranslated_msgs[untranslated_msgs['message_id'].isin(valid_guest_ids)]
        
        if api_key:
            st.markdown("### 🌐 Translation")
            
            # Counts
            count_needed = len(target_untranslated)
            count_done = len(guest_msgs) - count_needed if 'text_english' in guest_msgs.columns else 0
            # More precise count of 'done'
            count_done = len(guest_msgs[guest_msgs['text_english'].notnull()]) if 'text_english' in guest_msgs.columns else 0
            
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                if count_needed > 0:
                    st.info(f"**Required**: {count_needed} messages need translation.")
                    if st.button(f"Start Translation ({count_needed} msgs)"):
                        st.info("Translating messages to English... (Resumable)")
                        with st.spinner("Translating..."):
                            t_progress = st.progress(0)
                            t_status = st.empty()
                            total_t = count_needed
                            processed_t = 0
                            
                            translator = translate_messages_batch(target_untranslated, api_key, model_name="gemini-3-flash-preview")
                            
                            for batch_data in translator:
                                if batch_data:
                                    db.update_message_translations_bulk(batch_data)
                                    processed_t += len(batch_data)
                                    prog = min(processed_t / total_t, 1.0)
                                    t_progress.progress(prog)
                                    t_status.text(f"Translated {processed_t}/{total_t} messages...")
                                    
                            t_progress.empty()
                            t_status.empty()
                            st.success("Translation Complete!")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.success("✅ All messages translated.")
                    st.button("Start Translation (0)", disabled=True, key="dis_trans")
            
            with col_t2:
                if count_done > 0:
                     if st.button(f"🗑️ Clear ({count_done})", help="Resets translations for all guest messages"):
                        db.clear_message_translations()
                        st.rerun()
                else:
                     st.button("🗑️ Clear (0)", disabled=True, key="dis_clear_trans")
            
            st.divider()

        # Unanalyzed Check (Phase 1: Actionability)
        unprocessed_msgs = db.get_unprocessed_messages()
        # Filter: guest only
        unprocessed_guest = unprocessed_msgs[~unprocessed_msgs['sender_id'].isin(host_list)]
        
        # Get processed items for Phase 2 check
        processed_guest = msgs_df[~msgs_df['sender_id'].isin(host_list)]
        
        # --- AMBIGUITY QUEUE ---
        ambiguous_df = pd.DataFrame()
        if 'is_ambiguous' in unprocessed_guest.columns:
            ambiguous_df = unprocessed_guest[unprocessed_guest['is_ambiguous'] == 1]
            
        if not ambiguous_df.empty:
            st.warning(f"⚠️ **Ambiguity Queue**: {len(ambiguous_df)} messages need your review.")
            with st.expander("Review Ambiguous Messages", expanded=True):
                # Display Top 5
                to_review = ambiguous_df.head(5)
                for idx, row in to_review.iterrows():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Msg**: {row.get('text_english') or row.get('text')}")
                        st.caption(f"Reason: {row.get('ambiguity_reason')}")
                    with col2:
                        # Actions
                        if st.button("Mark Actionable", key=f"act_{row['message_id']}"):
                             db.update_message_activity(row['message_id'], is_actionable=True, is_ambiguous=False)
                             st.rerun()
                        if st.button("Ignore", key=f"ign_{row['message_id']}"):
                             db.update_message_activity(row['message_id'], is_actionable=False, is_ambiguous=False)
                             st.rerun()
                if len(ambiguous_df) > 5:
                    st.info(f"...and {len(ambiguous_df)-5} more.")
                    
        st.divider()

        # Phase 1 Button: Classify ALL unprocessed
        # Only show if there are items with NULL analysis_json
        items_needing_classification = unprocessed_guest[unprocessed_guest['analysis_json'].isnull()]
        
        # Check against translation status - warn if high % untranslated
        # But we already show the translate button above.
        
        # Phase 1 Button: Classify ALL unprocessed
        # Always show section
        items_needing_classification = unprocessed_guest[unprocessed_guest['analysis_json'].isnull()]
        
        if api_key:
            st.markdown("### 1️⃣ Phase 1: Filter Actionable Messages")
            st.caption("Classify messages as Actionable, Non-Actionable, or Ambiguous.")
            
            # Counts
            count_classify_needed = len(items_needing_classification)
            # Count classified = total guest - unclassified
            # Or safer: count where analysis_json is NOT NULL
            count_classified = len(guest_msgs[guest_msgs['analysis_json'].notnull()]) if 'analysis_json' in guest_msgs.columns else 0
            
            col_p1_1, col_p1_2 = st.columns([3, 1])
            
            with col_p1_1:
                if count_classify_needed > 0:
                    st.info(f"**Ready**: {count_classify_needed} messages need classification.")
                    if st.button(f"🚀 Classify {count_classify_needed} Messages"):
                        st.info("Classifying actionability...")
                        
                        with st.spinner("Analyzing..."):
                            c_progress = st.progress(0)
                            c_status = st.empty()
                            
                            total_c = count_classify_needed
                            processed_c = 0
                            
                            classifier = classify_actionability_batch(
                                items_needing_classification, 
                                api_key, 
                                model_name=model_choice, 
                                host_ids=host_list
                            )
                            
                            for batch_res in classifier:
                                if batch_res:
                                    db.update_message_actionability_bulk(batch_res)
                                    
                                    processed_c += len(batch_res)
                                    prog = min(processed_c / total_c, 1.0)
                                    c_progress.progress(prog)
                                    c_status.text(f"Classified {processed_c}/{total_c}...")
                                    
                            c_progress.empty()
                            c_status.empty()
                            st.success("Classification Complete! Check the Ambiguity Queue above.")
                            time.sleep(1)
                            st.rerun()
                else:
                     st.success("✅ All messages classified.")
                     st.button("🚀 Classify (0)", disabled=True, key="dis_class")

            with col_p1_2:
                 if count_classified > 0:
                     if st.button(f"🗑️ Clear ({count_classified})", help="Resets Actionable/Ambiguous tags"):
                         db.clear_message_analysis()
                         st.rerun()
                 else:
                     st.button("🗑️ Clear (0)", disabled=True, key="dis_clear_class")
                 


        st.divider()
        
        # --- PHASE 2: HIERARCHICAL CATEGORIZATION ---
        
        # Logic: We process ACTIONABLE messages that don't have a topic yet
        # processed_guest has 'action_required'
        actionable_msgs = processed_guest[processed_guest['action_required'] == 1]
        
        # Check if they have 'topic' column logic
        if 'topic' not in actionable_msgs.columns:
            actionable_msgs['topic'] = None
        if 'sub_issue' not in actionable_msgs.columns:
            actionable_msgs['sub_issue'] = None
            
        uncategorized_actionable = actionable_msgs[actionable_msgs['topic'].isnull()]
        count_uncat = len(uncategorized_actionable)
        count_cat = len(actionable_msgs[actionable_msgs['topic'].notnull()])
        
        if api_key:
            st.markdown("### 2️⃣ Phase 2: Actionable Categorization")
            st.caption("Classify actionable messages into standardized Topics & Sub-Issues.")
            
            col_p2_1, col_p2_2 = st.columns([3, 1])
            
            with col_p2_1:
                 if count_uncat > 0:
                     st.info(f"**Ready**: {count_uncat} actionable messages need categorization.")
                     if st.button(f"🏷️ Categorize {count_uncat} Messages"):
                         st.info("Categorizing topics and sub-issues...")
                         with st.spinner("Analyzing..."):
                            h_progress = st.progress(0)
                            h_status = st.empty()
                            total_h = count_uncat
                            processed_h = 0
                            
                            cats = classify_topics_hierarchical_batch(
                                uncategorized_actionable,
                                api_key, 
                                model_name=model_choice
                            )
                            
                            for batch_res in cats:
                                if batch_res:
                                    db.update_message_hierarchy_bulk(batch_res)
                                    processed_h += len(batch_res)
                                    prog = min(processed_h / total_h, 1.0)
                                    h_progress.progress(prog)
                                    h_status.text(f"Categorized {processed_h}/{total_h}...")
                                    
                            h_progress.empty()
                            h_status.empty()
                            st.success("Categorization Complete!")
                            time.sleep(1)
                            st.rerun()
                 else:
                     if len(actionable_msgs) > 0:
                         st.success("✅ All actionable messages categorized.")
                     else:
                         st.info("⚠️ No actionable messages found. Complete Phase 1 first.")
                     st.button("🏷️ Categorize (0)", disabled=True, key="dis_cat")
            
            with col_p2_2:
                if count_cat > 0:
                    if st.button(f"🗑️ Clear ({count_cat})"):
                        db.clear_message_hierarchy()
                        st.rerun()
                else:
                    st.button("🗑️ Clear (0)", disabled=True, key="dis_clear_hier")
            
            # DASHBOARD
            if count_cat > 0:
                st.subheader("📊 Operational Insights")
                
                # 1. Automation Breakdown
                if 'automation_type' in actionable_msgs.columns:
                    st.markdown("#### 🤖 Automation Feasibility")
                    auto_counts = actionable_msgs['automation_type'].value_counts().reset_index()
                    auto_counts.columns = ['Type', 'Count']
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.dataframe(auto_counts, use_container_width=True, hide_index=True)
                    with c2:
                         # Simple pie chart equivalent (bar chart)
                         st.bar_chart(auto_counts.set_index('Type'), horizontal=True)
                
                st.divider()
                
                # 2. Issue Drill-Down with Filters
                st.markdown("#### 🔎 Issue Drill-Down")
                
                # FILTER
                filter_auto_type = st.selectbox("Filter by Automation Type:", 
                                              ["All"] + (actionable_msgs['automation_type'].dropna().unique().tolist() if 'automation_type' in actionable_msgs.columns else []))
                
                filtered_msgs = actionable_msgs
                if filter_auto_type != "All":
                    filtered_msgs = actionable_msgs[actionable_msgs['automation_type'] == filter_auto_type]
                
                # Ranked Sub-Issues
                st.markdown(f"**Top Issues ({filter_auto_type})**")
                if 'sub_issue' in filtered_msgs.columns:
                    sub_counts = filtered_msgs['sub_issue'].value_counts().reset_index()
                    sub_counts.columns = ['Sub-Issue', 'Count']
                    st.dataframe(sub_counts.head(10), use_container_width=True, hide_index=True)
                
                # Topic Breakdown (Filtered)
                st.markdown(f"**Topic Breakdown ({filter_auto_type})**")
                topic_counts = filtered_msgs['topic'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']
                st.bar_chart(topic_counts.set_index('Topic'), horizontal=True)


        st.divider()
        


        # Display Filters
        st.divider()
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_action_only = st.checkbox("Show 'Action Required' Only", value=True)
        with col_f2:
            if 'category' in msgs_df.columns:
                cats = msgs_df['category'].unique().tolist()
                sel_cat = st.multiselect("Filter Category", cats)
                
        # Filter Logic
        disp_df = msgs_df.copy()
        
        # Extract Summary for display
        def extract_sum(x):
            try: return json.loads(x).get('summary') if x else None
            except: return None

        st.divider()
        
        # --- PHASE 3: STRATEGIC INSIGHTS ---
        if 'automation_type' in actionable_msgs.columns:
            st.markdown("### 3️⃣ Phase 3: Strategic Insights")
            st.caption("AI Consultant: Generate high-level improvement strategies for each automation bucket.")
            
            # Create Tabs for Groups
            tab_know, tab_act, tab_phys, tab_comp = st.tabs([
                "🧠 Chatbot (Knowledge)", 
                "⚡ Chatbot (Action)", 
                "🛠️ Physical/Human", 
                "📢 Complaint"
            ])
            
            def render_strategy_tab(tab, category_name, emoji):
                with tab:
                    # Filter Data
                    cat_data = actionable_msgs[actionable_msgs['automation_type'] == category_name]
                    count = len(cat_data)
                    
                    st.metric(f"Total {emoji} Issues", count)
                    
                    if count > 0:
                        btn_key = f"strat_btn_{category_name.replace(' ','')}"
                        if st.button(f"Generate {category_name} Strategy", key=btn_key):
                            with st.spinner(f"Consulting AI Strategy for {category_name}..."):
                                report = generate_category_strategy_report(cat_data, category_name, api_key, model_choice)
                                st.session_state[f'strategy_{category_name}'] = report
                        
                        # Show cached report
                        if f'strategy_{category_name}' in st.session_state:
                            st.markdown("---")
                            st.markdown(st.session_state[f'strategy_{category_name}'])
                            
                            # Show Drill Down
                            with st.expander(f"See Top {category_name} Issues"):
                                if 'sub_issue' in cat_data.columns:
                                     st.dataframe(cat_data['sub_issue'].value_counts().reset_index(), use_container_width=True)
                    else:
                        st.info(f"No messages found for {category_name}.")

            render_strategy_tab(tab_know, "Chatbot-Knowledge", "🧠")
            render_strategy_tab(tab_act, "Chatbot-Action", "⚡")
            render_strategy_tab(tab_phys, "Physical", "🛠️")
            render_strategy_tab(tab_comp, "Complaint", "📢")
            
        if 'analysis_json' in disp_df.columns:
            disp_df['summary'] = disp_df['analysis_json'].apply(extract_sum)
        
        if show_action_only and 'action_required' in disp_df.columns:
            disp_df = disp_df[disp_df['action_required'] == 1]
            
        if 'category' in disp_df.columns and 'sel_cat' in locals() and sel_cat:
            disp_df = disp_df[disp_df['category'].isin(sel_cat)]
            
        # Select clean columns
        # Ensure they exist
        cols_to_show = ['timestamp', 'category', 'summary', 'text_english', 'text']
        final_cols = [c for c in cols_to_show if c in disp_df.columns]
            
        st.dataframe(
            disp_df[final_cols],
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM HH:mm"),
                "category": st.column_config.TextColumn("Issue Type", width="medium"),
                "summary": st.column_config.TextColumn("AI Summary", width="medium"),
                "text_english": st.column_config.TextColumn("Message (English)", width="large"),
                "text": st.column_config.TextColumn("Original Message", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )
