import pandas as pd
import utils
import db
import os
import streamlit as st

st.title("Data Ingestion Tool")

def ingest_file(filepath):
    st.write(f"Processing **{filepath}**...")
    
    # 1. Parse Messages
    st.write(f"  - Parsing Messages...")
    df_msgs, host_ids = utils.parse_messages(filepath)
    if not df_msgs.empty:
        count = db.insert_messages(df_msgs)
        st.write(f"    - Inserted {count} messages.")
        st.write(f"    - Found potential Host IDs: {host_ids}")
    else:
        st.write("    - No messages found.")

    # 2. Parse Reviews
    st.write(f"  - Parsing Reviews...")
    df_reviews = utils.parse_excel_data(filepath)
    if not df_reviews.empty:
        count = db.insert_reviews(df_reviews)
        st.write(f"    - Inserted {count} reviews.")
    else:
        st.write("    - No reviews found.")
        
    # 3. Parse Listings (Nicknames)
    st.write(f"  - Parsing Listings...")
    listings = utils.parse_listings(filepath)
    if listings:
        count = db.upsert_listings(listings)
        st.write(f"    - Updated {count} listings.")

if True:
    with st.spinner("Ingesting..."):
        db.init_db()
        files = ["vbrfiona.xlsx", "vbrpaul.xlsx", "vbrpierre.xlsx"]
        
        for f in files:
            if os.path.exists(f):
                ingest_file(f)
            else:
                st.error(f"File not found: {f}")
        
        st.success("Ingestion Complete! You can close this app.")
