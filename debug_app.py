import streamlit as st
import pandas as pd
import re

st.title("Debug Parsing - Text Check")

file_path = "vbrfiona.xlsx"
target_id = "568520441"

try:
    # RAW
    df_raw = pd.read_excel(file_path, sheet_name='Messages', header=None, engine='openpyxl')
    
    threads = {}
    msg_pattern = re.compile(r"messageThreads\[(\d+)\]\.messagesAndContents\[(\d+)\]\.(.+)")
    
    found_target = False
    
    for index, row in df_raw.iterrows():
        key = str(row[0]).strip()
        value = row[1]
        m_match = msg_pattern.match(key)
        if m_match:
            t_idx = int(m_match.group(1))
            m_idx = int(m_match.group(2))
            attr = m_match.group(3)
            
            if t_idx not in threads: threads[t_idx] = {'messages': {}}
            if m_idx not in threads[t_idx]['messages']: threads[t_idx]['messages'][m_idx] = {}
            threads[t_idx]['messages'][m_idx][attr] = value

    # Inspect
    for t_idx, t_data in threads.items():
        if found_target: break
        for m_idx, m_data in t_data['messages'].items():
            # Check if this message has our target ID
            # Values might be int or string
            found = False
            for k, v in m_data.items():
                if str(v) == target_id:
                    found = True
                    break
            
            if found:
                st.success(f"FOUND TARGET MESSAGE: Thread {t_idx}, Msg {m_idx}")
                st.write("### Keys present:")
                st.json(list(m_data.keys()))
                
                # Check Text
                t1 = m_data.get('messageContent.textAndReferenceContent.text')
                t2 = m_data.get('message.text')
                st.write(f"Text pattern 1: {t1}")
                st.write(f"Text pattern 2: {t2}")
                
                # Look for other text-like keys
                st.write("### Potential Text fields:")
                for k, v in m_data.items():
                    if 'text' in k.lower() or 'body' in k.lower() or 'content' in k.lower():
                        st.code(f"{k}: {v}")
                
                found_target = True
                break

    if not found_target:
        st.error(f"Could not find any message object containing value {target_id}")

except Exception as e:
    st.error(f"Error: {e}")
