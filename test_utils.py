import pandas as pd
from utils import parse_excel_data, categorize_issues
import os

# Create a dummy excel file with the flattened structure
def create_dummy_excel(filename):
    data = [
        ["reviewsReceived[0].entityId", "12345"],
        ["reviewsReceived[0].submittedAt", "2025-12-14T20:00:00.000Z"],
        ["reviewsReceived[0].comment", "The place was dirty and had a bad smell."],
        ["reviewsReceived[0].ratingV2", 2],
        
        ["reviewsReceived[1].entityId", "12345"],
        ["reviewsReceived[1].submittedAt", "2025-12-15T10:00:00.000Z"],
        ["reviewsReceived[1].comment", "Great location! Host was responsive."],
        ["reviewsReceived[1].ratingV2", 5],
        
        ["reviewsReceived[2].entityId", "67890"],
        ["reviewsReceived[2].submittedAt", "2025-12-16T12:00:00.000Z"],
        ["reviewsReceived[2].comment", "Check-in was confusing. Key code didn't work."],
        ["reviewsReceived[2].ratingV2", 3],
    ]
    df = pd.DataFrame(data)
    # Save without header
    df.to_excel(filename, index=False, header=False)
    print(f"Created {filename}")

def test_pipeline():
    filename = "test_data.xlsx"
    create_dummy_excel(filename)
    
    print("\n--- Testing Parser ---")
    df = parse_excel_data(filename)
    print("Parsed DataFrame:")
    print(df[['entityId', 'submittedAt', 'comment']].to_string())
    
    if df.empty:
        print("Parser failed to return data.")
        return

    print("\n--- Testing Analyzer ---")
    df_analyzed = categorize_issues(df)
    print("Analyzed DataFrame:")
    print(df_analyzed[['comment', 'DetectedIssues']].to_string())
    
    # Assertions
    assert len(df) == 3
    assert df.iloc[0]['entityId'] == "12345"
    
    issues_0 = df_analyzed.iloc[0]['DetectedIssues']
    assert 'Cleanliness' in issues_0
    
    issues_1 = df_analyzed.iloc[1]['DetectedIssues']
    assert 'Communication' in issues_1 or 'Location' in issues_1
    
    issues_2 = df_analyzed.iloc[2]['DetectedIssues']
    assert 'Check-in' in issues_2
    
    print("\nSUCCESS: All tests passed.")
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    test_pipeline()
