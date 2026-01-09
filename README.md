# Review Aggregator & Analyzer

A powerful Streamlit-based application designed to process, aggregate, and analyze Airbnb data privacy request files (specifically UK data requests). It leverages Google's Gemini AI to provide actionable insights from guest reviews.

## 🚀 Key Features

*   **Excel Data Injection**: parses complex/messy Excel dumps from Airbnb data requests (specifically the "Reviews" and "Listings" sheets).
*   **AI-Powered Analysis**: Uses Google Gemini (supporting `gemini-3-flash-preview` and `gemini-3-pro-preview`) to:
    *   **Translate** non-English reviews automatically.
    *   **Extract** key "Positive Points" and "Negative Points" from both public comments and private feedback.
*   **Persistent Database**: Stores all processed reviews and listings in a local SQLite database (`reviews.db`) to build a historical portfolio view.
*   **Deduplication**: Smartly ignores duplicates when uploading new files, so you can upload cumulative reports without worry.
*   **Portfolio Reporting**: Generates comprehensive reports identifying systemic issues across multiple listings.
*   **Data Management**: Tools to delete specific reviews or entire listings from the database.

## 🛠️ Prerequisites

*   Python 3.10+
*   A [Google Gemini API Key](https://aistudio.google.com/app/apikey)

## 📦 Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Install NLTK data**:
    If required by specific text processing modules (not currently active in main path but good practice):
    ```bash
    python install_nltk.py
    ```

## 🖥️ Usage

1.  **Start the Application**:
    ```bash
    streamlit run app.py
    ```

2.  **Configuration**:
    *   Once the app opens in your browser, look at the **Sidebar**.
    *   Enter your **Gemini API Key**. The app remembers it for the session (and saves it locally to `reviews.db` for convenience).

3.  **Upload Data**:
    *   Upload your Airbnb Data Request Excel file (`.xlsx`).
    *   The app expects two specific sheets:
        *   `Reviews`: Contains the raw review data.
        *   `Listings`: Maps Listing IDs to Nicknames.

4.  **Analyze**:
    *   New reviews are added to the database.
    *   Click **"✨ Analyze New"** to run the AI processing on any un-analyzed reviews.
    *   Use the **Filters** to explore reviews by Date, Listing, or specific Issues (e.g., "Noise", "Cleanliness").

5.  **Generate Reports**:
    *   Use the "📝 Generate Portfolio Report" button to get a high-level summary of systemic issues and action plans for your maintenance teams.

## 📂 Project Structure

*   `app.py`: Main Streamlit application entry point. Handles UI and high-level logic.
*   `db.py`: Database management module (SQLite). Handles all CRUD operations.
*   `utils.py`: Helper functions for parsing the complex Excel structure and interacting with the Gemini API.
*   `requirements.txt`: Python package dependencies.

## 💾 Data Privacy

*   All data is stored locally in `reviews.db` on your machine.
*   Review content is sent to Google Gemini API for analysis but is subject to Google's API data privacy terms (usually not trained on for paid/API usage).
