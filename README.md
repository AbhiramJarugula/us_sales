# Project Title



### Team 16 - Ozempic Territorial Manager Dashboard



A Streamlit Dashboard of Real-Time County-Level Clinical Demand \& Market Opportunity Analysis for Weight Loss Drug



# Project Description



Streamlit dashboard designed to empower Ozempic territory managers with real-time, county-level insights into clinical demand for weight loss drugs. This tool supports pharma sales operations by identifying high-opportunity regions with the help of demand score which is calculated by utilizing "Measure" Column in dataset (*https://chronicdata.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-Census-Tract-D/cwsq-ngmh*), optimizing territory planning, and aligning sales efforts with areas of unmet clinical need.



***Formula for the Demand Score:***

***Raw Score = Σ ((Measure\_Value / 100) × Weight)***

***Normalized Score (0–100) = ((Raw Score − Min Raw Score) / (Max Raw Score − Min Raw Score)) × 100***



# Problem Statement



Territory managers in the pharmaceutical industry often lack granular, timely data to identify where demand for weight loss medications (e.g., semaglutide for obesity) is emerging. Without county-level clinical and demographic intelligence, sales efforts may be unfocused, leading to missed opportunities and inefficient resource allocation.



# App Deployment URL



The Streamlit app is deployed on community cloud platform offered by Streamlit

*URL - https://ussales.streamlit.app/*



# Local Setup Instructions



### Prerequisites



ensure you have the following installed:

Python 3.8 or higher (Download Python)

Git (Download Git)

pip (usually comes with Python)





Local Setup

---

**Step 1: Clone the Repository**
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:


*git clone https://github.com/AbhiramJarugula/us\_sales.git*

Then navigate into the project folder:



*cd us\_sales*



**Step 2: Create a Virtual Environment (Recommended)**



*python -m venv venv*

*venv\\Scripts\\activate*

**Step 3: Install Required Dependencies**
Open IDE terminal and type this command:


*pip install -r requirements.txt*



**Step 4: Run the Application**
Open terminal and type this command:


*streamlit run app.py*

**Step 5: Use the Dashboard**

Open your web browser and go to *http://localhost:8501*



# Project Structure:


us\_sales/

├── app.py                    # MAIN DASHBOARD - Start here

├── analyzer.py               # Demand score calculation logic

├── dataloading.py            # Data loading functions

├── datapreprocessing.py      # Data cleaning \& preprocessing

├── config.py                 # Configuration

├── exploratory\_data\_analysis.py  # EDA functions

├── main.py                   # Preprocessing pipeline 

└── limited\_data.parquet      # Data file

