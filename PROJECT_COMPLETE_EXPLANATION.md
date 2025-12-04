# HSU Early Warning System - Complete Project Explanation

## 📋 Table of Contents
1. [Application Overview](#1-application-overview)
2. [Problem Statement & Proposed Solution](#2-problem-statement--proposed-solution)
3. [Technology Stack - Frontend & Backend](#3-technology-stack---frontend--backend)
4. [Project File Structure](#4-project-file-structure)
5. [How Files Connect to Each Other](#5-how-files-connect-to-each-other)
6. [Data Flow Architecture](#6-data-flow-architecture)
7. [Database Connection Explained](#7-database-connection-explained)
8. [Authentication & User Management](#8-authentication--user-management)
9. [Machine Learning / AI Implementation](#9-machine-learning--ai-implementation)
10. [Analytics & Dynamic Data Presentation](#10-analytics--dynamic-data-presentation)
11. [Why SignUp Works But Doesn't Persist After Logout](#11-why-signup-works-but-doesnt-persist-after-logout)
12. [Deployment Guide](#12-deployment-guide)
13. [Policies & Permissions Design](#13-policies--permissions-design)

---

## 1. Application Overview

### What is This Project?
The **HSU Early Warning System (EWS)** is a web application designed to help **Hardin-Simmons University (HSU)** identify students who are at risk of dropping out. It uses **Machine Learning** to predict which students need help, and provides tools for advisors to take action.

### Who Uses This Application?

| User Type | What They Can Do |
|-----------|------------------|
| **Students** | View their own risk score, grades, attendance, schedule meetings with advisor |
| **Advisors** | See all students, filter by risk level, send emails, create interventions, schedule meetings |
| **Admins** | Everything advisors can do + manage users, view system statistics, access all reports |

### Main Features
1. **Risk Prediction** - ML model predicts dropout probability (94.33% accuracy)
2. **Dashboard** - Visual overview of all at-risk students
3. **Student Portal** - Students can see their own data
4. **Interventions** - Advisors can log support actions
5. **Analytics** - Charts showing risk distribution, GPA trends, equity gaps
6. **Email Alerts** - Queue emails to at-risk students
7. **Appointment Scheduling** - Book meetings with students

---

## 2. Problem Statement & Proposed Solution

### The Problem (Why This Project Exists)

```
📉 Student Attrition Rate: 24% (national average)
⏰ Warning Signs: Often identified TOO LATE
💰 Annual Revenue Loss: $67 Million per 10,000 students
```

Universities lose students because:
- **Academic struggles** go unnoticed until it's too late
- **Financial stress** causes students to drop out
- **Mental health issues** are not identified early
- **Disengagement** (not logging into LMS, missing classes) is ignored
- **Advisors** don't have time to manually track 500+ students

### The Solution (What This App Does)

```
🎯 PREDICT → 📊 ANALYZE → 👥 INTERVENE → 📈 TRACK
```

| Step | What Happens |
|------|--------------|
| **1. Predict Risk** | ML model analyzes 69+ features (GPA, attendance, LMS logins, payments, counseling visits) |
| **2. Categorize** | Students are placed in Critical (🔴), High (🟠), Medium (🟡), Low (🟢) risk buckets |
| **3. Alert Advisors** | Dashboard shows prioritized list of students needing help |
| **4. Take Action** | Advisors schedule meetings, send emails, log interventions |
| **5. Track Results** | System tracks intervention success rate (67% improvement) |

### Expected Impact

| Metric | Before | After (with EWS) |
|--------|--------|------------------|
| Retention Rate | 64% | 76% (+12%) |
| Early Intervention | 20% | 85% |
| Advisor Efficiency | 1:500 ratio | 1:150 focused |
| ROI | N/A | 22:1 |

---

## 3. Technology Stack - Frontend & Backend

### Simple Explanation

> **What is Frontend?**
> The part users SEE and CLICK - buttons, forms, charts, tables.

> **What is Backend?**
> The part that PROCESSES data - reads/writes database, runs ML model, handles login.

### This Project is Different!

In traditional web apps:
- Frontend = React/Angular/Vue (JavaScript)
- Backend = Flask/Django/Node.js (Python/JavaScript)
- They are SEPARATE

**In THIS project (Streamlit), frontend and backend are COMBINED!**

```
┌──────────────────────────────────────────────────────────────┐
│                     STREAMLIT APPLICATION                    │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │    FRONTEND     │    │           BACKEND               │  │
│  │  (What you see) │ ←→ │  (Processing behind the scenes) │  │
│  │                 │    │                                 │  │
│  │ - HTML/CSS      │    │ - Python code                   │  │
│  │ - Buttons       │    │ - Database queries              │  │
│  │ - Forms         │    │ - ML predictions                │  │
│  │ - Charts        │    │ - Authentication                │  │
│  │ - Tables        │    │ - Data processing               │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### What is Streamlit?

**Streamlit** is a Python framework that turns Python scripts into web apps.

- **It is NOT Next.js** (Next.js is JavaScript/React for complex websites)
- **It is NOT a tool** (It's a Python library)
- **It IS a framework** that automatically creates HTML/CSS from Python code

```python
# This Python code...
st.title("Hello World")
st.button("Click Me")

# ...automatically becomes HTML like this:
# <h1>Hello World</h1>
# <button>Click Me</button>
```

### Tools Used in This Project

| Tool/Library | What It Does | Category |
|--------------|--------------|----------|
| **Streamlit** | Creates the web interface | Framework (Frontend + Backend) |
| **Python** | Programming language | Core |
| **Pandas** | Data manipulation (DataFrames) | Data Processing |
| **Plotly** | Creates interactive charts | Visualization |
| **SQLite** | Stores user data, interventions | Database |
| **Scikit-learn** | Machine Learning algorithms | ML/AI |
| **SHAP** | Explains ML predictions | ML Explainability |
| **Hashlib** | Password hashing (security) | Security |
| **Joblib** | Saves/loads ML models | Utilities |

---

## 4. Project File Structure

```
HSU-Streamlit-App/
│
├── app.py                      # 🏠 MAIN ENTRY POINT (Landing Page)
│
├── pages/                      # 📄 ALL APPLICATION PAGES
│   ├── 0_🔐_Login.py           # Login form
│   ├── 0_✨_SignUp.py          # Registration form
│   ├── 1_🏠_Dashboard.py       # Advisor dashboard (main working area)
│   ├── 2_👥_Students.py        # Student list with search/filter
│   ├── 3_📊_Analytics.py       # Charts and visualizations
│   ├── 4_🎓_Student_Portal.py  # Student's own view
│   ├── 5_👔_Admin_Portal.py    # Admin controls
│   ├── 6_🎯_ML_Predictions.py  # ML model predictions
│   └── 7_📝_Interventions.py   # Intervention logging
│
├── utils/                      # 🔧 HELPER/UTILITY FILES
│   ├── auth.py                 # Authentication (login/logout)
│   ├── db_auth.py              # Database-based authentication
│   ├── data_loader.py          # Loads data from CSV files
│   ├── db_data_loader.py       # Loads data from SQLite database
│   ├── email_service.py        # Email sending logic
│   ├── intervention_manager.py # Intervention CRUD operations
│   └── premium_design.py       # CSS styling functions
│
├── database/                   # 💾 DATABASE FILES
│   ├── db_manager.py           # Database connection & queries
│   ├── schema.sql              # Table definitions (23 tables!)
│   ├── migrate_csv_to_db.py    # Migrates CSV data to SQLite
│   └── hsu_database.db         # SQLite database file (auto-created)
│
├── Data_Web/                   # 📊 CSV DATA FILES
│   ├── students.csv            # 152 students with demographics
│   ├── risk_scores.csv         # ML-generated risk scores
│   ├── enrollments.csv         # Course enrollments
│   ├── grades.csv              # Assignment grades
│   ├── attendance.csv          # Class attendance records
│   ├── logins.csv              # LMS login history
│   ├── payments.csv            # Tuition payments
│   ├── counseling.csv          # Counseling visits
│   ├── courses.csv             # Course catalog
│   ├── departments.csv         # Academic departments
│   ├── faculty.csv             # Faculty information
│   └── terms.csv               # Academic terms (semesters)
│
├── models/                     # 🤖 ML MODEL FILES
│   ├── metadata.json           # Model version, features, accuracy
│   └── feature_names.json      # List of 69 features used
│
├── .streamlit/                 # ⚙️ STREAMLIT CONFIGURATION
│   ├── config.toml             # Theme, server settings
│   └── pages.toml              # Page navigation order
│
├── requirements.txt            # 📦 Python dependencies
├── Procfile                    # 🚀 Heroku deployment config
├── setup.sh                    # 🐧 Server setup script
└── README.md                   # 📖 Project documentation
```

---

## 5. How Files Connect to Each Other

### Connection Diagram

```
                              ┌─────────────┐
                              │   app.py    │ (Entry Point)
                              │  Landing    │
                              │    Page     │
                              └──────┬──────┘
                                     │
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
┌───────────────┐                                        ┌─────────────────┐
│ 0_🔐_Login.py │                                        │ 0_✨_SignUp.py  │
│               │                                        │                 │
│  IMPORTS:     │                                        │   IMPORTS:      │
│  utils/auth.py│                                        │   utils/auth.py │
└───────┬───────┘                                        └────────┬────────┘
        │                                                          │
        │      ┌───────────────────────────────────────────┐       │
        └──────►          utils/auth.py                    ◄───────┘
               │  - authenticate_user()                    │
               │  - register_user()                        │
               │  - hash_password()                        │
               │                                           │
               │  IMPORTS: database/db_manager.py          │
               │           utils/data_loader.py            │
               └───────────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────────┐   ┌──────────────────┐    ┌─────────────────────┐
│ database/         │   │ utils/           │    │ utils/              │
│ db_manager.py     │   │ data_loader.py   │    │ db_data_loader.py   │
│                   │   │                  │    │                     │
│ - SQLite connect  │   │ - load_students()│    │ - load_students()   │
│ - CRUD operations │   │ - load_grades()  │    │ - SQL queries       │
│ - User auth       │   │ - (from CSV)     │    │ - (from Database)   │
└─────────┬─────────┘   └────────┬─────────┘    └──────────┬──────────┘
          │                      │                         │
          ▼                      ▼                         ▼
┌─────────────────┐     ┌────────────────┐      ┌──────────────────────┐
│ database/       │     │  Data_Web/     │      │  database/           │
│ hsu_database.db │     │  *.csv files   │      │  hsu_database.db     │
│ (SQLite file)   │     │                │      │                      │
└─────────────────┘     └────────────────┘      └──────────────────────┘
```

### How Pages Load Data

Each page in `pages/` folder imports utilities:

```python
# Example: pages/1_🏠_Dashboard.py

# Step 1: Import authentication
from utils.auth import require_role, display_user_info

# Step 2: Try database loader, fallback to CSV
try:
    from utils.db_data_loader import load_students, load_risk_scores
    DATABASE_MODE = True
except ImportError:
    from utils.data_loader import load_students, load_risk_scores
    DATABASE_MODE = False

# Step 3: Load data
students = load_students()      # Returns pandas DataFrame
risk_scores = load_risk_scores() # Returns pandas DataFrame

# Step 4: Merge and display
df = students.merge(risk_scores, on='StudentID')
st.dataframe(df)  # Show in web UI
```

---

## 6. Data Flow Architecture

### Step-by-Step Data Flow

```
USER ACTION                       WHAT HAPPENS IN CODE
──────────────────────────────────────────────────────────────────────

1. User opens app              → app.py loads
                               → Streamlit renders HTML/CSS
                               
2. User clicks "Login"         → st.switch_page("pages/0_🔐_Login.py")
                               
3. User enters credentials     → authenticate_user(email, password)
   and clicks "Sign In"        → db_manager.py queries SQLite:
                                  SELECT * FROM users WHERE email=?
                               → If match: st.session_state["authenticated"] = True
                               → Redirect to Dashboard
                               
4. Dashboard loads             → load_students() called
                               → load_risk_scores() called
                               → Data merged: students.merge(risk_scores)
                               → Charts rendered: plotly.express.pie()
                               → st.dataframe() shows table
                               
5. Advisor clicks              → st.button("📅 Schedule Meeting")
   "Schedule Meeting"          → INSERT INTO appointments VALUES(...)
                               → st.success("Meeting scheduled!")
                               
6. Advisor filters by          → st.selectbox("Risk Level")
   "Critical" risk             → filtered_df = df[df['RiskCategory']=='Critical']
                               → Table updates dynamically
```

### CSV Mode vs Database Mode

The app supports TWO data sources:

| Mode | How It Works | When Used |
|------|--------------|-----------|
| **CSV Mode** | Reads `Data_Web/*.csv` files directly | Default, simpler, read-only |
| **Database Mode** | Reads/writes SQLite `hsu_database.db` | Full features, persistent |

```python
# Code that switches between modes (in each page file)

try:
    from utils.db_data_loader import load_students  # Try database
    DATABASE_MODE = True
except ImportError:
    from utils.data_loader import load_students     # Fallback to CSV
    DATABASE_MODE = False
```

---

## 7. Database Connection Explained

### What Database is Used?

**SQLite** - A file-based database that doesn't need a server.

- Database file: `database/hsu_database.db`
- Created automatically when app first runs
- Schema defined in: `database/schema.sql`

### 23 Tables in the Database

| Table Name | Purpose |
|------------|---------|
| `users` | Login credentials (email, password_hash, role) |
| `students` | Student demographics (name, email, classification) |
| `advisors` | Advisor profiles |
| `enrollments` | Student-course relationships |
| `grades` | Assignment scores |
| `attendance` | Class attendance records |
| `logins` | LMS login history |
| `payments` | Tuition payment records |
| `counseling` | Counseling visit records |
| `risk_scores` | ML-generated risk predictions |
| `interventions` | Advisor actions logged |
| `appointments` | Scheduled meetings |
| `notifications` | System alerts |
| `email_queue` | Emails waiting to be sent |
| `audit_logs` | User action history |
| ... | (8 more tables) |

### How Database Connection Works

```python
# database/db_manager.py

class DatabaseManager:
    def __init__(self):
        self.db_path = "database/hsu_database.db"
    
    def get_connection(self):
        """Opens connection to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def authenticate_user(self, email, password):
        """Check if user credentials are valid"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, email, role, first_name, last_name
                FROM users
                WHERE email = ? AND password_hash = ?
            """, (email, password_hash))
            
            user = cursor.fetchone()
            return dict(user) if user else None

# Global instance - used everywhere
db = DatabaseManager()
```

### Where is Login Data Saved?

When a user signs up:

```python
# utils/auth.py → register_user()

def register_user(first_name, last_name, email, password, role):
    conn = sqlite3.connect('database/hsu_database.db')
    cursor = conn.cursor()
    
    # Hash password for security
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    # Insert into users table
    cursor.execute("""
        INSERT INTO users (first_name, last_name, email, password_hash, role)
        VALUES (?, ?, ?, ?, ?)
    """, (first_name, last_name, email, hashed_password, role))
    
    conn.commit()
    conn.close()
```

---

## 8. Authentication & User Management

### How Login Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LOGIN FLOW                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User enters email + password                                    │
│                 │                                                   │
│                 ▼                                                   │
│  2. authenticate_user(email, password)                              │
│                 │                                                   │
│                 ▼                                                   │
│  3. Hash password: SHA256("password123")                            │
│        Result: "ef92b778bafe7..."                                   │
│                 │                                                   │
│                 ▼                                                   │
│  4. Query database:                                                 │
│     SELECT * FROM users                                             │
│     WHERE email = 'user@hsu.edu'                                    │
│     AND password_hash = 'ef92b778bafe7...'                          │
│                 │                                                   │
│        ┌───────┴───────┐                                            │
│        ▼               ▼                                            │
│     FOUND           NOT FOUND                                       │
│        │               │                                            │
│        ▼               ▼                                            │
│  5a. Store in      5b. Show error                                   │
│      session:          "Invalid credentials"                        │
│      - authenticated=True                                           │
│      - role="advisor"                                               │
│      - name="Dr. Johnson"                                           │
│                 │                                                   │
│                 ▼                                                   │
│  6. Redirect to appropriate page:                                   │
│     - Student → Student Portal                                      │
│     - Advisor → Dashboard                                           │
│     - Admin → Admin Portal                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Session State (Streamlit's Memory)

Streamlit uses `st.session_state` to remember things across page refreshes:

```python
# After successful login
st.session_state["authenticated"] = True
st.session_state["role"] = "advisor"
st.session_state["name"] = "Dr. Sarah Johnson"
st.session_state["email"] = "advisor@hsu.edu"

# Checking authentication on each page
if st.session_state.get("authenticated") != True:
    st.warning("Please log in")
    st.stop()
```

### Role-Based Access Control

```python
# On Dashboard page (advisors + admins only)
if st.session_state.get("role") not in ['advisor', 'admin']:
    st.error("Access Denied: This page is for Advisors and Admins only")
    st.stop()

# On Student Portal (students only)
if st.session_state.get("role") != 'student':
    st.error("Access Denied: This page is for Students only")
    st.stop()
```

---

## 9. Machine Learning / AI Implementation

### What ML Does in This Project

The ML model predicts: **Will this student drop out?**

- **Input**: 69 features (GPA, attendance, logins, payments, etc.)
- **Output**: Risk score (0-100) and category (Critical/High/Medium/Low)
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 94.33%

### Where is the ML Code?

```
ml_pipeline/
├── src/
│   ├── ml_pipeline_complete.py   # Main training script
│   ├── production_predict.py      # Makes predictions on new students
│   └── show_model_accuracy.py     # Displays model performance
├── models/
│   ├── metadata.json              # Model info (accuracy, features)
│   └── feature_names.json         # List of 69 features
└── results/
    ├── feature_importance.csv     # Which features matter most
    └── test_predictions.csv       # Predictions on test data
```

### 69 Features Used (Grouped)

| Category | Example Features | Root Cause Addressed |
|----------|------------------|---------------------|
| **Demographic** | FirstGenStudent, International | Background risk |
| **Academic** | GPA, MidtermGrade, CourseLoad | Academic struggle |
| **Engagement** | LMS_Logins, Assignment_OnTime | Disengagement |
| **Financial** | PaymentRate, HasHold | Financial stress |
| **Wellness** | CounselingVisits, CrisisFlag | Mental health |

### How Predictions are Displayed

```python
# pages/6_🎯_ML_Predictions.py

# Load pre-computed risk scores from CSV/database
risk_scores = load_risk_scores()

# Display in Dashboard
for student in filtered_students:
    risk = risk_scores[risk_scores['StudentID'] == student['StudentID']]
    
    if risk['OverallRiskScore'] >= 0.7:
        st.error(f"🔴 CRITICAL: {student['Name']} - Score: {risk['OverallRiskScore']*100:.0f}")
    elif risk['OverallRiskScore'] >= 0.5:
        st.warning(f"🟠 HIGH: {student['Name']} - Score: {risk['OverallRiskScore']*100:.0f}")
```

### Model Performance Metrics

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Accuracy** | 94.33% | 94 out of 100 predictions correct |
| **Precision** | 97.06% | When we say "at-risk", we're usually right |
| **Recall** | 77.42% | We catch 77% of actually at-risk students |
| **AUC-ROC** | 93.20% | Model distinguishes well between risk levels |

---

## 10. Analytics & Dynamic Data Presentation

### How Charts are Created

All charts use **Plotly** library:

```python
# pages/3_📊_Analytics.py

import plotly.express as px
import plotly.graph_objects as go

# Pie chart for risk distribution
risk_counts = df['RiskCategory'].value_counts()
fig = px.pie(
    values=risk_counts.values,
    names=risk_counts.index,
    color=risk_counts.index,
    color_discrete_map={
        'Critical': '#DC2626',
        'High': '#F59E0B',
        'Medium': '#FBBF24',
        'Low': '#10B981'
    }
)
st.plotly_chart(fig)

# Bar chart for GPA by classification
fig = px.bar(df, x='Classification', y='GPA', color='RiskCategory')
st.plotly_chart(fig)
```

### Dynamic Updates

Streamlit automatically re-runs the page when:
- User changes a dropdown/slider/checkbox
- User clicks a button
- Data is refreshed

```python
# When user changes filter, page re-runs
selected_risk = st.selectbox("Filter by Risk", ['All', 'Critical', 'High', 'Medium', 'Low'])

if selected_risk != 'All':
    filtered_df = df[df['RiskCategory'] == selected_risk]
else:
    filtered_df = df

# Chart updates automatically with filtered data
st.plotly_chart(px.pie(filtered_df, ...))
```

---

## 11. Why SignUp Works But Doesn't Persist After Logout

### The Issue

When a user signs up:
1. ✅ Account is created successfully
2. ✅ User can log in immediately
3. ❌ After logout and re-login, the account may not work

### Root Cause

There are **TWO authentication systems** in the code:

| System | File | How It Works |
|--------|------|--------------|
| **Demo Users (Hardcoded)** | `utils/auth.py` | Dictionary of fixed users (advisor@hsu.edu, admin@hsu.edu) |
| **Database Users** | `database/db_manager.py` | SQLite table with registered users |

The login function checks **database first**, then **demo users**, then **CSV students**:

```python
def authenticate_user(email, password):
    # 1. Check database
    user = db_check(email, password)
    if user:
        return user
    
    # 2. Check hardcoded demo users
    if email in DEMO_USERS:
        return DEMO_USERS[email]
    
    # 3. Check CSV student emails
    student = get_student_from_csv(email)
    if student:
        return student
    
    return None  # Login failed
```

### Why It Sometimes Fails

1. **Database file might reset** - If `hsu_database.db` is deleted/recreated, registered users are lost
2. **Cache issues** - Streamlit caches data; if cache isn't cleared, old data is used
3. **Path issues** - Database path might be relative and change between runs

### The Fix

To ensure persistence:

```python
# Always use absolute path for database
import os
DB_PATH = os.path.join(os.path.dirname(__file__), 'hsu_database.db')

# Don't recreate database if it exists
if not os.path.exists(DB_PATH):
    create_tables()

# Clear cache on logout
def logout():
    st.cache_data.clear()  # Important!
    for key in st.session_state.keys():
        del st.session_state[key]
```

---

## 12. Deployment Guide

### How to Run Locally

```bash
# 1. Navigate to project folder
cd HSU-Streamlit-App

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. Open browser at http://localhost:8501
```

### How to Deploy on Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set main file: `app.py`
5. Click "Deploy"

### How to Deploy on Heroku

```bash
# Procfile contains:
web: sh setup.sh && streamlit run app.py --server.port $PORT

# setup.sh contains:
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

---

## 13. Policies & Permissions Design

### Role-Based Access Matrix

| Feature | Student | Advisor | Admin |
|---------|---------|---------|-------|
| View own data | ✅ | ❌ | ❌ |
| View all students | ❌ | ✅ | ✅ |
| Filter/search students | ❌ | ✅ | ✅ |
| View analytics | ❌ | ✅ | ✅ |
| Schedule meetings | ✅ (own) | ✅ | ✅ |
| Create interventions | ❌ | ✅ | ✅ |
| Send emails | ❌ | ✅ | ✅ |
| Manage users | ❌ | ❌ | ✅ |
| View audit logs | ❌ | ❌ | ✅ |
| Access ML predictions | ❌ | ✅ | ✅ |

### How Permissions are Enforced

```python
# At the top of each protected page

# Method 1: Check authentication
if not st.session_state.get("authenticated"):
    st.warning("Please log in")
    st.stop()

# Method 2: Check specific role
if st.session_state.get("role") not in ['advisor', 'admin']:
    st.error("Access Denied")
    st.stop()

# Method 3: Hide navigation items using CSS
st.markdown("""
    <style>
    /* Hide Student Portal from advisors */
    [data-testid="stSidebarNav"] li:has(a[href*="Student_Portal"]) {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)
```

### Data Privacy

| Data Type | Who Can See | How Protected |
|-----------|-------------|---------------|
| Student personal info | Student (own) + Advisor | Role check |
| Risk scores | Advisors + Admins | Role check |
| Counseling notes | Only counselors | Not shown in app |
| Login history | Advisors (aggregated) | No individual details |
| Passwords | Nobody | SHA-256 hashed |

### Audit Logging

Every action is logged to `audit_logs` table:

```python
# database/db_manager.py

def log_action(self, user_id, action, entity_type=None, entity_id=None):
    """Log user action for audit trail"""
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (user_id, action, entity_type, entity_id)
            VALUES (?, ?, ?, ?)
        """, (user_id, action, entity_type, entity_id))

# Usage:
db.log_action(user_id, 'USER_LOGIN')
db.log_action(user_id, 'INTERVENTION_CREATED', 'interventions', intervention_id)
db.log_action(user_id, 'EMAIL_SENT', 'students', student_id)
```

---

## Summary Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    HSU EARLY WARNING SYSTEM ARCHITECTURE                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│   │   STUDENT    │     │   ADVISOR    │     │    ADMIN     │              │
│   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘              │
│          │                    │                    │                       │
│          ▼                    ▼                    ▼                       │
│   ┌────────────────────────────────────────────────────────────┐          │
│   │                    STREAMLIT FRONTEND                       │          │
│   │   (app.py + pages/*.py + utils/premium_design.py)          │          │
│   └─────────────────────────────┬──────────────────────────────┘          │
│                                 │                                          │
│   ┌─────────────────────────────┼──────────────────────────────┐          │
│   │                    PYTHON BACKEND                           │          │
│   │                             │                               │          │
│   │   ┌─────────────┐   ┌──────┴───────┐   ┌─────────────┐     │          │
│   │   │ utils/      │   │ database/    │   │ ml_pipeline/│     │          │
│   │   │ auth.py     │   │ db_manager.py│   │ *.py        │     │          │
│   │   │ data_loader │   │              │   │             │     │          │
│   │   └──────┬──────┘   └──────┬───────┘   └──────┬──────┘     │          │
│   │          │                 │                  │             │          │
│   └──────────┼─────────────────┼──────────────────┼─────────────┘          │
│              │                 │                  │                        │
│              ▼                 ▼                  ▼                        │
│   ┌──────────────┐   ┌────────────────┐   ┌─────────────────┐             │
│   │  Data_Web/   │   │ hsu_database   │   │  models/        │             │
│   │  *.csv       │   │ .db (SQLite)   │   │  *.joblib       │             │
│   │  (152 students)│ │ (23 tables)    │   │  (ML Model)     │             │
│   └──────────────┘   └────────────────┘   └─────────────────┘             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Commands

```bash
# Run locally
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Test database connection
python database/db_manager.py

# Run ML pipeline
python ml_pipeline/src/ml_pipeline_complete.py

# Migrate CSV to database
python database/migrate_csv_to_db.py
```

---

**Document Created:** December 3, 2025  
**Author:** Team Infinite - Group 6  
**Version:** 3.0 Premium Edition

---

*If you have questions, contact: support@hsu.edu*
