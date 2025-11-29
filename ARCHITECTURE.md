# HSU Early Warning System – Enterprise Architecture

This document describes the end‑to‑end architecture of the HSU Early Warning System, following three main layers: **Business Architecture**, **Information Systems (IS) Architecture**, and **Technology Architecture**. It is intended to complement the main README and serve as the reference architecture for the project report and portfolio site.

---

## 1. Solution Architecture Overview

The HSU Early Warning System is an integrated platform that:

- Consolidates academic, engagement, financial, and wellness data for each student.
- Uses a machine‑learning model to generate an `OverallRiskScore` and `RiskCategory`.
- Exposes role‑based dashboards for students, advisors, and administrators.

At a high level, the architecture can be viewed as three columns:

1. **Business Architecture** – the student success processes and stakeholders.
2. **IS Architecture** – the data hub, ML pipeline, and Streamlit application.
3. **Technology Architecture** – infrastructure, security, and supporting tools.

> **Figure 1 – Enterprise Solution Architecture**  
> *(Insert your architecture diagram image here, e.g. `docs/architecture_solution.png`.)*

---

## 2. Business Architecture

The business layer focuses on how Horizon State University (HSU) manages student success and retention.

### 2.1 Core Business Processes

- **Student Success & Advising**  
  Risk‑based advising workflows, outreach campaigns, appointment scheduling, and follow‑up tracking.

- **Academic Operations**  
  Monitoring course enrollment, grades, GPA trends, attendance, and LMS engagement for early signs of struggle.

- **Financial & Wellness Support**  
  Identifying students with large balances, financial holds, or high‑severity counseling visits and coordinating appropriate support.

### 2.2 Stakeholders

- **Students** – receive transparent feedback about their performance, risk, and available resources via the Student Portal.
- **Academic Advisors & Success Coaches** – use the Advisor Dashboard and Students Directory to identify who needs outreach and why.
- **Student Success / Retention Office** – analyze cohort‑level risk and intervention coverage.
- **Administrators** – track retention, equity gaps, and program impact through the Admin Portal.

---

## 3. Information Systems Architecture

The IS layer connects data sources, analytics, and user interfaces.

### 3.1 Data Layer (Student Data Hub)

In the prototype, the data hub is built from **synthetic CSV datasets** and an optional SQLite database stored under `Data_Web/` and `database/`:

- **Students** – demographics, classification, first‑gen, international, HS GPA.
- **Enrollments & Courses** – course registrations per term.
- **Grades** – assignment‑level performance feeding CalculatedGPA.
- **Attendance & LMS Logins** – engagement metrics by course and student.
- **Payments** – amounts owed/paid, balances, holds, financial aid.
- **Counseling** – visits, concern types, severity, crisis flags.
- **Risk Scores** – `OverallRiskScore` and `RiskCategory` per student.

These tables are joined primarily on `StudentID` and `EnrollmentID`, forming the logical student data hub used by both the ML pipeline and the dashboards.

### 3.2 Analytics & ML Layer

- **Feature Engineering Pipeline (ml_pipeline/)**  
  Creates ~69 engineered features across demographic, academic, engagement, financial, wellness, pathway, and interaction dimensions.

- **Risk Model**  
  A Random Forest classifier (plus preprocessing) that outputs a probability of dropout risk and converts it into `OverallRiskScore` and `RiskCategory`.

- **Batch Scoring & Outputs**  
  Weekly or term‑level predictions are written back to `risk_scores.csv` and consumed by the Streamlit app.

### 3.3 Application Layer (Streamlit UI)

- **Advisor Dashboard (`1_🏠_Dashboard.py`)** – KPIs and prioritized list of at‑risk students.
- **Students Directory (`2_👥_Students.py`)** – searchable list and multi‑tab profile pages.
- **Analytics (`3_📊_Analytics.py`)** – cohort analytics and equity charts.
- **Student Portal (`4_🎓_Student_Portal.py`)** – personalized status for each student.
- **Admin Portal (`5_👔_Admin_Portal.py`)** – executive KPIs and configuration.
- **ML Predictions (`6_🎯_ML_Predictions.py`)** – single and batch prediction views.
- **Interventions (`7_📝_Interventions.py`)** – intervention logging and analysis.

Shared services such as authentication, data loading, and design live in `utils/`.

---

## 4. Technology Architecture

The technology layer describes the enabling platforms, infrastructure, and security controls.

### 4.1 Infrastructure & Platforms

- **Computation & Application Runtime**  
  - Python 3.10+ runtime.  
  - Streamlit framework for rapid dashboard development.  
  - scikit‑learn, pandas, NumPy for ML and data processing.

- **Storage**  
  - CSV files and SQLite database for the prototype.  
  - In a production deployment, these would be replaced by a cloud data warehouse (e.g., PostgreSQL on Azure/AWS) and object storage.

- **Deployment**  
  - Streamlit Community Cloud (or similar PaaS) hosting `app.py`.  
  - GitHub repository as the code base and documentation site.

### 4.2 Security, Access & Governance

- **Authentication & Authorization**  
  Role‑based access implemented in `utils/auth.py` with Student, Advisor, and Admin roles.

- **Data Protection**  
  In the prototype, only synthetic data is used; in production, TLS encryption, database access controls, and auditing would be required.

- **Governance & Monitoring**  
  Logging of model runs and dashboard usage, plus periodic review of model performance, fairness, and data quality.

> **Figure 2 – Tools & Platforms Architecture**  
> *(Insert a second diagram here if desired, e.g. `docs/architecture_tools_platforms.png`, focusing on concrete tools, cloud services, and security components.)*

---

This architecture can be exported as diagrams for the appendix (Enterprise Architecture and Tools/Platforms Architecture) and referenced throughout the final report.
