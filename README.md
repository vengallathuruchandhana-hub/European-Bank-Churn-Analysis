# Customer Segmentation & Churn Pattern Analytics in European Banking

## Project Overview
This project analyzes customer churn patterns in a European banking dataset using Python, Machine Learning, and Streamlit. The goal is to identify customers who are likely to leave the bank, understand the factors influencing churn, and provide actionable business recommendations.

## Business Problem
Customer churn affects customer retention and long-term profitability. This project uses customer data to:
- Analyze overall customer churn.
- Identify high-risk customer segments.
- Discover important factors associated with churn.
- Build a machine learning model to predict customer churn.
- Provide actionable retention recommendations.

## Dataset
- Total records: 10,000 customers
- Total original columns: 14
- Target variable: `Exited`
- Dataset year: 2025

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Plotly
- Scikit-learn
- Joblib
- Streamlit

## Project Workflow
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Demographic customer segmentation
4. Behavioral customer segmentation
5. Machine learning model development
6. Model evaluation and comparison
7. Customer churn risk analysis
8. Business insights and recommendations
9. Interactive Streamlit dashboard
10. Individual customer churn prediction

## Key Findings
- Overall customer churn rate: 20.37%.
- Germany had the highest geographical churn rate at 32.44%.
- Female customers had a churn rate of 25.07%.
- Inactive customers had a churn rate of 26.85%.
- Age was the most influential feature in the tuned Random Forest model.

## Model Performance

The tuned Random Forest model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 82.30% |
| Precision | 54.97% |
| Recall | 71.99% |
| F1-Score | 62.34% |
| ROC-AUC | 86.10% |

## Interactive Dashboard
The Streamlit dashboard includes:
- Executive overview
- Churn analysis
- Customer segmentation
- Model performance
- Customer risk analysis
- Business recommendations
- Individual customer churn prediction

## Project Structure
```text
European-Bank-Churn-Analysis/
│
├── dashboard/
│   └── app.py
├── data/
├── documentation/
├── outputs/
│   ├── models/
│   ├── modeling/
│   └── visualizations/
├── reports/
│   └── final_business_insights_report.txt
├── src/
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd European-Bank-Churn-Analysis
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the environment (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the dashboard
```bash
python -m streamlit run dashboard/app.py
```

## Business Recommendations
- Prioritize customer retention campaigns for high-risk customer groups.
- Investigate the reasons behind higher churn in Germany.
- Improve engagement among inactive customers.
- Develop targeted retention strategies based on customer age and product usage.
- Monitor customer churn risk using predictive analytics.

## Conclusion
This project demonstrates the use of exploratory data analysis, customer segmentation, machine learning, and interactive dashboards to understand and predict customer churn. The findings can support data-driven customer retention strategies in banking.

## Author
**Chandhana Vengallathuru**

Data Science Portfolio Project
