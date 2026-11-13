
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib
import numpy as np
# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="European Bank | Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODELING_DIR = OUTPUT_DIR / "modeling"

# ==========================================
# 2. LOAD DATA
# ==========================================

@st.cache_data
def load_data(file_path):
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


customer_df = load_data(
    BASE_DIR / "data" / "raw" / "European_Bank.csv"
)

demographic_df = load_data(
    OUTPUT_DIR / "customer_segment_business_profile.csv"
)

behavioral_df = load_data(
    OUTPUT_DIR / "behavioral_cluster_profile.csv"
)

feature_df = load_data(
    MODELING_DIR / "random_forest_feature_importance.csv"
)

risk_df = load_data(
    MODELING_DIR / "customer_churn_risk_summary.csv"
)

model_df = load_data(
    MODELING_DIR / "final_model_comparison.csv"
)


# ==========================================
# LOAD TRAINED CHURN MODEL
# ==========================================

MODEL_PATH = OUTPUT_DIR / "models" / "tuned_random_forest_model.pkl"

@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)

churn_model = load_churn_model()


# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🏦 European Banking Customer Churn Analytics")

st.markdown(
    """
    An interactive analytics dashboard for understanding customer churn,
    customer segmentation, prediction performance and retention strategies.
    """
)

st.divider()

# Check the main dataset
if customer_df is None:
    st.error(
        "Dataset not found. Please check "
        "data/raw/European_Bank.csv"
    )
    st.stop()

# ==========================================
# 4. SIDEBAR FILTERS
# ==========================================

st.sidebar.title("Dashboard Filters")

geography_options = sorted(
    customer_df["Geography"].dropna().unique()
)

selected_geography = st.sidebar.multiselect(
    "Select Geography",
    geography_options,
    default=geography_options
)

gender_options = sorted(
    customer_df["Gender"].dropna().unique()
)

selected_gender = st.sidebar.multiselect(
    "Select Gender",
    gender_options,
    default=gender_options
)

filtered_df = customer_df[
    customer_df["Geography"].isin(selected_geography)
    & customer_df["Gender"].isin(selected_gender)
].copy()

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# ==========================================
# 5. DASHBOARD TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Executive Overview",
    "Churn Analysis",
    "Customer Segmentation",
    "Model Performance",
    "Customer Risk",
    "Business Recommendations",
    "Predict Customer Churn"
])


# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================

with tab1:

    st.header("Executive Overview")

    total_customers = len(filtered_df)
    churned_customers = int(filtered_df["Exited"].sum())
    retained_customers = total_customers - churned_customers

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    retention_rate = 100 - churn_rate

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

    st.subheader("Customer Retention Distribution")

    overview_chart = pd.DataFrame({
        "Customer Status": ["Retained", "Churned"],
        "Customers": [retained_customers, churned_customers]
    })

    fig = px.pie(
        overview_chart,
        names="Customer Status",
        values="Customers",
        hole=0.45,
        title="Retained vs Churned Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Current filter selection shows a churn rate of "
        f"{churn_rate:.2f}%."
    )


# ==========================================
# TAB 2: CHURN ANALYSIS
# ==========================================

with tab2:

    st.header("Customer Churn Analysis")

    col1, col2 = st.columns(2)

    # Geography churn
    with col1:

        geography_churn = (
            filtered_df.groupby("Geography")["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            geography_churn,
            x="Geography",
            y="Churn Rate",
            color="Geography",
            title="Churn Rate by Geography",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Gender churn
    with col2:

        gender_churn = (
            filtered_df.groupby("Gender")["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            gender_churn,
            x="Gender",
            y="Churn Rate",
            color="Gender",
            title="Churn Rate by Gender",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Age group churn
    with col3:

        age_df = filtered_df.copy()

        age_df["Age Group"] = pd.cut(
            age_df["Age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[
                "18-25",
                "26-35",
                "36-45",
                "46-55",
                "56-65",
                "Above 65"
            ]
        )

        age_churn = (
            age_df.groupby("Age Group", observed=False)["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            age_churn,
            x="Age Group",
            y="Churn Rate",
            title="Churn Rate by Age Group",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Product churn
    with col4:

        product_churn = (
            filtered_df.groupby("NumOfProducts")["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            product_churn,
            x="NumOfProducts",
            y="Churn Rate",
            title="Churn Rate by Number of Products",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Active member churn
    active_churn = (
        filtered_df.groupby("IsActiveMember")["Exited"]
        .mean()
        .mul(100)
        .reset_index(name="Churn Rate")
    )

    active_churn["Membership"] = active_churn[
        "IsActiveMember"
    ].map({
        0: "Inactive",
        1: "Active"
    })

    fig = px.bar(
        active_churn,
        x="Membership",
        y="Churn Rate",
        title="Churn Rate by Membership Activity",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 3: CUSTOMER SEGMENTATION
# ==========================================

with tab3:

    st.header("Customer Segmentation Analysis")

    if demographic_df is not None:

        st.subheader("Demographic Customer Segments")

        fig = px.bar(
            demographic_df,
            x="Cluster",
            y="Churn_Rate_Percent",
            color="Churn_Rate_Percent",
            title="Churn Rate by Demographic Cluster",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            demographic_df,
            use_container_width=True
        )

    else:
        st.warning("Demographic segment file not found.")

    st.divider()

    if behavioral_df is not None:

        st.subheader("Behavioral Customer Segments")

        fig = px.bar(
            behavioral_df,
            x="Behavioral_Cluster",
            y="Churn_Rate_Percent",
            color="Churn_Rate_Percent",
            title="Churn Rate by Behavioral Cluster",
            text_auto=".2f"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            behavioral_df,
            use_container_width=True
        )

    else:
        st.warning("Behavioral segment file not found.")


# ==========================================
# TAB 4: MODEL PERFORMANCE
# ==========================================

with tab4:

    st.header("Churn Prediction Model Performance")

    if model_df is not None:

        st.subheader("Model Comparison")

        fig = px.bar(
            model_df,
            x="Model",
            y="ROC_AUC",
            color="Model",
            title="ROC-AUC Comparison",
            text_auto=".3f"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            model_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1_Score": "{:.2%}",
                "ROC_AUC": "{:.2%}"
            }),
            use_container_width=True
        )

        best_model = model_df.loc[
            model_df["ROC_AUC"].idxmax()
        ]

        st.success(
            f"Best Model: {best_model['Model']} | "
            f"ROC-AUC: {best_model['ROC_AUC']:.2%}"
        )

    else:
        st.warning("Model comparison file not found.")

    st.divider()

    if feature_df is not None:

        st.subheader("Important Churn Prediction Features")

        top_features = feature_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Feature importance file not found.")


# ==========================================
# TAB 5: CUSTOMER RISK ANALYSIS
# ==========================================

with tab5:

    st.header("Customer Churn Risk Analysis")

    if risk_df is not None:

        st.subheader("Customer Risk Distribution")

        col1, col2, col3 = st.columns(3)

        risk_lookup = risk_df.set_index("Risk_Category")

        for column, category in zip(
            [col1, col2, col3],
            ["High Risk", "Medium Risk", "Low Risk"]
        ):

            if category in risk_lookup.index:

                count = int(
                    risk_lookup.loc[category, "Customer_Count"]
                )

                column.metric(
                    category,
                    f"{count:,} customers"
                )

        fig = px.bar(
            risk_df,
            x="Risk_Category",
            y="Customer_Count",
            color="Risk_Category",
            title="Customers by Risk Category",
            text_auto=True,
            category_orders={
                "Risk_Category": [
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ]
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            risk_df,
            x="Risk_Category",
            y="Average_Churn_Probability",
            color="Risk_Category",
            title="Average Predicted Churn Probability",
            text_auto=".2f",
            category_orders={
                "Risk_Category": [
                    "High Risk",
                    "Medium Risk",
                    "Low Risk"
                ]
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            risk_df,
            use_container_width=True
        )

        st.caption(
            "Risk analysis represents the model's 2,000-customer "
            "test set, not the full 10,000-customer dataset."
        )

    else:
        st.warning("Customer risk summary file not found.")


# ==========================================
# TAB 6: BUSINESS RECOMMENDATIONS
# ==========================================

with tab6:

    st.header("Business Recommendations")

    st.markdown(
        """
        Based on the exploratory analysis, customer segmentation
        and churn prediction results, the following actions
        are recommended.
        """
    )

    recommendations = {
        "1. High-Risk Customer Retention":
            "Prioritize outreach to customers with high predicted churn risk.",

        "2. Geography-Based Strategy":
            "Investigate the higher observed churn rate in Germany and "
            "develop targeted retention initiatives.",

        "3. Customer Engagement":
            "Create engagement programs for inactive customers.",

        "4. Product Optimization":
            "Investigate customer experiences and product combinations "
            "associated with higher churn.",

        "5. Age-Based Personalization":
            "Explore personalized services for older customer groups "
            "with higher observed churn.",

        "6. High-Balance Customers":
            "Monitor high-balance customers with low engagement.",

        "7. Predictive Monitoring":
            "Use churn probabilities to prioritize customer outreach "
            "and monitor retention outcomes.",

        "8. Model Maintenance":
            "Evaluate model performance regularly and retrain when "
            "customer behavior or data changes."
    }

    for title, description in recommendations.items():

        with st.expander(title):
            st.write(description)

    st.divider()

    st.subheader("Project Conclusion")

    st.write(
        """
        This project uses exploratory data analysis, customer
        segmentation and machine learning to understand customer
        attrition in European banking.

        The Tuned Random Forest model achieved an ROC-AUC of 86.10%
        on the test dataset. The analysis identified customer groups
        with different churn patterns and highlighted opportunities
        for more focused customer retention strategies.
        """
    )

    

# ==========================================
# TAB 7: INDIVIDUAL CUSTOMER CHURN PREDICTION
# ==========================================

with tab7:

    st.header("Individual Customer Churn Prediction")

    st.write(
        "Enter customer details to estimate the probability "
        "that the customer may leave the bank."
    )

    with st.form("customer_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            credit_score = st.number_input(
                "Credit Score", 300, 900, 650
            )

            geography = st.selectbox(
                "Geography", ["France", "Germany", "Spain"]
            )

            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            age = st.number_input(
                "Age", 18, 100, 35
            )

            tenure = st.number_input(
                "Tenure (Years)", 0, 10, 5
            )

        with col2:

            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=75000.0
            )

            products = st.selectbox(
                "Number of Products", [1, 2, 3, 4]
            )

            has_card = st.selectbox(
                "Has Credit Card?", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            active_member = st.selectbox(
                "Is Active Member?", [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=100000.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn"
        )

    if submitted:

        customer_input = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active_member,
            "EstimatedSalary": salary
        }])

        try:

            prediction = churn_model.predict(customer_input)[0]

            probability = churn_model.predict_proba(
                customer_input
            )[0][list(churn_model.classes_).index(1)]

            st.divider()
            st.subheader("Prediction Results")

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            if probability >= 0.60:
                st.error("High Risk – Prioritize customer retention.")

            elif probability >= 0.30:
                st.warning("Medium Risk – Monitor this customer.")

            else:
                st.success("Low Risk – Lower estimated churn probability.")

            st.write(
                "Model prediction:",
                "Customer likely to churn" if prediction == 1
                else "Customer likely to stay"
            )

            st.caption(
                "This is a model-based estimate, not a guarantee "
                "of future customer behavior."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")



# ==========================================
# FOOTER
# ==========================================


st.divider()

st.caption(
    "European Banking Customer Segmentation & Churn Pattern Analytics | "
    "Data Science Portfolio Project"
)
