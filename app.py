import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Title
st.title("📊 Data Analysis App")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.header("📁 Dataset Preview")
    st.write(df.head())

    st.markdown("---")

    # Column selection
    st.header("⚙️ Select Analysis")

    analysis_type = st.selectbox(
        "Choose Analysis Type",
        ["Select", "Summary", "Regression"]
    )

    # 📌 Summary stats
    if analysis_type == "Summary":
        st.subheader("📈 Summary Statistics")
        st.write(df.describe())

    # 📌 Regression
    elif analysis_type == "Regression":
        st.subheader("📉 Linear Regression")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns")
        else:
            y_var = st.selectbox("Select Dependent Variable (Y)", numeric_cols)
            x_var = st.selectbox("Select Independent Variable (X)", numeric_cols)

            if st.button("Run Regression"):
                try:
                    model = smf.ols(f"{y_var} ~ {x_var}", data=df).fit()
                    st.text(model.summary())
                except:
                    st.error("Error running regression")

    st.markdown("---")
    st.success("✅ Done")

else:
    st.info("👈 Please upload a CSV file from the sidebar")