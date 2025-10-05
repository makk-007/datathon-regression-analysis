import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


# ==============================================
# 🧩 FUNCTION: Load and Cache Data
# ==============================================
@st.cache_data
def load_data(uploaded_file):
    """
    Reads a CSV file uploaded by the user and returns a pandas DataFrame.
    Streamlit's caching ensures faster reloads on repeated runs.
    """
    data = pd.read_csv(uploaded_file)
    return data


# ==============================================
# 🧩 FUNCTION: Train and Evaluate Model
# ==============================================
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    """
    Trains a regression model, makes predictions, and returns evaluation metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ==============================================
# 🚀 MAIN FUNCTION: Regression Section
# ==============================================
def regression_section():
    """
    Interactive regression dashboard allowing users to upload data,
    select models, tune hyperparameters, train, and compare performance.
    """
    st.title("📊 Regression Model Comparison Dashboard")
    st.write(
        "Upload a dataset, select regression models, tune hyperparameters, and compare performance interactively."
    )

    # -----------------------------
    # File Upload Section
    # -----------------------------
    uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])

    if not uploaded_file:
        st.info("👆 Upload a CSV file to begin.")
        return

    data = load_data(uploaded_file)
    st.subheader("📄 Preview of Dataset")
    st.dataframe(data.head())

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    with st.expander("📈 Explore Feature Correlations"):
        st.write("Heatmap showing linear relationships between numeric variables.")
        corr = data.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr, cmap="coolwarm", interpolation="nearest")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im)
        st.pyplot(fig)

    # -----------------------------
    # Feature Selection
    # -----------------------------
    st.subheader("🧮 Feature Selection")
    target_col = st.selectbox("🎯 Select Target Column (Y)", options=data.columns)
    feature_cols = st.multiselect(
        "🧩 Select Feature Columns (X)",
        options=[col for col in data.columns if col != target_col],
    )

    if len(feature_cols) == 0:
        st.warning("⚠️ Please select at least one feature column.")
        return

    X = data[feature_cols]
    y = data[target_col]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    test_size = st.slider("📊 Test Set Size (Percentage)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    st.divider()

    # -----------------------------
    # Model Selection and Tuning
    # -----------------------------
    st.subheader("⚙️ Model & Hyperparameter Selection")

    model_choice = st.selectbox(
        "Select Regression Model or Compare All",
        [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Random Forest Regressor",
            "Compare All Models",
        ],
    )

    # ==================================================
    # 🔄 Compare All Models
    # ==================================================
    if model_choice == "Compare All Models":
        if st.button("🚀 Compare All Models"):
            st.info("Training all models, please wait...")

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=1.0),
                "Random Forest Regressor": RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=42
                ),
            }

            results_list = []
            for name, model in models.items():
                result = train_and_evaluate(model, X_train, X_test, y_train, y_test)
                results_list.append(
                    {
                        "Model": name,
                        "MAE": result["mae"],
                        "MSE": result["mse"],
                        "R²": result["r2"],
                    }
                )

            results_df = pd.DataFrame(results_list).sort_values(
                by="R²", ascending=False
            )

            st.subheader("🏁 Model Comparison Results")
            st.dataframe(
                results_df.style.highlight_max(subset=["R²"], color="lightgreen")
            )

            # Highlight best model
            best_model = results_df.iloc[0]
            st.success(
                f"🏆 **Best Model:** {best_model['Model']} (R² = {best_model['R²']:.4f})"
            )

            # -----------------------------
            # Improved Comparison Visuals
            # -----------------------------
            st.subheader("📊 Model Performance Overview")

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].bar(results_df["Model"], results_df["MAE"], color="orange")
            ax[0].set_title("Mean Absolute Error (↓)")
            ax[0].set_xticklabels(results_df["Model"], rotation=20, ha="right")

            ax[1].bar(results_df["Model"], results_df["MSE"], color="red")
            ax[1].set_title("Mean Squared Error (↓)")
            ax[1].set_xticklabels(results_df["Model"], rotation=20, ha="right")

            ax[2].bar(results_df["Model"], results_df["R²"], color="skyblue")
            ax[2].set_title("R² Score (↑)")
            ax[2].set_xticklabels(results_df["Model"], rotation=20, ha="right")

            plt.tight_layout()
            st.pyplot(fig)

            # -----------------------------
            # Download Results
            # -----------------------------
            st.download_button(
                label="📥 Download Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="regression_comparison_results.csv",
                mime="text/csv",
            )

        return  # Exit here after comparison

    # ==================================================
    # 🧠 Single Model Mode
    # ==================================================
    if model_choice == "Linear Regression":
        model = LinearRegression()

    elif model_choice == "Ridge Regression":
        alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
        model = Ridge(alpha=alpha)

    elif model_choice == "Lasso Regression":
        alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
        model = Lasso(alpha=alpha)

    elif model_choice == "Random Forest Regressor":
        n_estimators = st.slider(
            "Number of Trees (n_estimators)", 10, 300, 100, step=10
        )
        max_depth = st.slider("Max Depth", 1, 30, 5)
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )

    # -----------------------------
    # Train and Evaluate Selected Model
    # -----------------------------
    if st.button("🚀 Train Model"):
        results = train_and_evaluate(model, X_train, X_test, y_train, y_test)

        st.success(f"✅ Model trained successfully: {model_choice}")
        st.subheader("📈 Performance Metrics")
        st.write(f"**Mean Absolute Error (MAE):** {results['mae']:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {results['mse']:.4f}")
        st.write(f"**R² Score:** {results['r2']:.4f}")

        # Plot Actual vs Predicted
        st.subheader("📉 Actual vs Predicted Plot")
        fig, ax = plt.subplots()
        ax.scatter(results["y_test"], results["y_pred"], color="blue", alpha=0.6)
        ax.plot(
            [results["y_test"].min(), results["y_test"].max()],
            [results["y_test"].min(), results["y_test"].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{model_choice} - Actual vs Predicted")
        st.pyplot(fig)

        # Show feature importances for Random Forest
        if model_choice == "Random Forest Regressor":
            st.subheader("🌲 Feature Importances")
            importance_df = pd.DataFrame(
                {"Feature": feature_cols, "Importance": model.feature_importances_}
            ).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df)
            st.bar_chart(importance_df.set_index("Feature"))
