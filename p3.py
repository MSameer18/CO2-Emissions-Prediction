import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title and Introduction
st.title("🚗 CO2 Emissions Prediction")
st.write(
    "Predict vehicle CO2 emissions based on engine size, cylinders, and fuel consumption. "
    "Compare models, explore feature importance, estimate annual fuel costs, and get recommendations for reducing emissions."
)

# Sidebar for user input
st.sidebar.header("Input Features for Prediction")
engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 8.4, 3.0, 0.1)
cylinders = st.sidebar.slider("Cylinders", 3, 12, 6, 1)
fuel_consumption = st.sidebar.slider("Fuel Consumption (L/100km)", 4.7, 25.8, 8.5, 0.1)


# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("FuelConsumptionCo2.csv")


data = load_data()

# Rename columns for user-friendly names
data.rename(
    columns={
        "ENGINESIZE": "Engine Size (L)",
        "CYLINDERS": "Cylinders",
        "FUELCONSUMPTION_COMB": "Fuel Consumption (L/100km)",
        "CO2EMISSIONS": "CO2 Emissions (g/km)"
    },
    inplace=True
)

# Show raw data option
if st.checkbox("Show raw dataset"):
    st.write(data)

# Split data into training and testing
X = data[["Engine Size (L)", "Cylinders", "Fuel Consumption (L/100km)"]]
y = data["CO2 Emissions (g/km)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Model": model
    }

# Show model evaluation metrics
st.subheader("Model Evaluation Metrics")
model_comparison = pd.DataFrame(results).T[["MAE", "MSE", "R2"]].sort_values("R2", ascending=False)
st.write(model_comparison)

# Select the best model for prediction
best_model_name = model_comparison.index[0]
best_model = results[best_model_name]["Model"]
st.write(f"Best Model: **{best_model_name}**")

# Predict CO2 Emission for user inputs
input_data = np.array([[engine_size, cylinders, fuel_consumption]])
predicted_emission = best_model.predict(input_data)[0]
st.sidebar.subheader("Predicted CO2 Emission")
st.sidebar.write(f"{predicted_emission:.2f} g/km")

# SHAP Feature Importance
st.subheader("Feature Importance with SHAP")
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)
    X_train_renamed = pd.DataFrame(X_train, columns=["Engine Size (L)", "Cylinders", "Fuel Consumption (L/100km)"])

    # Create SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_train_renamed, plot_type="bar", show=False)

    # Override x-axis label
    ax = plt.gca()  # Get current axis
    ax.set_xlabel("Average Impact on Prediction (g/km)", fontsize=12)  # Update label

    # Display SHAP plot
    st.pyplot(fig)
except Exception as e:
    st.error(f"An error occurred while generating SHAP explanations: {e}")

# Visualization: CO2 Emissions vs Engine Size
st.subheader("Visualization")
fig = px.scatter(
    data,
    x="Engine Size (L)",
    y="CO2 Emissions (g/km)",
    color="Cylinders",
    title="CO2 Emissions vs Engine Size",
    labels={"Engine Size (L)": "Engine Size (L)", "CO2 Emissions (g/km)": "CO2 Emissions (g/km)"},
    hover_data=["Fuel Consumption (L/100km)"],
)
st.plotly_chart(fig)

# Annual Fuel Cost Estimator
st.subheader("Annual Fuel Cost Estimator")
fuel_prices = {"Gasoline": 1.5, "Diesel": 1.3, "Electric": 0.2}  # Prices per liter/kWh
fuel_type = st.radio("Select Fuel Type", list(fuel_prices.keys()))
annual_distance = st.number_input("Enter distance driven annually (km):", min_value=0, step=1000, value=15000)

# Calculate dynamic fuel cost
fuel_cost = annual_distance * fuel_consumption / 100 * fuel_prices[fuel_type]
st.write(f"**Estimated Annual Fuel Cost for {fuel_type}: ${fuel_cost:.2f}**")

# Update comparison of fuel costs dynamically
fuel_costs = {ftype: annual_distance * fuel_consumption / 100 * price for ftype, price in fuel_prices.items()}
fuel_cost_df = pd.DataFrame(list(fuel_costs.items()), columns=["Fuel Type", "Annual Cost"])
fig = px.bar(
    fuel_cost_df,
    x="Fuel Type",
    y="Annual Cost",
    title="Annual Fuel Costs by Type",
    labels={"Annual Cost": "Cost ($)", "Fuel Type": "Type of Fuel"},
)
st.plotly_chart(fig)
