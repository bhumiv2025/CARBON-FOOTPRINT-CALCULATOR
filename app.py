import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Generate synthetic dataset
# -----------------------------
np.random.seed(42)
n = 1000

distance = np.random.randint(1, 200, n)              # km per month
transport = np.random.choice(["car", "bus", "train", "bike"], n)
electricity = np.random.randint(50, 500, n)          # kWh per month
diet = np.random.choice(["veg", "non-veg"], n)
waste = np.random.randint(5, 50, n)                  # kg per month

transport_factors = {"car": 0.21, "bus": 0.08, "train": 0.05, "bike": 0.02}
diet_factors = {"veg": 100, "non-veg": 200}

footprint = (
    distance * [transport_factors[t] for t in transport] +
    electricity * 0.85 +
    [diet_factors[d] for d in diet] +
    waste * 2.5
)

df = pd.DataFrame({
    "distance": distance,
    "transport": transport,
    "electricity": electricity,
    "diet": diet,
    "waste": waste,
    "footprint": footprint
})

# Encode categorical variables
le_transport = LabelEncoder()
df["transport_enc"] = le_transport.fit_transform(df["transport"])

le_diet = LabelEncoder()
df["diet_enc"] = le_diet.fit_transform(df["diet"])

X = df[["distance", "transport_enc", "electricity", "diet_enc", "waste"]]
y = df["footprint"]

# -----------------------------
# Train models
# -----------------------------
# Regression model
reg_model = LinearRegression()
reg_model.fit(X, y)

# Classification model
df["category"] = pd.cut(df["footprint"],
                        bins=[0, 300, 600, np.inf],
                        labels=["Low", "Medium", "High"])
clf = RandomForestClassifier(random_state=42)
clf.fit(X, df["category"])

# Clustering model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 AI/ML Carbon Footprint Calculator")

st.markdown("Enter your lifestyle details and let AI/ML models predict your **carbon footprint** 🚀")

# User inputs
distance_in = st.slider("🚗 Distance travelled per month (km)", 1, 500, 50)
transport_in = st.selectbox("🚌 Transport mode", ["car", "bus", "train", "bike"])
electricity_in = st.slider("💡 Electricity usage (kWh/month)", 50, 1000, 200)
diet_in = st.radio("🍽️ Diet type", ["veg", "non-veg"])
waste_in = st.slider("🗑️ Waste generated (kg/month)", 1, 100, 20)

# Encode inputs
transport_enc = le_transport.transform([transport_in])[0]
diet_enc = le_diet.transform([diet_in])[0]
user_data = [[distance_in, transport_enc, electricity_in, diet_enc, waste_in]]

# Predictions
if st.button("🔮 Predict My Footprint"):
    footprint_pred = reg_model.predict(user_data)[0]
    category_pred = clf.predict(user_data)[0]
    cluster_pred = kmeans.predict(scaler.transform(user_data))[0]

    st.subheader("📊 Results")
    st.write(f"**Estimated Carbon Footprint:** {footprint_pred:.2f} kg CO₂/month")
    st.write(f"**Category:** {category_pred}")
    st.write(f"**Cluster (lifestyle group):** {cluster_pred}")

    # Simple suggestions
    st.subheader("🌱 Suggestions to Reduce Footprint")
    if transport_in == "car" and distance_in > 100:
        st.write("- Try using public transport or carpooling to reduce emissions.")
    if electricity_in > 400:
        st.write("- Consider switching to energy-efficient appliances or reducing electricity use.")
    if diet_in == "non-veg":
        st.write("- Reducing meat consumption can significantly lower your footprint.")
    if waste_in > 30:
        st.write("- Recycle or compost to reduce waste-related emissions.")
