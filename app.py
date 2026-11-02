import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Real Estate Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium appearance
st.markdown("""
<style>
    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }
    
    /* App header */
    .hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Result card */
    .valuation-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        box-shadow: 0 12px 28px -6px rgba(37, 99, 235, 0.35);
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .val-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #93C5FD;
        font-weight: 600;
    }
    .val-amount {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin: 8px 0;
    }
    .val-exact {
        font-size: 1.05rem;
        color: #DBEAFE;
        font-weight: 500;
    }
    
    /* Property spec pill */
    .spec-pill {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #E2E8F0;
    }
    
    /* Card wrapper for input sections */
    .input-section-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADER (PKL WITH TRAIN FALLBACK)
# ==========================================
FEATURE_COLUMNS = [
    "area_sqft", "bedrooms", "bathrooms", "location_score",
    "property_age", "distance_city_km", "near_school",
    "near_metro", "crime_rate_index", "sale_year", "sale_month"
]

MODEL_FILE = "house_price_model.pkl"
DATA_FILE = "Advanced_Regression_HousePrice_Dataset_3800.csv"


@st.cache_resource
def get_prediction_model():
    """Load model from .pkl file if available, or train and cache."""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    # Fallback: train and save
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
        data["sale_date"] = pd.to_datetime(data["sale_date"])
        data["sale_year"] = data["sale_date"].dt.year
        data["sale_month"] = data["sale_date"].dt.month
        y = data["house_price_inr"]
        X = data[FEATURE_COLUMNS]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, MODEL_FILE)
        return rf_model

    raise FileNotFoundError("Neither model file nor dataset found.")


model = get_prediction_model()


# ==========================================
# CURRENCY & NUMBER FORMATTERS
# ==========================================
def format_inr_short(amount):
    """Format into standard Lakhs / Crores string."""
    amount = float(amount)
    if amount >= 10000000:
        return f"₹ {amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹ {amount / 100000:.2f} L"
    else:
        return f"₹ {amount:,.0f}"


def format_inr_exact(amount):
    """Formatted comma-separated Indian Rupee amount."""
    return f"₹ {float(amount):,.0f}"


def calculate_estimated_emi(loan_amount, interest_rate=8.5, tenure_years=20):
    """Calculate approximate monthly mortgage EMI (assuming 80% loan-to-value)."""
    p = loan_amount * 0.80
    r = (interest_rate / 12) / 100
    n = tenure_years * 12
    if r == 0:
        return p / n
    emi = (p * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return emi


# ==========================================
# PRESET PROFILES DEFINITION
# ==========================================
PRESETS = {
    "Custom Configuration": {
        "area_sqft": 1750,
        "bedrooms": 3,
        "bathrooms": 3,
        "property_age": 12,
        "location_score": 6.8,
        "distance_city_km": 12.0,
        "crime_rate_index": 3.8,
        "near_school": "Yes",
        "near_metro": "No",
        "sale_year": 2024,
        "sale_month": 6
    },
    "Prime Luxury Villa": {
        "area_sqft": 3400,
        "bedrooms": 5,
        "bathrooms": 5,
        "property_age": 2,
        "location_score": 9.4,
        "distance_city_km": 4.5,
        "crime_rate_index": 1.2,
        "near_school": "Yes",
        "near_metro": "Yes",
        "sale_year": 2024,
        "sale_month": 8
    },
    "Suburban Family Home": {
        "area_sqft": 1900,
        "bedrooms": 3,
        "bathrooms": 3,
        "property_age": 10,
        "location_score": 7.2,
        "distance_city_km": 11.5,
        "crime_rate_index": 3.2,
        "near_school": "Yes",
        "near_metro": "Yes",
        "sale_year": 2024,
        "sale_month": 5
    },
    "City Metro Apartment": {
        "area_sqft": 1250,
        "bedrooms": 2,
        "bathrooms": 2,
        "property_age": 6,
        "location_score": 8.0,
        "distance_city_km": 7.0,
        "crime_rate_index": 2.8,
        "near_school": "Yes",
        "near_metro": "Yes",
        "sale_year": 2024,
        "sale_month": 7
    },
    "Budget Starter Flat": {
        "area_sqft": 800,
        "bedrooms": 2,
        "bathrooms": 1,
        "property_age": 25,
        "location_score": 4.6,
        "distance_city_km": 22.0,
        "crime_rate_index": 5.4,
        "near_school": "No",
        "near_metro": "Yes",
        "sale_year": 2024,
        "sale_month": 3
    },
    "Outskirts Independent House": {
        "area_sqft": 1400,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_age": 35,
        "location_score": 3.4,
        "distance_city_km": 30.0,
        "crime_rate_index": 7.5,
        "near_school": "No",
        "near_metro": "No",
        "sale_year": 2024,
        "sale_month": 10
    }
}


# ==========================================
# SESSION STATE PRESET HANDLER
# ==========================================
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Custom Configuration"

# Load initial defaults
for k, v in PRESETS[st.session_state.selected_preset].items():
    if f"input_{k}" not in st.session_state:
        st.session_state[f"input_{k}"] = v


def on_preset_change():
    """Update input state when a preset is picked."""
    choice = st.session_state.preset_selector
    st.session_state.selected_preset = choice
    for key, val in PRESETS[choice].items():
        st.session_state[f"input_{key}"] = val


# ==========================================
# APP HEADER
# ==========================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">
        <span>🏠</span>
        <span>House Price Prediction</span>
    </div>
    <div class="hero-subtitle">
        Instant real estate valuation powered by trained regression intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Preset Selector Bar
s_col1, s_col2 = st.columns([1.5, 2.5])
with s_col1:
    st.selectbox(
        "Choose Property Profile / Preset:",
        options=list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.selected_preset),
        key="preset_selector",
        on_change=on_preset_change
    )
with s_col2:
    st.markdown("<div style='padding-top: 28px;'>", unsafe_allow_html=True)
    if st.session_state.selected_preset != "Custom Configuration":
        st.info(f"Loaded **{st.session_state.selected_preset}** profile. You can still adjust any value below.")
    else:
        st.caption("Configuring custom property parameters.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ==========================================
# PROPERTY INPUT CONTROLS
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="input-section-card">
        <div class="section-title">📐 Dimensions & Living Space</div>
    """, unsafe_allow_html=True)

    area_sqft = st.slider(
        "Built-up Area (sq. ft.)",
        min_value=500,
        max_value=4000,
        step=25,
        key="input_area_sqft"
    )

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=8,
            step=1,
            key="input_bedrooms"
        )
    with c_b2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=6,
            step=1,
            key="input_bathrooms"
        )

    property_age = st.slider(
        "Property Age (Years)",
        min_value=0,
        max_value=80,
        step=1,
        key="input_property_age"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="input-section-card">
        <div class="section-title">📅 Valuation Timeline</div>
    """, unsafe_allow_html=True)

    t_c1, t_c2 = st.columns(2)
    with t_c1:
        sale_year = st.selectbox(
            "Sale Year",
            options=list(range(2010, 2031)),
            key="input_sale_year"
        )
    with t_c2:
        sale_month = st.selectbox(
            "Sale Month",
            options=list(range(1, 13)),
            key="input_sale_month"
        )

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class="input-section-card">
        <div class="section-title">📍 Location & Neighborhood Index</div>
    """, unsafe_allow_html=True)

    location_score = st.slider(
        "Location Quality Score (1.0 - 10.0)",
        min_value=1.0,
        max_value=10.0,
        step=0.1,
        key="input_location_score"
    )

    distance_city_km = st.slider(
        "Distance to City Center (km)",
        min_value=1.0,
        max_value=40.0,
        step=0.5,
        key="input_distance_city_km"
    )

    crime_rate_index = st.slider(
        "Crime Rate Index (0.5 - 12.0)",
        min_value=0.5,
        max_value=12.0,
        step=0.1,
        key="input_crime_rate_index"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="input-section-card">
        <div class="section-title">🚇 Infrastructure & Connectivity</div>
    """, unsafe_allow_html=True)

    conn_1, conn_2 = st.columns(2)
    with conn_1:
        near_metro = st.radio(
            "Within 1km of Metro?",
            options=["Yes", "No"],
            horizontal=True,
            key="input_near_metro"
        )
    with conn_2:
        near_school = st.radio(
            "Within 1km of School?",
            options=["Yes", "No"],
            horizontal=True,
            key="input_near_school"
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# PREDICTION COMPUTATION & RESULTS
# ==========================================
input_features = pd.DataFrame([{
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "location_score": location_score,
    "property_age": property_age,
    "distance_city_km": distance_city_km,
    "near_school": 1 if near_school == "Yes" else 0,
    "near_metro": 1 if near_metro == "Yes" else 0,
    "crime_rate_index": crime_rate_index,
    "sale_year": sale_year,
    "sale_month": sale_month
}])[FEATURE_COLUMNS]

predicted_price = float(model.predict(input_features)[0])
price_per_sqft = predicted_price / area_sqft if area_sqft > 0 else 0
estimated_emi = calculate_estimated_emi(predicted_price)

# Property Category
if predicted_price >= 25000000:
    tier = "Luxury Premium"
    tier_color = "#10B981"
elif predicted_price >= 12000000:
    tier = "Mid-to-High Tier"
    tier_color = "#3B82F6"
else:
    tier = "Affordable / Value"
    tier_color = "#F59E0B"

# Display Valuation Hero Banner
st.markdown(f"""
<div class="valuation-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="val-label">Estimated Property Market Value</span>
        <span style="background: {tier_color}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase;">{tier}</span>
    </div>
    <div class="val-amount">{format_inr_short(predicted_price)}</div>
    <div class="val-exact">Exact Estimated Valuation: {format_inr_exact(predicted_price)}</div>
</div>
""", unsafe_allow_html=True)

# Key Financial & Property Metrics
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="Price per Sq. Ft.",
        value=f"₹ {price_per_sqft:,.0f} / sqft"
    )

with m2:
    st.metric(
        label="Est. Monthly EMI (80% LTV)",
        value=format_inr_short(estimated_emi),
        help="Estimated monthly mortgage EMI at 8.5% interest over 20 years for 80% financing."
    )

with m3:
    st.metric(
        label="Location Rating",
        value=f"{location_score:.1f} / 10",
        delta=f"{distance_city_km:.1f} km from center"
    )

with m4:
    st.metric(
        label="Property Layout",
        value=f"{bedrooms} BHK",
        delta=f"{bathrooms} Baths • {area_sqft} sqft"
    )

# Property Specification Summary Badges
st.markdown("<div style='margin-top: 14px;'>", unsafe_allow_html=True)
st.markdown(f"""
    <span class="spec-pill">📐 {area_sqft:,} Built-up Sq. Ft.</span>
    <span class="spec-pill">🛏️ {bedrooms} Bedrooms</span>
    <span class="spec-pill">🚿 {bathrooms} Bathrooms</span>
    <span class="spec-pill">⏳ Age: {property_age} Years</span>
    <span class="spec-pill">📍 Location Score: {location_score}/10</span>
    <span class="spec-pill">🚇 Metro: {'Available' if near_metro == 'Yes' else 'None'}</span>
    <span class="spec-pill">🏫 School: {'Available' if near_school == 'Yes' else 'None'}</span>
    <span class="spec-pill">🛡️ Crime Index: {crime_rate_index}</span>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
