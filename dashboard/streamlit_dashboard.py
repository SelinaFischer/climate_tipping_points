import os
print("Current Working Directory:", os.getcwd())
import streamlit as st
import pandas as pd
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Climate Tipping Points Dashboard", layout="wide")

# --- LOAD DATA ---
df = pd.read_csv("data/cleaned/enhanced_energy_features.csv") 

# --- SIDEBAR ---
st.sidebar.title("Filters")
years = sorted(df['year'].unique())
selected_year = st.sidebar.slider("Select Year", min_value=min(years), max_value=max(years), value=max(years))
regions = df['region'].dropna().unique().tolist()
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)

# --- PAGE TITLE ---
st.title("Climate Tipping Points: How Renewables & Efficiency Cut CO₂ for a Greener Future")
st.markdown("""
Explore global energy trends, CO₂ emissions, and renewable energy adoption.
Use the filters to examine who’s leading the change and where tipping points are accelerating climate action.
""")

# --- FILTER DATA ---
df_filtered = df[(df['year'] == selected_year) & (df['region'].isin(selected_regions))]
df_global = df[df['region'].isin(selected_regions)]

# --- SECTION 1: Global Trends Over Time ---
st.subheader("1. Global Progress Over Time")
global_trend = df_global.groupby('year').agg({
    'renewables_share_pct': 'mean',
    'co2_per_capita_t': 'mean',
    'energy_intensity_mj_usd': 'mean'
}).reset_index()

fig_trend = px.line(
    global_trend,
    x='year',
    y=['renewables_share_pct', 'co2_per_capita_t', 'energy_intensity_mj_usd'],
    labels={'value': 'Metric', 'variable': 'Indicator', 'year': 'Year'},
    title='Global Trends: Renewables, CO₂ per Capita & Energy Intensity'
)
st.plotly_chart(fig_trend, use_container_width=True)

# --- SECTION 2: Country Leaders ---
st.subheader("2. Country Leaders in Climate Action")
col1, col2 = st.columns(2)

with col1:
    top_renew = df_filtered.sort_values(by='renewables_share_pct', ascending=False).head(10)
    fig1 = px.bar(
        top_renew,
        x='renewables_share_pct',
        y='country',
        orientation='h',
        color='region',
        title=f'Top 10 Countries by Renewables Share ({selected_year})'
    )
    fig1.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    co2_diff = df.pivot_table(index='country', columns='year', values='co2_per_capita_t')
    co2_diff['change'] = co2_diff[max(years)] - co2_diff[min(years)]
    top_reducers = co2_diff.nsmallest(10, 'change').reset_index()
    top_reducers = top_reducers.merge(df[['country', 'region']].drop_duplicates(), on='country', how='left')
    fig2 = px.bar(
        top_reducers.dropna(subset=['region']),
        x='change',
        y='country',
        orientation='h',
        color='region',
        title=f'Top 10 CO₂ Reducers ({min(years)}–{max(years)})'
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

# --- SECTION 3: Emissions vs Efficiency Explorer ---
st.subheader("3. Emissions vs Efficiency Explorer")
fig3 = px.scatter(
    df_filtered,
    x='energy_intensity_mj_usd',
    y='co2_per_capita_t',
    size='renewables_share_pct',
    color='region',
    hover_name='country',
    title='Energy Intensity vs CO₂ per Capita with Renewables Share',
    labels={
        'energy_intensity_mj_usd': 'Energy Intensity (MJ/USD)',
        'co2_per_capita_t': 'CO₂ per Capita (t)',
        'renewables_share_pct': 'Renewables Share (%)'
    },
    size_max=60
)
st.plotly_chart(fig3, use_container_width=True)

# --- SECTION 4: Global Map ---
st.subheader("4. Global Map: Clean Energy or Emissions")
map_metric = st.selectbox("Select Metric to Display on Map", ["renewables_share_pct", "co2_per_capita_t"])
fig4 = px.choropleth(
    df[df['year'] == selected_year],
    locations="country",
    locationmode="country names",
    color=map_metric,
    hover_name="country",
    color_continuous_scale="Viridis",
    title=f"Global Map ({selected_year}) - {map_metric.replace('_', ' ').title()}"
)
st.plotly_chart(fig4, use_container_width=True)

# --- FOOTER ---
st.markdown("""
---
**Data Source:** Aggregated climate & energy indicators (2000–2020)  
                 World Bank Population subset (2000 - 2020)
                 UNSD Region Mapping
**Author:** Selinafish
**Tools:** Python, Pandas, Plotly, Streamlit
""")

