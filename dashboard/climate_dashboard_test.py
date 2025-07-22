import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit page config (safe for production)
st.set_page_config(page_title='Climate Tipping Point Dashboard', layout='wide')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned/enhanced_energy_features_final.csv')

df = load_data()

# Precompute renewables 5-year change ONCE
df['renewables_5yr_change'] = df.groupby('country')['renewables_share_pct'].diff(periods=5)

# Sidebar filters
st.sidebar.header('Filters')
selected_year = st.sidebar.slider('Select Year', int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
selected_regions = st.sidebar.multiselect('Region(s)', sorted(df['region'].dropna().unique()), default=sorted(df['region'].dropna().unique()))

# Define labels for better display
metric_labels = {
    'renewables_share_pct': 'Renewables Share (%)',
    'co2_per_capita_t': 'CO₂ per Capita (tonnes)'
}
selected_metric = st.sidebar.radio('View by:', list(metric_labels.keys()), format_func=lambda x: metric_labels[x])

# Filter dataframe based on selection
df_year = df[df['year'] == selected_year]
df_filtered = df_year[df_year['region'].isin(selected_regions)]

# Early safety checks
if selected_metric not in df.columns:
    st.error(f"Selected metric '{selected_metric}' not found in data.")
    st.stop()

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust year or regions.")
    st.stop()

# 1. Heatmap: Regional Momentum (5-year change by selected metric)
with st.container():
    st.markdown(f"### 1. Regional Momentum in {metric_labels[selected_metric]} (5-Year Change)")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    momentum = df.groupby(['region', 'year'])[selected_metric].mean().reset_index()
    momentum_pivot = momentum.pivot(index='region', columns='year', values=selected_metric).diff(axis=1).fillna(0)
    fig_heat = px.imshow(momentum_pivot, text_auto=True, aspect='auto', color_continuous_scale='Greens')
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)

# 2. Quadrant Chart
with st.container():
    st.markdown(f"### 2. CO₂ per Capita vs Energy Intensity ({selected_year})")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    fig_quad = px.scatter(
        df_filtered,
        x='energy_intensity_mj_usd',
        y='co2_per_capita_t',
        color='region',
        size='log_gdp_pc_usd',
        size_max=60,
        hover_name='country',
        labels={
            'energy_intensity_mj_usd': 'Energy Intensity (MJ/$)',
            'co2_per_capita_t': 'CO₂ per Capita (t)',
            'log_gdp_pc_usd': 'Log GDP per Capita'
        }
    )
    fig_quad.add_shape(
        type='line',
        x0=df_filtered['energy_intensity_mj_usd'].mean(),
        x1=df_filtered['energy_intensity_mj_usd'].mean(),
        y0=df_filtered['co2_per_capita_t'].min(),
        y1=df_filtered['co2_per_capita_t'].max(),
        line=dict(dash='dash', color='gray')
    )
    fig_quad.add_shape(
        type='line',
        y0=df_filtered['co2_per_capita_t'].mean(),
        y1=df_filtered['co2_per_capita_t'].mean(),
        x0=df_filtered['energy_intensity_mj_usd'].min(),
        x1=df_filtered['energy_intensity_mj_usd'].max(),
        line=dict(dash='dash', color='gray')
    )
    st.plotly_chart(fig_quad, use_container_width=True)
    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)

# 3. Boxplot
with st.container():
    st.markdown("### 3. CO₂ per Capita Distribution Over Time")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    df_box = df[df['region'].isin(selected_regions)].copy()
    df_box['period'] = pd.cut(df_box['year'], bins=[2000, 2010, 2020, 2025], labels=['2001–2010', '2011–2020', '2021–2025'])
    fig_box = px.box(
        df_box,
        x='period',
        y='co2_per_capita_t',
        color='region',
        labels={'co2_per_capita_t': 'CO₂ per Capita (t)', 'period': 'Time Period'}
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)

# 4. Sankey Diagram
with st.container():
    st.markdown(f"### 4. Energy Mix Transition by Region ({selected_year})")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    mix = df_year[df_year['region'].isin(selected_regions)].groupby('region')[['fossil_elec_twh', 'renew_elec_twh']].sum().reset_index()
    labels = list(mix['region']) + ['Fossil', 'Renewables']
    source = list(range(len(mix))) * 2
    target = [len(mix)] * len(mix) + [len(mix)+1] * len(mix)
    value = list(mix['fossil_elec_twh']) + list(mix['renew_elec_twh'])
    region_node_color = ["#a1d99b"] * len(mix)
    special_node_color = ["#e34a33", "#2b8cbe"]

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=40,
            line=dict(color="black", width=0.8),
            label=labels,
            color=region_node_color + special_node_color
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=["rgba(160,160,160,0.2)"] * len(mix) + ["rgba(34,139,34,0.3)"] * len(mix)
        )
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)

# 5. Bar Chart: Top Gainers
with st.container():
    st.markdown("### 5. Top Countries by 5-Year Gain in Renewables Share")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    top_gainers = df[df['year'] == selected_year].sort_values(by='renewables_5yr_change', ascending=False).head(10)
    fig_bar = px.bar(
        top_gainers,
        x='country',
        y='renewables_5yr_change',
        labels={'renewables_5yr_change': '5-Year Gain (%)'},
        color='region'
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)


# 6. World Map: Global View – Energy & Emissions Landscape
with st.container():
    st.markdown("### 6. Global View: Energy & Emissions Landscape")
    st.markdown("<hr style='margin-top:0; margin-bottom:1rem; border:1px solid #ddd;'>", unsafe_allow_html=True)

    # Dropdown options for map metrics
    map_metric_options = {
        "Renewables Share (%)": "renewables_share_pct",
        "CO₂ per Capita (t)": "co2_per_capita_t",
        "Energy Intensity (MJ/$)": "energy_intensity_mj_usd",
        "Log GDP per Capita": "log_gdp_pc_usd",
        "Low‑Carbon Electricity (%)": "low_carbon_elec_pct",
        "Electricity Access (%)": "elec_access_pct",
        "Clean Fuel Access (%)": "clean_fuels_access_pct",
        "Renewable Capacity (kW/person)": "renew_cap_kw_pc",
        "Climate Finance (USD)": "climate_finance_usd"
    }

    selected_map_label = st.selectbox("Select Metric to Display on Map", list(map_metric_options.keys()))
    selected_map_metric = map_metric_options[selected_map_label]

    map_data = df[df['year'] == selected_year].copy()
    map_data = map_data[map_data[selected_map_metric].notna()]

    fig_map = px.choropleth(
        map_data,
        locations="country",
        locationmode="country names",
        color=selected_map_metric,
        hover_name="country",
        hover_data={
            selected_map_metric: True,
            "region": True,
            "renewables_share_pct": True,
            "co2_per_capita_t": True
        },
        color_continuous_scale="Viridis",
        title=f"{selected_map_label} by Country – {selected_year}"
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=selected_map_label,
            ticks="outside",
            len=0.75
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("<hr style='margin-top:2rem; margin-bottom:2rem; border:1px solid #eee;'>", unsafe_allow_html=True)
