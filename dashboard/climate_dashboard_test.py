import os
print("Current Working Directory:", os.getcwd())
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Climate Tipping Point Dashboard', layout='wide')

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/cleaned/enhanced_energy_features.csv')
df = load_data()

# Sidebar filters
st.sidebar.header('Filters')
selected_year = st.sidebar.slider('Select Year', int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
selected_regions = st.sidebar.multiselect('Region(s)', sorted(df['region'].dropna().unique()), default=sorted(df['region'].dropna().unique()))
selected_metric = st.sidebar.radio('View by:', ['renewables_share_pct', 'co₂_per_capita_t'])

df_year = df[df['year'] == selected_year]
df_filtered = df_year[df_year['region'].isin(selected_regions)]

# 1. Heatmap: Regional Momentum (5-year renewables change)
st.subheader('Regional Momentum in Renewables Adoption (5-Year Change)')
momentum = df.groupby(['region', 'year'])['renewables_share_pct'].mean().reset_index()
momentum_pivot = momentum.pivot(index='region', columns='year', values='renewables_share_pct').diff(axis=1).fillna(0)
fig_heat = px.imshow(momentum_pivot, text_auto=True, aspect='auto', color_continuous_scale='Greens')
st.plotly_chart(fig_heat, use_container_width=True)

# 2. Quadrant Chart: CO₂ vs Energy Intensity
st.subheader(f'CO₂ per Capita vs Energy Intensity ({selected_year})')
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
    },
    title=None
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

# 3. Boxplot: CO₂ per Capita Distribution Over Time
st.subheader('CO₂ per Capita Distribution Over Time')
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

# 4. Sankey Diagram: Energy Mix Transition (Fossil → Renewables)
st.subheader(f'Energy Mix Transition by Region ({selected_year})')
mix = df_year.groupby('region')[['fossil_elec_twh', 'renew_elec_twh']].sum().reset_index()
labels = list(mix['region']) + ['Fossil', 'Renewables']
source = list(range(len(mix))) * 2
target = [len(mix)] * len(mix) + [len(mix)+1] * len(mix)
value = list(mix['fossil_elec_twh']) + list(mix['renew_elec_twh'])

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(label=labels),
    link=dict(source=source, target=target, value=value)
)])
st.plotly_chart(fig_sankey, use_container_width=True)

# 5. Top Movers: Countries with Highest 5-Year Renewable Gains
st.subheader('Top Countries by 5-Year Gain in Renewables Share')
df['renewables_5yr_change'] = df.groupby('country')['renewables_share_pct'].diff(periods=5)
top_gainers = df[df['year'] == selected_year].sort_values(by='renewables_5yr_change', ascending=False).head(10)
fig_bar = px.bar(
    top_gainers,
    x='country',
    y='renewables_5yr_change',
    labels={'renewables_5yr_change': '5-Year Gain (%)'},
    color='region'
)
st.plotly_chart(fig_bar, use_container_width=True)
