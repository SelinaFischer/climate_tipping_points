# streamlit dashboard was created with the help of chatgpt

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Climate Tipping Point Dashboard', layout='wide')

@st.cache_data(ttl=600)
def load_data():
    df = pd.read_csv('data/cleaned/enhanced_energy_features_final.csv')
    df = df.sort_values(['country', 'year'])
    df['renewables_5yr_change'] = df.groupby('country')['renewables_share_pct'].diff(periods=5)
    df = df.dropna(subset=['region', 'country', 'co2_per_capita_t'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

# 1. Metric Selector (for Visuals 3 & 5)
st.sidebar.markdown("#### Metric Comparision (Visuals 3 & 5)")

selected_metric = st.sidebar.radio(
    label="Select a metric to display:",
    options=["renewables_share_pct", "co2_per_capita_t"],
    format_func=lambda x: "Renewables Share (%)" if x == "renewables_share_pct" else "CO₂ per Capita (tonnes)",
    key="selected_metric",
    help="Updates the metric shown in Visuals 3 and 5."
)

st.sidebar.markdown("---")

# 2. Select Year (for most visuals)
selected_year = st.sidebar.slider(
    label="Select Year",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=int(df['year'].max()),
    key="select_year_slider"  # Unique key to prevent duplication
)

# 3. Select Region(s)
selected_regions = st.sidebar.multiselect(
    label="Select Region(s)",
    options=sorted(df['region'].dropna().unique()),
    default=sorted(df['region'].dropna().unique()),
    key="select_regions_multiselect"  # Unique key to prevent duplication
)


st.sidebar.markdown("---")

# --- Filtered Data ---
df_year = df[df['year'] == selected_year]
df_filtered = df_year[df_year['region'].isin(selected_regions)]

if selected_metric not in df.columns:
    st.error(f"Selected metric '{selected_metric}' not found in data.")
    st.stop()

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust year or regions.")
    st.stop()

# --- PAGE TITLE ---
st.title("Climate Tipping Points: How Renewables & Efficiency Cut CO₂ for a Greener Future")

# --- Enhanced Intro Text with Larger Font ---
st.markdown(
    """
    <div style='font-size:17px; line-height:1.6; margin-bottom: 1rem;'>
        Explore global energy trends, CO₂ emissions, and renewable energy adoption.  
        Use the filters to examine who’s leading the change and where tipping points are accelerating climate action.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Styled Expander Label with Tooltip and Bigger Font ---
st.markdown(
    """
    <div title="See which filters affect which sections of the dashboard." 
         style='font-size:17px; font-weight:bold; margin-bottom: -8px;'>

        ℹ️ How Filters Work
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("ℹ Filter Guide 👈"):
    st.markdown(
        """
        - **Year** and **Region(s)** filters apply to Visuals **1-4, 6, and 7**
        - **Metric Selection** updates Visuals **3 and 5**  
        - **Section 8** always shows the most recent year (not affected by the slider) 
        - **Section 9** provides a global overview based on all available years
        """,
        unsafe_allow_html=True
    )


# 1. Global Progress Over Time

st.markdown("### 1. Global Progress Over Time (2000–2020)")

yearly = df[df['region'].isin(selected_regions)].groupby('year').agg({
        'renewables_share_pct': 'mean',
        'co2_per_capita_t': 'mean',
        'elec_access_pct': 'mean'
    }).reset_index()

yearly_long = yearly.melt(id_vars='year', var_name='Indicator', value_name='Value')

fig_trend = px.line(
        yearly_long,
        x='year', y='Value',
        color='Indicator',
        facet_col='Indicator',
        facet_col_wrap=1,
        color_discrete_map={
            'renewables_share_pct': '#1f77b4',
            'co2_per_capita_t': '#ff7f0e',
            'elec_access_pct': '#2ca02c',
        },
        labels={'year': 'Year', 'Value': ''},
        title='Global Trends (2000–2020)'
    )

fig_trend.update_yaxes(matches=None, title_text='')  # Remove Y-axis label
fig_trend.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
fig_trend.update_layout(showlegend=False, title={'x': 0.5})

st.plotly_chart(fig_trend, use_container_width=True)


# 2. Country Leaders in Climate Action
st.markdown("---")
st.markdown("### 2. Country Leaders in Climate Action")
  
                                                                                                          
col1, col2 = st.columns(2)
with col1:
        top_renew = df_filtered.sort_values(by='renewables_share_pct', ascending=False).head(10)
        fig1 = px.bar(top_renew, x='renewables_share_pct', y='country', orientation='h', color='region',
                      title=f'Top 10 Countries by Renewables Share ({selected_year})')
        fig1.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig1, use_container_width=True)
with col2:
        co2_diff = df.pivot_table(index='country', columns='year', values='co2_per_capita_t')
        co2_diff = co2_diff.dropna(subset=[min(df["year"]), max(df["year"])])
        co2_diff['change'] = co2_diff[max(df["year"])] - co2_diff[min(df["year"])]
        top_reducers = co2_diff.nsmallest(10, 'change').reset_index()
        top_reducers = top_reducers.merge(df[['country', 'region']].drop_duplicates(), on='country', how='left')
        fig2 = px.bar(top_reducers.dropna(subset=['region']), x='change', y='country', orientation='h', color='region',
                      title=f'Top 10 CO₂ Reducers ({min(df["year"])}–{max(df["year"])})')
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

# 3. Emissions vs Efficiency Explorer
st.markdown("---")
st.markdown("### 3. Emissions vs Efficiency Explorer")

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


# 4. Regional Momentum in Renewables Share (5-Year Change)
st.markdown("---")
st.markdown("### 4. Regional Momentum in Renewables Share (5-Year Change)")

momentum = df.groupby(['region', 'year'])[selected_metric].mean().reset_index()

momentum_pivot = momentum.pivot(index='region', columns='year', values=selected_metric).diff(axis=1).fillna(0)

fig_heat = px.imshow(
    momentum_pivot,
    text_auto=True,
    aspect='auto',
    color_continuous_scale='Greens',
    labels=dict(color="5-Year Δ in Metric")
)

# 🔍 Improve axis label visibility
fig_heat.update_layout(
    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12))
)

st.plotly_chart(fig_heat, use_container_width=True)


# 5. CO₂ per Capita vs Energy Intensity (Quadrant)
st.markdown("---")
st.markdown("### 5. CO₂ per Capita vs Energy Intensity (Quadrant)")

fig_quad = px.scatter(
    df_filtered,
    x='energy_intensity_mj_usd',
    y='co2_per_capita_t',
    color='region',
    color_discrete_sequence=px.colors.qualitative.Set2,  # distinct new color theme
    size='log_gdp_pc_usd',
    size_max=60,
    hover_name='country',
    labels={
        'energy_intensity_mj_usd': 'Energy Intensity (MJ/$)',
        'co2_per_capita_t': 'CO₂ per Capita (t)',
        'log_gdp_pc_usd': 'Log GDP per Capita'
    },
    title='CO₂ per Capita vs Energy Intensity by Region'
)

# Add quadrant lines
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



# 6. CO₂ Distribution Over Time
st.markdown("---")
st.markdown("### 6. CO₂ Distribution Over Time")

df_box = df[df['region'].isin(selected_regions)].copy()

# Fix to ensure bin edges are unique
max_year = df['year'].max()
bins = [2000, 2010, 2020, max_year + 1]
labels = ['2001–2010', '2011–2020', f'2021–{max_year}']

df_box['period'] = pd.cut(df_box['year'], bins=bins, labels=labels)

fig_box = px.box(
    df_box,
    x='period',
    y='co2_per_capita_t',
    color='region',
    labels={
        'co2_per_capita_t': 'CO₂ per Capita (t)',
        'period': 'Time Period'
    },
    title='CO₂ per Capita Distribution by Region Over Time'
)

st.plotly_chart(fig_box, use_container_width=True)


# 7. Energy Mix Transition by Region
st.markdown("---")
st.markdown("### 7. Energy Mix Transition by Region")

mix = df_year[df_year['region'].isin(selected_regions)] \
    .groupby('region')[['fossil_elec_twh', 'renew_elec_twh', 'nuclear_elec_twh']].sum().reset_index()

if not mix.empty:
    labels = list(mix['region']) + ['Fossil', 'Renewables']
    source = list(range(len(mix))) * 2
    target = [len(mix)] * len(mix) + [len(mix)+1] * len(mix)
    value = list(mix['fossil_elec_twh']) + list(mix['renew_elec_twh'])

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=40, line=dict(color="black", width=0.8),
            label=labels,
            color=["#a1d99b"]*len(mix) + ["#e34a33", "#2b8cbe"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=["rgba(160,160,160,0.2)"]*len(mix) + ["rgba(34,139,34,0.3)"]*len(mix)
        )
    )])

    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.warning("No data available to plot the energy mix Sankey diagram.")


# 8. Top Countries by 5-Year Gain in Renewables
st.markdown("---")
st.markdown("### 8. Top Countries by 5-Year Gain in Renewables")

top_gainers = df[df['year'] == selected_year] \
    .sort_values(by='renewables_5yr_change', ascending=False).head(10)

fig_bar = px.bar(
    top_gainers,
    x='country',
    y='renewables_5yr_change',
    labels={'renewables_5yr_change': '5-Year Gain (%)'},
    color='region',
    title=f'Top 10 Countries by Renewable Energy Growth ({selected_year})'
)

st.plotly_chart(fig_bar, use_container_width=True)


# 9. Global View: Energy & Emissions Landscape
st.markdown("---")
st.markdown("### 🌍 9. Global View: Energy & Emissions Landscape")

# Enhanced heading above dropdown
st.markdown("##### **Select a Metric from the Dropdown to Display on the Map**")

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

selected_map_label = st.selectbox(
    label="Select a Metric",
    options=list(map_metric_options.keys()),
    label_visibility="collapsed"  # Hides label but avoids warning
)
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