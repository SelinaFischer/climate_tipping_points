# streamlit dashboard was created with the help of chatgpt

# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------- OPTIONAL TREE SELECTOR IMPORT ----------
try:
    from streamlit_tree_select import tree_select
    TREE_OK = True
except Exception:
    TREE_OK = False

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title='Climate Tipping Point Dashboard', layout='wide')

# ---------- DATA LOAD ----------
@st.cache_data(ttl=600)
def load_data():
    df = pd.read_csv('data/cleaned/enhanced_energy_features_final.csv')
    df = df.sort_values(['country', 'year'])
    df['renewables_5yr_change'] = df.groupby('country')['renewables_share_pct'].diff(periods=5)
    df = df.dropna(subset=['country', 'region', 'co2_per_capita_t'])
    return df

df = load_data()

# ---------- HELPERS ----------
def choose_geo_col_for_color(levels_to_use):
    return "subregion" if "Subregion" in levels_to_use and "subregion" in df.columns else "region"

def select_all_clear_all(label, options, key_prefix):
    c1, c2, c3 = st.sidebar.columns([3, 1, 1])
    with c1:
        picked = st.sidebar.multiselect(label, options, key=f"{key_prefix}_multiselect")
    with c2:
        if st.sidebar.button("All", key=f"{key_prefix}_all"):
            picked = options
            st.session_state[f"{key_prefix}_multiselect"] = options
    with c3:
        if st.sidebar.button("None", key=f"{key_prefix}_none"):
            picked = []
            st.session_state[f"{key_prefix}_multiselect"] = []
    return picked

def build_tree(meta):
    tree = {}
    for _, r in meta.iterrows():
        region = r.get("region", "Unknown")
        subreg = r.get("subregion", "Unknown") if "subregion" in meta.columns else None
        country = r["country"]
        tree.setdefault(region, {})
        if subreg:
            tree[region].setdefault(subreg, []).append(country)
        else:
            tree[region].setdefault("_countries", []).append(country)
    return tree

def to_nodes(tree):
    nodes = []
    for reg, subs in tree.items():
        kids = []
        for sub, countries in subs.items():
            if sub == "_countries":
                kids.extend([{"label": c, "value": c} for c in countries])
            else:
                kids.append({"label": sub, "value": sub,
                             "children": [{"label": c, "value": c} for c in countries]})
        nodes.append({"label": reg, "value": reg, "children": kids})
    return nodes

def safe_tree_select(nodes):
    try:
        sel = tree_select(nodes, check_model="leaf", only_leaf_check=True, expand_all=True)
    except TypeError:
        try:
            sel = tree_select(nodes, check_model="leaf")
        except TypeError:
            sel = tree_select(nodes)
    if isinstance(sel, dict):
        checked = sel.get("checked", [])
        return [x.get("value", x) if isinstance(x, dict) else x for x in checked]
    return [x.get("value", x) if isinstance(x, dict) else x for x in sel]

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("🔍 Filters")
st.sidebar.subheader("🌍 Geography Filter")

geo_cols = [c for c in ["region", "subregion", "country"] if c in df.columns]
if "country" not in geo_cols:
    st.error("The dataset needs a 'country' column.")
    st.stop()

geo_meta = df[geo_cols].drop_duplicates()

use_tree = st.sidebar.checkbox(
    "Use tree selector (Region → Subregion → Country)",
    value=TREE_OK,
    disabled=not TREE_OK
)

countries_selected = None
levels_to_use = []

if use_tree and TREE_OK:
    tree_data = build_tree(geo_meta)
    nodes = to_nodes(tree_data)
    # render in sidebar
    with st.sidebar:
        picked = safe_tree_select(nodes)
    countries_selected = picked if picked else geo_meta["country"].unique()
    sel_rows = geo_meta[geo_meta["country"].isin(countries_selected)]
    if "subregion" in geo_cols and sel_rows["subregion"].nunique() > 1:
        levels_to_use.append("Subregion")
    if sel_rows["region"].nunique() > 1:
        levels_to_use.append("Region")
else:
    levels_to_use = st.sidebar.multiselect(
        "Filter by any combination of levels:",
        [c.capitalize() for c in geo_cols if c != "country"],
        default=[l.capitalize() for l in geo_cols if l in ["region", "subregion"]]
    )
    include_countries = st.sidebar.checkbox("Add / limit by specific countries", False)

    picked_sets = []
    if "Region" in levels_to_use:
        regions = sorted(geo_meta["region"].dropna().unique())
        picked_r = select_all_clear_all("Choose Region(s)", regions, "regions")
        if picked_r:
            picked_sets.append(set(geo_meta[geo_meta["region"].isin(picked_r)]["country"].unique()))
    if "Subregion" in levels_to_use and "subregion" in geo_cols:
        subs = sorted(geo_meta["subregion"].dropna().unique())
        picked_s = select_all_clear_all("Choose Subregion(s)", subs, "subregions")
        if picked_s:
            picked_sets.append(set(geo_meta[geo_meta["subregion"].isin(picked_s)]["country"].unique()))
    if include_countries:
        all_c = sorted(geo_meta["country"].dropna().unique())
        picked_c = select_all_clear_all("Choose Country/Countries", all_c, "countries")
        if picked_c:
            picked_sets.append(set(picked_c))

    countries_selected = sorted(set().union(*picked_sets)) if picked_sets else geo_meta["country"].unique()

color_col = choose_geo_col_for_color(levels_to_use)

# Metric & Year
st.sidebar.markdown("#### Metric Comparison (Visuals 3, 4 & 5)")
selected_metric = st.sidebar.radio(
    "Select a metric to display:",
    ["renewables_share_pct", "co2_per_capita_t"],
    format_func=lambda x: "Renewables Share (%)" if x == "renewables_share_pct" else "CO₂ per Capita (tonnes)",
    key="selected_metric"
)

st.sidebar.markdown("---")

selected_year = st.sidebar.slider(
    "Select Year",
    int(df['year'].min()),
    int(df['year'].max()),
    int(df['year'].max()),
    key="select_year_slider"
)

st.sidebar.markdown("---")

# ---------- APPLY FILTERS ----------
df_year = df[df['year'] == selected_year]
df_filtered = df_year[df_year['country'].isin(countries_selected)]
if df_filtered.empty:
    st.warning("No data for the chosen filters. Adjust your selections.")
    st.stop()

# ---------- PAGE TOP ----------
st.title("Climate Tipping Points: How Renewables & Efficiency Cut CO₂ for a Greener Future")

st.markdown(
    """
    <div style='font-size:17px; line-height:1.6; margin-bottom: 1rem;'>
        Explore global energy trends, CO₂ emissions, and renewable energy adoption.<br>
        Use the filters to see who’s leading the change and where tipping points are accelerating climate action.
    </div>
    """,
    unsafe_allow_html=True
)

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
    lvl_txt = ", ".join(levels_to_use) if levels_to_use else "All"
    st.markdown(
        f"""
        **Filter Logic:**

        - **Geography (Region/Subregion/Country)**  → Applies to Visuals **1–9**  
          *(If you leave everything unchecked, all countries are included by default.)*

         - **Year Slider**  
           - **Applies to Visuals** **2 - 9**  
           - **Does NOT change Visual 1** *(Visual 1 always shows the full 2000–2020 trend)*

        - **Metric Selector** (Sidebar radio button) → Applies to Visuals **3 -5**  

        **Current Selection:**  
          - Countries included: **{len(countries_selected)}**  
          - Geographic Levels chosen: **{lvl_txt}**  
          - Selected Year: **{selected_year}**
        """,
        unsafe_allow_html=True
    )

# ---------- 1. Global Progress Over Time ----------
st.markdown("### 1. Global Progress Over Time (2000–2020)")
yearly = df[df['country'].isin(countries_selected)].groupby('year').agg({
    'renewables_share_pct': 'mean',
    'co2_per_capita_t': 'mean',
    'elec_access_pct': 'mean'
}).reset_index()
yearly_long = yearly.melt(id_vars='year', var_name='Indicator', value_name='Value')

fig_trend = px.line(
    yearly_long, x='year', y='Value',
    color='Indicator', facet_col='Indicator', facet_col_wrap=1,
    color_discrete_map={'renewables_share_pct': '#1f77b4',
                        'co2_per_capita_t': '#ff7f0e',
                        'elec_access_pct': '#2ca02c'},
    labels={'year': 'Year', 'Value': ''},
    title='Global Trends (2000–2020)'
)
fig_trend.update_yaxes(matches=None, title_text='')
fig_trend.for_each_annotation(lambda a: a.update(text=a.text.split('=')[1]))
fig_trend.update_layout(showlegend=False, title={'x': 0.5})
st.plotly_chart(fig_trend, use_container_width=True)

# ---------- 2. Country Leaders ----------
st.markdown("---")
st.markdown("### 2. Country Leaders in Climate Action")

col1, col2 = st.columns(2)

with col1:
    top_renew = df_filtered.sort_values(by='renewables_share_pct', ascending=False).head(10)
    fig1 = px.bar(
        top_renew, x='renewables_share_pct', y='country',
        orientation='h', color=color_col,
        title=f'Top 10 Countries by Renewables Share ({selected_year})'
    )
    fig1.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    tmp = df[df['country'].isin(countries_selected)][['country', 'year', 'co2_per_capita_t']].dropna()
    co2_diff = tmp.pivot_table(index='country', columns='year', values='co2_per_capita_t')
    year_cols = sorted(col for col in co2_diff.columns if pd.api.types.is_number(col))
    if len(year_cols) >= 2:
        co2_diff = co2_diff.dropna(subset=[year_cols[0], year_cols[-1]])
        co2_diff['change'] = co2_diff[year_cols[-1]] - co2_diff[year_cols[0]]
        top_reducers = co2_diff.nsmallest(10, 'change').reset_index()
        top_reducers = top_reducers.merge(df[['country', color_col]].drop_duplicates(),
                                          on='country', how='left')
        fig2 = px.bar(
            top_reducers.dropna(subset=[color_col]),
            x='change', y='country',
            orientation='h', color=color_col,
            title=f"Top 10 CO₂ Reducers ({year_cols[0]}–{year_cols[-1]})"
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough year data to compute CO₂ change.")

# ---------- 3. Emissions vs Efficiency ----------
st.markdown("---")
st.markdown("### 3. Emissions vs Efficiency Explorer")

fig3 = px.scatter(
    df_filtered,
    x='energy_intensity_mj_usd',
    y='co2_per_capita_t',
    size='renewables_share_pct',
    color=color_col,
    hover_name='country',
    title='Energy Intensity vs CO₂ per Capita (bubble size = Renewables %)',
    labels={'energy_intensity_mj_usd': 'Energy Intensity (MJ/USD)',
            'co2_per_capita_t': 'CO₂ per Capita (t)',
            'renewables_share_pct': 'Renewables Share (%)'},
    size_max=60
)
st.plotly_chart(fig3, use_container_width=True)

# ---------- 4. Momentum Heatmap ----------
st.markdown("---")
st.markdown("### 4. Momentum in Selected Metric (5-Year Change)")

agg_col = color_col
momentum = df[df['country'].isin(countries_selected)].groupby([agg_col, 'year'])[selected_metric].mean().reset_index()
momentum_pivot = momentum.pivot(index=agg_col, columns='year', values=selected_metric).diff(axis=1).fillna(0)

fig_heat = px.imshow(
    momentum_pivot,
    text_auto=True,
    aspect='auto',
    color_continuous_scale='Greens',
    labels=dict(color="5-Year Δ")
)
fig_heat.update_layout(
    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
    title=f"5-Year Change in {selected_metric.replace('_',' ').title()} by {agg_col.title()}"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ---------- 5. Quadrant Chart ----------
st.markdown("---")
st.markdown("### 5. CO₂ per Capita vs Energy Intensity (Quadrant)")

fig_quad = px.scatter(
    df_filtered,
    x='energy_intensity_mj_usd',
    y='co2_per_capita_t',
    color=color_col,
    color_discrete_sequence=px.colors.qualitative.Set2,
    size='log_gdp_pc_usd',
    size_max=60,
    hover_name='country',
    labels={'energy_intensity_mj_usd': 'Energy Intensity (MJ/$)',
            'co2_per_capita_t': 'CO₂ per Capita (t)',
            'log_gdp_pc_usd': 'Log GDP per Capita'},
    title=f'CO₂ per Capita vs Energy Intensity by {color_col.title()}'
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

# ---------- 6. CO₂ Distribution Over Time ----------
st.markdown("---")
st.markdown("### 6. CO₂ Distribution Over Time")

df_box = df[df['country'].isin(countries_selected)].copy()
max_year = df['year'].max()
bins = [2000, 2010, 2020, max_year + 1]
labels = ['2001–2010', '2011–2020', f'2021–{max_year}']
df_box['period'] = pd.cut(df_box['year'], bins=bins, labels=labels)

fig_box = px.box(
    df_box,
    x='period',
    y='co2_per_capita_t',
    color=color_col,
    labels={'co2_per_capita_t': 'CO₂ per Capita (t)', 'period': 'Time Period'},
    title=f'CO₂ per Capita Distribution by {color_col.title()} Over Time'
)
st.plotly_chart(fig_box, use_container_width=True)

# ---------- 7. Energy Mix Transition ----------
st.markdown("---")
st.markdown("### 7. Energy Mix Transition by Region")

mix = df_year[df_year['country'].isin(countries_selected)] \
    .groupby('region')[['fossil_elec_twh', 'renew_elec_twh', 'nuclear_elec_twh']].sum().reset_index()

if not mix.empty:
    labels = list(mix['region']) + ['Fossil', 'Renewables']
    source = list(range(len(mix))) * 2
    target = [len(mix)] * len(mix) + [len(mix) + 1] * len(mix)
    value = list(mix['fossil_elec_twh']) + list(mix['renew_elec_twh'])

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=40, line=dict(color="black", width=0.8),
                  label=labels,
                  color=["#a1d99b"] * len(mix) + ["#e34a33", "#2b8cbe"]),
        link=dict(source=source, target=target, value=value,
                  color=["rgba(160,160,160,0.2)"] * len(mix) + ["rgba(34,139,34,0.3)"] * len(mix))
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.warning("No data available to plot the energy mix Sankey diagram.")

# ---------- 8. Top 5-Year Renewable Gainers ----------
st.markdown("---")
st.markdown("### 8. Top Countries by 5-Year Gain in Renewables")

top_gainers = df[df['year'] == selected_year]
top_gainers = top_gainers[top_gainers['country'].isin(countries_selected)] \
    .sort_values(by='renewables_5yr_change', ascending=False).head(10)

fig_bar = px.bar(
    top_gainers,
    x='country',
    y='renewables_5yr_change',
    labels={'renewables_5yr_change': '5-Year Gain (%)'},
    color=color_col,
    title=f'Top 10 Countries by Renewable Energy Growth ({selected_year})'
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------- 9. Global Map ----------
st.markdown("---")
st.markdown("### 🌍 9. Global View: Energy & Emissions Landscape")
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
    label_visibility="collapsed"
)
selected_map_metric = map_metric_options[selected_map_label]

map_data = df[df['year'] == selected_year]
map_data = map_data[map_data['country'].isin(countries_selected)]
map_data = map_data[map_data[selected_map_metric].notna()]

fig_map = px.choropleth(
    map_data,
    locations="country",
    locationmode="country names",
    color=selected_map_metric,
    hover_name="country",
    hover_data={selected_map_metric: True, color_col: True,
                "renewables_share_pct": True, "co2_per_capita_t": True},
    color_continuous_scale="Viridis",
    title=f"{selected_map_label} by Country – {selected_year}"
)
fig_map.update_layout(
    margin=dict(l=0, r=0, t=50, b=0),
    coloraxis_colorbar=dict(title=selected_map_label, ticks="outside", len=0.75)
)
st.plotly_chart(fig_map, use_container_width=True)
