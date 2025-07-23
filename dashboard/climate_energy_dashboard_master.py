# streamlit dashboard was created with the help of chatgpt

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
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

# ---------- SIDEBAR FILTERS (CLEANED) ----------
st.sidebar.header("🔍 Filters")

geo_cols = [c for c in ["region", "subregion", "country"] if c in df.columns]
if "country" not in geo_cols:
    st.error("The dataset needs a 'country' column.")
    st.stop()
geo_meta = df[geo_cols].drop_duplicates()

# ---- Geography (tree only) ----
with st.sidebar.expander("🌍 Geography", expanded=True):
    if not TREE_OK:
        st.error("`streamlit-tree-select` is missing. Add it to requirements.txt.")
        st.stop()

    tree_data = build_tree(geo_meta)
    nodes = to_nodes(tree_data)
    picked = safe_tree_select(nodes)
    countries_selected = picked or geo_meta["country"].tolist()

    # figure out which level to color/legend by
    sel_rows = geo_meta[geo_meta["country"].isin(countries_selected)]
    levels_to_use = []
    if "subregion" in geo_cols and sel_rows["subregion"].nunique() > 1:
        levels_to_use.append("Subregion")
    if sel_rows["region"].nunique() > 1:
        levels_to_use.append("Region")

color_col = choose_geo_col_for_color(levels_to_use)

# ---- Year ----
with st.sidebar.expander("📅 Year", expanded=True):
    selected_year = st.slider(
        "Select Year",
        int(df['year'].min()),
        int(df['year'].max()),
        int(df['year'].max()),
        key="select_year_slider"
    )

# ---- Metric ----
with st.sidebar.expander("📈 Metric (for Visuals 3, 4 & 5)", expanded=False):
    selected_metric = st.radio(
        "Select a metric to display:",
        ["renewables_share_pct", "co2_per_capita_t"],
        format_func=lambda x: "Renewables Share (%)" if x == "renewables_share_pct" else "CO₂ per Capita (tonnes)",
        key="selected_metric"
    )

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

        - **Geography (Region/Subregion/Country)** → Applies to Visuals **1–9**  
          *(If you leave everything unchecked, all countries are included by default.)*

        - **Year Slider**  
          - **Applies to Visuals** **2–9** and the **scatter/metric in Visual 10** 
          - **Visual 1 always shows the full 2000–2020 trend**

        - **Metric Selector** (Sidebar radio button) → Applies to Visuals **3–5**  

        **Visual 10 – Filter rules**
        - Heatmap (model surface) → **does NOT change** with sidebar filters  
        -  Red dots & ✖ comparison country → follow **Geography + Year**  
        - ⭐ Your scenario → set only by the two sliders inside Visual 10  
        -  30% dashed line =  tipping point threshold  
        -  Sidebar Metric radio button is **not used** here

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


# ---------- 10. CO₂ per Capita – Predictive What‑If Explorer ----------
st.markdown("---")
st.markdown("### 10. CO₂ per Capita – Predictive What‑If Explorer")

from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

# 1) Train simple model on ALL data (use df_filtered if you want the surface to react to filters)
train_cols = ['renewables_share_pct', 'energy_intensity_mj_usd', 'co2_per_capita_t']
df_model = df[train_cols].dropna().copy()
df_model['tip30']    = (df_model['renewables_share_pct'] >= 30).astype(int)
df_model['re_x_ei']  = df_model['renewables_share_pct'] * df_model['energy_intensity_mj_usd']
df_model['re_x_tip'] = df_model['renewables_share_pct'] * df_model['tip30']

y = np.log(df_model['co2_per_capita_t'])
X = df_model[['renewables_share_pct','energy_intensity_mj_usd','tip30','re_x_ei','re_x_tip']]
model = LinearRegression().fit(X, y)

# 2) Prediction grid for heatmap
r_min, r_max   = 0.0, float(df['renewables_share_pct'].max())
ei_min, ei_max = float(df['energy_intensity_mj_usd'].min()), float(df['energy_intensity_mj_usd'].max())
r_grid  = np.linspace(r_min, r_max, 80)
ei_grid = np.linspace(ei_min, ei_max, 80)

R, EI = np.meshgrid(r_grid, ei_grid)
TIP = (R >= 30).astype(int)

grid_df = pd.DataFrame({
    'renewables_share_pct': R.ravel(),
    'energy_intensity_mj_usd': EI.ravel(),
    'tip30': TIP.ravel()
})
grid_df['re_x_ei']  = grid_df['renewables_share_pct'] * grid_df['energy_intensity_mj_usd']
grid_df['re_x_tip'] = grid_df['renewables_share_pct'] * grid_df['tip30']

pred_log = model.predict(grid_df[['renewables_share_pct','energy_intensity_mj_usd','tip30','re_x_ei','re_x_tip']])
pred     = np.exp(pred_log)
Z        = pred.reshape(EI.shape)

# 3) Base traces
heat = go.Heatmap(
    x=r_grid, y=ei_grid, z=Z,
    colorscale="Viridis",
    colorbar=dict(title="Predicted CO₂/t")
)
line30 = go.Scatter(
    x=[30, 30], y=[ei_min, ei_max],
    mode="lines",
    line=dict(color="white", dash="dash"),
    name="30% Tipping Point Threshold"
)
scatter_pts = go.Scatter(
    x=df_filtered['renewables_share_pct'],
    y=df_filtered['energy_intensity_mj_usd'],
    mode="markers",
    marker=dict(size=6, opacity=0.7, color="red"),
    text=df_filtered['country'],
    name="Current selected countries"
)
fig_pred = go.Figure(data=[heat, line30, scatter_pts])

# 4) Controls (your scenario + comparison country)
with st.expander("Try your own values", expanded=True):
    country_for_diff = st.selectbox("Compare to country (current year)", df_filtered['country'].unique())

    base_row = df_filtered[df_filtered['country'] == country_for_diff]
    if base_row.empty:
        re_default = float(df_filtered['renewables_share_pct'].median())
        ei_default = float(df_filtered['energy_intensity_mj_usd'].median())
    else:
        re_default = float(base_row['renewables_share_pct'].iloc[0])
        ei_default = float(base_row['energy_intensity_mj_usd'].iloc[0])

    c1, c2 = st.columns(2)
    with c1:
        re_user = st.slider("Renewables Share (%)", float(r_min), float(r_max), re_default, step=0.5)
    with c2:
        ei_user = st.slider("Energy Intensity (MJ/$)", float(ei_min), float(ei_max), ei_default, step=0.1)

    # Predict scenario
    tip_user = 1 if re_user >= 30 else 0
    X_user = pd.DataFrame({
        'renewables_share_pct':[re_user],
        'energy_intensity_mj_usd':[ei_user],
        'tip30':[tip_user],
        're_x_ei':[re_user*ei_user],
        're_x_tip':[re_user*tip_user]
    })
    co2_pred = float(np.exp(model.predict(X_user))[0])

    base_val  = float(base_row['co2_per_capita_t'].mean()) if not base_row.empty else np.nan
    delta     = co2_pred - base_val if not np.isnan(base_val) else np.nan
    delta_str = f"Δ {delta:+.2f} t vs {country_for_diff}" if not np.isnan(delta) else "N/A"

    st.metric("Predicted CO₂ per Capita (t)", f"{co2_pred:,.2f}", delta_str)

# 5) Scenario & comparison markers
scenario_trace = go.Scatter(
    x=[re_user], y=[ei_user],
    mode="markers",
    marker=dict(symbol="star", size=16, color="white", line=dict(color="black", width=1)),
    name="Your scenario"
)
fig_pred.add_trace(scenario_trace)

if not base_row.empty:
    fig_pred.add_trace(go.Scatter(
        x=[base_row['renewables_share_pct'].iloc[0]],
        y=[base_row['energy_intensity_mj_usd'].iloc[0]],
        mode="markers",
        marker=dict(symbol="x", size=12, color="yellow", line=dict(width=1, color="black")),
        name=country_for_diff
    ))

# 6) Hypotheses callouts
# Light shade to visualize tipping-zone (H2)
fig_pred.add_shape(
    type="rect", x0=30, x1=r_max, y0=ei_min, y1=ei_max,
    line_width=0, fillcolor="rgba(255,255,255,0.04)", layer="below"
)
# H2 label (no arrow)
fig_pred.add_annotation(
    x=30, y=ei_max,
    text="H2: ≥30% renewables → faster CO₂ decline",
    showarrow=False,
    xanchor="left", yanchor="top",
    font=dict(size=11, color="white"),
    bgcolor="rgba(0,0,0,0.35)", borderpad=3
)
# H1 arrow (keep)
fig_pred.add_annotation(
    x=r_max*0.85, y=ei_min + (ei_max-ei_min)*0.15,
    ax=r_max*0.55, ay=ei_min + (ei_max-ei_min)*0.15,
    text="H1: ↑ Renewables  →  ↓ CO₂",
    showarrow=True, arrowhead=2, arrowsize=1.2, arrowcolor="white",
    font=dict(size=11, color="white"),
    bgcolor="rgba(0,0,0,0.35)"
)
# H3 label ONLY (no arrow)
fig_pred.add_annotation(
    x=r_min + (r_max-r_min)*0.08,
    y=ei_max * 0.80,
    text="H3: ↓ Energy intensity  →  ↓ CO₂",
    showarrow=False,
    xanchor="left", yanchor="middle",
    font=dict(size=11, color="white"),
    bgcolor="rgba(0,0,0,0.35)",
    borderpad=3
)

# 7) Legend / colorbar / spacing tweaks
fig_pred.data[0].showlegend = False                  # hide heatmap from legend
fig_pred.data[0].colorbar.update(x=1.14, len=0.75)   # move colorbar right

fig_pred.update_layout(
    legend=dict(
        orientation="h",
        x=0.5, y=-0.28,                 # push legend further down
        xanchor="center", yanchor="top",
        bgcolor="rgba(255,255,255,0.6)",
        borderwidth=0,
        font=dict(size=11)
    ),
    margin=dict(l=0, r=140, t=70, b=190),            # extra bottom for spacing
    xaxis_title="Renewables Share (%)",
    yaxis_title="Energy Intensity (MJ/$)",
    title={"text": "Predicted CO₂ per Capita Surface (model-based)", "x": 0.02}
)

st.plotly_chart(fig_pred, use_container_width=True)

# Optional extra whitespace under the chart (visual separation)
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)


# Optional: caption instead of legend
# st.caption("⭐ Your scenario   ✖ Comparison country   • Red dots = actual countries")
