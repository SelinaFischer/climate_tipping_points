# streamlit dashboard was created with the help of chatgpt

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import itertools

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

    # Normalize strings
    for col in ["country", "region", "subregion"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                                .str.replace("\u00a0", " ", regex=False)
                                .str.replace(r"\s+", " ", regex=True)
                                .str.strip())

    # Unify UK variants
    df["country"] = df["country"].replace({
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "Great Britain": "United Kingdom"
    })
    mask_uk = df["country"].str.contains(r"united\s*kingdom", case=False, na=False)
    df.loc[mask_uk, "country"] = "United Kingdom"

    # 5‑yr change
    if 'renewables_share_pct' in df.columns:
        df['renewables_5yr_change'] = df.groupby('country')['renewables_share_pct'].diff(5)

    # Keep only rows with country + co2
    df = df.dropna(subset=['country', 'co2_per_capita_t'])

    # Ensure UK geo fields
    uk_mask = df['country'].eq('United Kingdom')
    if 'region' in df.columns:
        df.loc[uk_mask & df['region'].isna(), 'region'] = 'Europe'
    if 'subregion' in df.columns:
        df.loc[uk_mask & df['subregion'].isna(), 'subregion'] = 'Western Europe'

    # Fill key cols for Visual 10
    for col in ['renewables_share_pct', 'energy_intensity_mj_usd']:
        if col in df.columns:
            df[col] = df.groupby('country')[col].transform(lambda s: s.ffill().bfill())

    return df

df = load_data()

# ---------- CONSTANTS ----------
FORCE_GEO = {
    'United Kingdom': {'region': 'Europe', 'subregion': 'Western Europe'},
    'Netherlands':    {'region': 'Europe', 'subregion': 'Western Europe'},
}

# ---------- HELPERS ----------
def choose_geo_col_for_color(levels_to_use):
    return "subregion" if "Subregion" in levels_to_use and "subregion" in df.columns else "region"

def build_tree(meta: pd.DataFrame):
    tree = {}
    for _, r in meta.iterrows():
        region  = r["region"] if pd.notna(r.get("region")) else "Unknown"
        subreg  = r["subregion"] if ("subregion" in meta.columns and pd.notna(r.get("subregion"))) else None
        country = r["country"]
        if region == "Unknown":
            continue
        tree.setdefault(region, {})
        if subreg and not str(subreg).startswith("Unknown"):
            tree[region].setdefault(subreg, set()).add(country)
        else:
            tree[region].setdefault("_countries", set()).add(country)
    for reg in tree:
        for sub in tree[reg]:
            tree[reg][sub] = sorted(tree[reg][sub])
    return tree

_uid = itertools.count()
def _node_val(prefix, label):
    return f"__{prefix}__::{label}::{next(_uid)}"

def to_nodes(tree: dict):
    nodes = []
    for reg in sorted(tree):
        subs = tree[reg]
        kids = []
        for sub in sorted(subs):
            countries = subs[sub]
            if sub == "_countries":
                kids.extend([{"label": c, "value": c} for c in countries])
            else:
                kids.append({
                    "label": sub,
                    "value": _node_val("subregion", sub),
                    "children": [{"label": c, "value": c} for c in countries]
                })
        nodes.append({"label": reg, "value": _node_val("region", reg), "children": kids})
    return nodes

def safe_tree_select(nodes):
    try:
        sel = tree_select(nodes, check_model="leaf", only_leaf_check=True, expand_all=True)
    except TypeError:
        try:
            sel = tree_select(nodes, check_model="leaf")
        except TypeError:
            sel = tree_select(nodes)
    checked = sel.get("checked", []) if isinstance(sel, dict) else sel
    return [v if isinstance(v, str) else v.get("value", v)
            for v in checked
            if isinstance(v, str) and not v.startswith("__")]

def enforce_geo_labels(df_in, geo_lookup, geo_col, keep_fixed=None):
    """Fill NaNs in geo_col using lookup; optionally force values for given countries.
       If df has no 'country', just return it."""
    if 'country' not in df_in.columns:
        return df_in
    out = df_in.copy()
    fill_map = geo_lookup.set_index('country')[geo_col]
    out[geo_col] = out[geo_col].fillna(out['country'].map(fill_map))
    if keep_fixed:
        for c, vals in keep_fixed.items():
            if c in out['country'].values and geo_col in vals:
                out.loc[out['country'].eq(c), geo_col] = vals[geo_col]
    return out

def force_geo(df_in, mapping=FORCE_GEO):
    if 'country' not in df_in.columns:
        return df_in
    out = df_in.copy()
    for c, vals in mapping.items():
        if c in out['country'].values:
            for col, val in vals.items():
                if col in out.columns:
                    out.loc[out['country'].eq(c), col] = val
    return out

def drop_nan_category(df_in, cat_col, protect=list(FORCE_GEO.keys())):
    """Drop rows where cat_col is NaN/'nan', except protected countries."""
    if cat_col not in df_in.columns:
        return df_in
    out = df_in.copy()
    mask_nan = out[cat_col].isna() | (out[cat_col].astype(str).str.lower() == 'nan')
    if 'country' in out.columns:
        mask_nan &= ~out['country'].isin(protect)
    return out.loc[~mask_nan].copy()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("🔍 Filters")

geo_cols = [c for c in ["region", "subregion", "country"] if c in df.columns]
if "country" not in geo_cols:
    st.error("The dataset needs a 'country' column.")
    st.stop()

geo_meta = df[geo_cols].drop_duplicates().replace({'nan': np.nan})

# Hard corrections in meta
for c, vals in FORCE_GEO.items():
    if c in geo_meta['country'].values:
        for col, val in vals.items():
            if col in geo_meta.columns:
                geo_meta.loc[geo_meta['country'] == c, col] = val

def pick_best(g):
    good = g[g['region'].notna() & (g['region']!='Unknown')]
    if 'subregion' in g.columns:
        good = good[good['subregion'].notna() & ~good['subregion'].str.contains('Unknown', na=True)]
    return good.head(1) if not good.empty else g.head(1)

geo_meta = (geo_meta
            .sort_values(['country','region','subregion'])
            .groupby('country', group_keys=False)
            .apply(pick_best)
            .reset_index(drop=True))

# Inject UK if missing
if "United Kingdom" in df["country"].values and \
   "United Kingdom" not in geo_meta["country"].values:
    uk_row = df[df["country"]=="United Kingdom"][geo_cols].drop_duplicates().head(1)
    if uk_row.empty:
        uk_row = pd.DataFrame([{"region":"Europe","subregion":"Western Europe","country":"United Kingdom"}])
    geo_meta = pd.concat([geo_meta, uk_row], ignore_index=True).drop_duplicates()


# Geography tree
with st.sidebar.expander("🌍 Geography", expanded=True):
    if not TREE_OK:
        st.error("`streamlit-tree-select` is missing. Add it to requirements.txt.")
        st.stop()

    tree_data = build_tree(geo_meta)
    nodes = to_nodes(tree_data)
    picked = safe_tree_select(nodes)
    countries_selected = picked or geo_meta["country"].tolist()

    sel_rows = geo_meta[geo_meta["country"].isin(countries_selected)]
    levels_to_use = []
    if "subregion" in geo_cols and sel_rows["subregion"].nunique() > 1:
        levels_to_use.append("Subregion")
    if sel_rows["region"].nunique() > 1:
        levels_to_use.append("Region")

# Auto-ensure UK is kept if it's in the data (no UI)
if "United Kingdom" in df["country"].values and "United Kingdom" not in countries_selected:
    countries_selected.append("United Kingdom")

color_col = choose_geo_col_for_color(levels_to_use)
protect_val = FORCE_GEO['United Kingdom'][color_col] if color_col in FORCE_GEO['United Kingdom'] else None

# Year
with st.sidebar.expander("📅 Year", expanded=True):
    selected_year = st.slider(
        "Select Year",
        int(df['year'].min()),
        int(df['year'].max()),
        int(df['year'].max()),
        key="year_slider"
    )

# Metric
with st.sidebar.expander("📈 Metric (for Visuals 5-7)", expanded=False):
    selected_metric = st.radio(
        "Select a metric to display:",
        ["renewables_share_pct", "co2_per_capita_t"],
        format_func=lambda x: "Renewables Share (%)" if x == "renewables_share_pct" else "CO₂ per Capita (tonnes)",
        key="metric_radio"
    )

# ---------- APPLY FILTERS ----------
df_year = df[df['year'] == selected_year]
df_filtered = df_year[df_year['country'].isin(countries_selected)].copy()

# Fix geo labels, lock UK/Netherlands, then drop remaining NaNs
for _df in [df_filtered, df_year]:
    _df[:] = enforce_geo_labels(_df, geo_meta[['country', color_col]], color_col, FORCE_GEO)
    _df[:] = force_geo(_df)
    _df[:] = drop_nan_category(_df, color_col)

if df_filtered.empty:
    st.warning("No data for the chosen filters. Adjust your selections.")
    st.stop()


# ---------- PAGE TOP ----------
st.title("Climate Tipping Points: How Renewables & Efficiency Cut CO₂ for a Greener Future")

# Intro paragraph
st.markdown("""
<div style='font-size:18px; line-height:1.6; margin-bottom: 1rem;'>
    This project explores global energy trends, CO₂ emissions, and renewable energy adoption.<br>
    Use the filters to see who’s leading the change and where tipping points are accelerating climate action.
</div>

---
""", unsafe_allow_html=True)


st.markdown(
    """
    <div title="See which filters affect which sections of the dashboard." 
         style='font-size:17px; font-weight:bold; margin-bottom: -8px;'>
        ℹ️ How Filters Work
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("ℹ **Filter Guide** 👈"):
    lvl_txt = ", ".join(levels_to_use) if levels_to_use else "All"
    st.markdown(
        f"""
**Geography (Region/Subregion/Country)** → Applies to Visuals **2–10**  
*(No selection = all countries by default)*

**Year slider** → Applies to Visuals **3-4 6-10** and the **dots/metric in Visual 11**  
*(Visual 2 always shows Year=2000–2020)*

**Metric selector (radio button)** → Applies to Visuals **5–7**

**Visual 11**  
- Heatmap (model surface) = fixed model (not filtered)  
- Red dots & ✖ = selected Geography + Year  
- ⭐ = your scenario → set by sliders inside Visual 10  
- 30% dashed line = tipping point threshold

**Current Selection**  
- Countries: **{len(countries_selected)}**  
- Levels: **{lvl_txt}**  
- Year: **{selected_year}**
        """,
        unsafe_allow_html=True
    )

# ---------- EXPANDER: PROJECT WORKFLOW & METHODOLOGY ----------
with st.expander("📝 Project Workflow & Methodology", expanded=False):
    st.markdown("""
    This dashboard supports the **Climate Tipping Points** project — an investigation into whether crossing a 30% renewables share leads to a faster decline in CO₂ emissions per capita.

    #### 1. Data Collection  
    - Sourced primarily from **Kaggle**, supplemented by **World Bank** and **UNSD** data (2000–2020)  
    - Dataset includes country-level indicators on renewable energy share, CO₂ emissions, GDP, energy intensity, and population  
    - Kaggle dataset: [Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy)  
    - World Bank dataset: [Total Population (WDI)](https://data360.worldbank.org/en/indicator/WB_WDI_SP_POP_TOTL)  
    - UNSD regions: [UN M49 Geoscheme](https://unstats.un.org/unsd/methodology/m49/overview/)

    #### 2. Data Cleaning  
    - Removed nulls in key columns to avoid skewed comparisons and incomplete trends  
    - Normalised country names to ensure grouping and aggregation accuracy  
    - Created derived fields like 5-year change and tipping point flag to support longitudinal and threshold-based analysis

    ### 3. Exploratory Analysis  
    - Used descriptive statistics, boxplots, and correlation heatmaps to understand variation and identify potential drivers  
    - Comparative analysis across tipping-point groups to observe behavioural shifts in emissions

    #### 4. Hypothesis Testing  
    - Applied OLS regression with interaction terms to detect stepwise changes in emission trends post-30% renewables  
    - Used Mann–Whitney U tests for non-parametric comparison of emission levels before and after the threshold  
    - Added WLS weighting by population to reduce over-representation from small-population countries and validate generalisability

    #### 5. Machine Learning  
    - Used Polynomial Regression to capture curved trends, Random Forest for non-linear interactions, and XGBoost for regularised ensemble performance  
    - Model comparison based on R², RMSE, and MAE:  
        - **R² (R-squared)** measures how much of the variance in the target variable is explained by the model — closer to 1 indicates better fit  
        - RMSE and MAE provide measures of average prediction error, with lower values indicating higher accuracy

    #### 6. Visualisation  
    - Interactive dashboard with filters, maps, and prediction sliders to enable scenario testing  
    - Designed to support public understanding and exploratory decision-making
    """)



with st.expander("📊 **Descriptive Analysis**", expanded=False):
    lvl_txt = ", ".join(levels_to_use) if levels_to_use else "All"

    # ---------- PAGE SETUP ----------
    st.markdown("### Descriptive Analysis")

    st.markdown("""
    <div style='font-size:17px; line-height:1.6;'>

    This section lays the foundation for identifying patterns and supporting the <b>Climate Tipping Points hypothesis</b> through key statistics and data exploration.<br><br>

    <b>Descriptive statistics</b> help summarise and interpret data across <b>energy access</b>, <b>CO₂ emissions</b>, and <b>renewables share</b>.<br><br>

    <b>Core concepts include:</b><br>
    • <b>Mean</b> – the average value, offering a sense of central tendency<br>
    • <b>Median</b> – the middle value in an ordered dataset, helpful when data is skewed<br>
    • <b>Standard Deviation</b> – shows how much values vary around the mean<br><br>

    These metrics simplify complex datasets, revealing <b>typical behaviours</b>, <b>outliers</b>, and <b>regional disparities</b> in how countries consume energy or emit carbon.

    </div>
    """, unsafe_allow_html=True)

    # ---------- VARIABILITY & UNCERTAINTY ----------
    st.markdown("#### Understanding Variability & Uncertainty")
    st.markdown("""
    Probability plays a key role in assessing uncertainty. It supports data-driven predictions and hypothesis testing — for example, evaluating whether an increase in <b>climate finance support</b> is associated with a rise in <b>renewables share</b>.
    """, unsafe_allow_html=True)

    # ---------- DATA METRICS (2000–2020) ----------
    st.markdown("#### What the Data Shows (2000–2020)")
    st.markdown("""
    This dashboard opens with a statistical lens on three key metrics:

    **Central tendencies — Mean & Median for:**
    - Renewable energy share
    - CO₂ emissions per capita
    - Energy intensity (MJ per USD)

    **Variability — Standard deviation & variance**, highlighting uneven progress across countries and regions.
    """)

    # ---------- HYPOTHESIS TESTING PLACEHOLDER ----------
    st.markdown("#### Hypothesis Exploration")
    st.markdown("""
    A simple linear regression explores three hypotheses:

    - **H1:** Higher renewable share → lower CO₂ per capita  
    - **H2:** Crossing 30% renewables share acts as a tipping point that accelerates CO₂ decline  
    - **H3:** Lower energy intensity (MJ/$) → lower CO₂ per capita  
    """)

    # Optional vertical space
    st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ---------- VISUAL 1:Summary Boxplot of Key Indicators ----------

st.markdown("### 1. Statistical Summary: Renewable Energy, CO₂ Emissions, and Energy Intensity")

st.markdown("<br><br>", unsafe_allow_html=True)

# Prepare data
df_long = df[['renewables_share_pct', 'co2_per_capita_t', 'energy_intensity_mj_usd']].copy()
df_long = df_long.rename(columns={
    'renewables_share_pct': 'Renewables Share (%)',
    'co2_per_capita_t': 'CO₂ per Capita (t)',
    'energy_intensity_mj_usd': 'Energy Intensity (MJ/USD)'
})
df_long = df_long.melt(var_name='Indicator', value_name='Value')

# Calculate median and mean for display
summary_stats = df_long.groupby('Indicator')['Value'].agg(['mean', 'median']).reset_index()
summary_stats[['mean', 'median']] = summary_stats[['mean', 'median']].round(2)

# Plot
fig = px.box(
    df_long,
    x="Indicator",
    y="Value",
    color="Indicator",
    points=False,
    title="Distribution of Key Indicators (2000–2020)",
    color_discrete_sequence=["#1f77b4", "#2ca02c", "#ff7f0e"]
)

# Add mean and median lines
for i, row in summary_stats.iterrows():
    fig.add_shape(
        type="line",
        x0=i - 0.4,
        x1=i + 0.4,
        y0=row['mean'],
        y1=row['mean'],
        line=dict(color="black", dash="dash"),
        xref="x", yref="y"
    )
    fig.add_shape(
        type="line",
        x0=i - 0.4,
        x1=i + 0.4,
        y0=row['median'],
        y1=row['median'],
        line=dict(color="black", dash="solid"),
        xref="x", yref="y"
    )

fig.update_layout(
    showlegend=False,
    height=500,
    margin=dict(t=50, b=30),
    yaxis_title=None,
    xaxis_title=None,
    title_font_size=18
)

st.plotly_chart(fig, use_container_width=True)

# Display summary table
st.markdown("**Mean and Median Values**")
st.dataframe(summary_stats.rename(columns={'mean': 'Mean', 'median': 'Median'}), use_container_width=True)


st.markdown("""
This boxplot summarises the global spread of three critical metrics between 2000 and 2020. Each plot provides insights into the <b>“typical” situation</b> (via mean and median) and the <b>variability</b> (range and outliers) that characterise different countries’ progress on energy and climate.<br><br>

<b>1. Renewables Share (%)</b><br>
• <b>Mean:</b> 33.4%&nbsp;&nbsp;&nbsp;&nbsp;<b>Median:</b> 24.68%<br>
• The median is below 30%, highlighting that <b>over half of the countries remain below the tipping point threshold</b> identified in Hypothesis 2.<br>
• The wide range (0–98%) and high variance show <b>uneven global adoption</b>—some countries rely heavily on renewables, while many lag far behind.<br>
• This validation can be found in the statistical and visual analyses on whether crossing the <b>30% renewables threshold triggers CO₂ decline presented in the report</b>.<br><br>

<b>2. CO₂ per Capita (tonnes)</b><br>
• <b>Mean:</b> 4.7&nbsp;&nbsp;&nbsp;&nbsp;<b>Median:</b> 2.66<br>
• The mean is skewed upward by a small group of <b>high-emitting countries</b> (some over 20 tonnes per person).<br>
• Most countries emit <b>less than 5 tonnes per capita</b>, showing large disparities in emissions footprints.<br>
• This supports Hypothesis 1 — <b>lower emissions may be tied to higher renewables or better energy efficiency</b>.<br><br>

<b>3. Energy Intensity (MJ/USD)</b><br>
• <b>Mean:</b> 5.36&nbsp;&nbsp;&nbsp;&nbsp;<b>Median:</b> 4.37<br>
• A lower energy intensity means more economic output per unit of energy — a proxy for <b>energy efficiency</b>.<br>
• The right-skew suggests <b>most countries have improved efficiency</b>, though a few remain highly energy intensive.<br>
• This underpins Hypothesis 3 — <b>better energy efficiency tends to correlate with lower CO₂ emissions</b>.<br><br>


<b>Summary:</b><br>
The statistical summary shows that <b>renewables adoption is still below the 30% tipping point for most countries</b>, while <b>CO₂ emissions and energy efficiency vary widely</b>. This highlights the importance of tracking when and how countries cross key thresholds — especially 30% renewables — as potential turning points in the fight against climate change.

---
""", unsafe_allow_html=True)



# ---------- 2. Global Progress Over Time ----------
st.markdown("### 2. Global Averages: Renewables, CO₂ per Capita & Energy Access (2000–2020)")
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

# ---------- 3. Country Leaders ----------
st.markdown("---")
st.markdown("### 3. Country Leaders in Climate Action")

col1, col2 = st.columns(2)

with col1:
    top_renew = df_filtered.sort_values(by='renewables_share_pct', ascending=False).head(10)
    top_renew = force_geo(top_renew)
    top_renew = drop_nan_category(top_renew, color_col)
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
    year_cols = sorted([c for c in co2_diff.columns if isinstance(c, (int, float, np.integer, np.floating))])
    if len(year_cols) >= 2:
        co2_diff = co2_diff.dropna(subset=[year_cols[0], year_cols[-1]])
        co2_diff['change'] = co2_diff[year_cols[-1]] - co2_diff[year_cols[0]]
        top_reducers = co2_diff.nsmallest(10, 'change').reset_index()
        top_reducers = top_reducers.merge(df[['country', color_col]].drop_duplicates(),
                                          on='country', how='left')
        top_reducers = enforce_geo_labels(top_reducers, geo_meta[['country', color_col]], color_col, FORCE_GEO)
        top_reducers = force_geo(top_reducers)
        top_reducers = drop_nan_category(top_reducers, color_col)

        fig2 = px.bar(
            top_reducers,
            x='change', y='country',
            orientation='h', color=color_col,
            title=f"Top 10 CO₂ Reducers ({year_cols[0]}–{year_cols[-1]})"
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Not enough year data to compute CO₂ change.")

# ---------- 4. Emissions vs Efficiency ----------
st.markdown("---")
st.markdown("### 4. CO₂ Emissions vs Energy Intensity (Bubble Size: Renewables Share)")

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

# ---------- 5. Momentum Heatmap ----------
st.markdown("---")
st.markdown("### 5. Renewables Momentum Heatmap (5-Year Change)")

agg_col = color_col
momentum = df[df['country'].isin(countries_selected)].groupby([agg_col, 'year'])[selected_metric].mean().reset_index()
momentum = drop_nan_category(momentum, agg_col)  # no country col? still ok
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

# ---------- 6. Quadrant Chart ----------
st.markdown("---")
st.markdown("### 6. Energy Efficiency vs CO₂ Emissions: Quadrant Analysis")

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


# ---------- 7 CO₂ Distribution Over Time ----------
st.markdown("---")
st.markdown("### 7. CO₂ Distribution Over Time")

# how many years per bucket
bucket = 10

start_year = int(df['year'].min())
end_year   = int(selected_year)

# build bin edges every `bucket` years, ending at selected_year
edges = list(range(start_year, end_year + 1, bucket))
if edges[-1] != end_year:
    edges.append(end_year)
edges.append(end_year + 1)  # upper bound for pd.cut

# labels like "2001–2010"
labels = [f"{edges[i]}–{edges[i+1]-1}" for i in range(len(edges)-1)]

df_box = df[(df['country'].isin(countries_selected)) &
            (df['year'] >= start_year) &
            (df['year'] <= end_year)].copy()

# geo fixes
df_box = enforce_geo_labels(df_box, geo_meta[['country', color_col]], color_col, FORCE_GEO)
df_box = force_geo(df_box)
df_box = drop_nan_category(df_box, color_col)

df_box['period'] = pd.cut(df_box['year'], bins=edges, labels=labels, right=True, include_lowest=True)

fig_box = px.box(
    df_box,
    x='period',
    y='co2_per_capita_t',
    color=color_col,
    labels={'co2_per_capita_t': 'CO₂ per Capita (t)', 'period': 'Time Period'},
    title=f'CO₂ per Capita Distribution by {color_col.title()} (≤ {selected_year})'
)
st.plotly_chart(fig_box, use_container_width=True)

# ---------- 8. Energy Mix Transition ----------
st.markdown("---")
st.markdown("### 8. Energy Mix Transition by Region/Subregion")

mix = df_year[df_year['country'].isin(countries_selected)] \
    .groupby('region')[['fossil_elec_twh', 'renew_elec_twh', 'nuclear_elec_twh']].sum().reset_index()

mix = mix.replace({'nan': np.nan})
mix = mix[mix['region'].notna() & (mix['region']!='Unknown')]

if not mix.empty:
    labels = list(mix['region']) + ['Fossil', 'Renewables']
    source = list(range(len(mix))) * 2
    target = [len(mix)] * len(mix) + [len(mix) + 1] * len(mix)
    value = list(mix['fossil_elec_twh']) + list(mix['renew_elec_twh'])

    fig_sankey = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=40, line=dict(color="black", width=0.8),
                  label=labels,
                  color=["#a1d99b"] * len(mix) + ["#e34a33", "#2b8cbe"] ),
        link=dict(source=source, target=target, value=value,
                  color=["rgba(160,160,160,0.2)"] * len(mix) + ["rgba(34,139,34,0.3)"] * len(mix))
    )])
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.warning("No data available to plot the energy mix Sankey diagram.")

# ---------- 9. Top 5-Year Renewable Gainers ----------
st.markdown("---")
st.markdown("### 9. Top Countries by 5-Year Gain in Renewables")

top_gainers = df[df['year'] == selected_year]
top_gainers = top_gainers[top_gainers['country'].isin(countries_selected)] \
    .sort_values(by='renewables_5yr_change', ascending=False).head(10)
top_gainers = enforce_geo_labels(top_gainers, geo_meta[['country', color_col]], color_col, FORCE_GEO)
top_gainers = force_geo(top_gainers)
top_gainers = drop_nan_category(top_gainers, color_col)

fig_bar = px.bar(
    top_gainers,
    x='country',
    y='renewables_5yr_change',
    labels={'renewables_5yr_change': '5-Year Gain (%)'},
    color=color_col,
    title=f'Top 10 Countries by Renewable Energy Growth ({selected_year})'
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------- 10. Global Map ----------
st.markdown("---")
st.markdown("### 🌍 10. Global View: Energy & Emissions Landscape")
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
map_data = enforce_geo_labels(map_data, geo_meta[['country', color_col]], color_col, FORCE_GEO)
map_data = force_geo(map_data)
map_data = drop_nan_category(map_data, color_col)

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

# ---------- 11. CO₂ per Capita – Predictive What‑If Explorer ----------
st.markdown("---")
st.markdown("### 11. CO₂ per Capita – Predictive Scenario Explorer")

train_cols = ['renewables_share_pct', 'energy_intensity_mj_usd', 'co2_per_capita_t']
df_model = df[train_cols].dropna().copy()
df_model['tip30']    = (df_model['renewables_share_pct'] >= 30).astype(int)
df_model['re_x_ei']  = df_model['renewables_share_pct'] * df_model['energy_intensity_mj_usd']
df_model['re_x_tip'] = df_model['renewables_share_pct'] * df_model['tip30']

y = np.log(df_model['co2_per_capita_t'])
X = df_model[['renewables_share_pct','energy_intensity_mj_usd','tip30','re_x_ei','re_x_tip']]
model = LinearRegression().fit(X, y)

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
Z = np.exp(pred_log).reshape(EI.shape)

heat = go.Heatmap(
    x=r_grid, y=ei_grid, z=Z,
    colorscale="Viridis",
    colorbar=dict(title="Predicted CO₂/t")
)
line30 = go.Scatter(
    x=[30, 30], y=[ei_min, ei_max],
    mode="lines",
    line=dict(color="white", dash="dash"),
    name="30% tipping threshold"
)

need_cols = ['renewables_share_pct', 'energy_intensity_mj_usd']
scatter_df = df[(df['year'] == selected_year) & (df['country'].isin(countries_selected))].copy()
scatter_df = force_geo(scatter_df)
scatter_df = drop_nan_category(scatter_df, color_col)  # color not used here, but safe

if 'United Kingdom' in countries_selected:
    uk_sel = scatter_df[scatter_df['country'] == 'United Kingdom']
    if uk_sel.empty or uk_sel[need_cols].isna().any().any():
        uk_full = df[(df['country'] == 'United Kingdom') & df[need_cols].notna().all(axis=1)]
        if not uk_full.empty:
            idx_near = (uk_full['year'] - selected_year).abs().idxmin()
            uk_clone = uk_full.loc[[idx_near]].copy()
            uk_clone['year'] = selected_year
            scatter_df = pd.concat([scatter_df, uk_clone], ignore_index=True)

scatter_df = scatter_df.dropna(subset=need_cols)

scatter_pts = go.Scatter(
    x=scatter_df['renewables_share_pct'],
    y=scatter_df['energy_intensity_mj_usd'],
    mode="markers",
    marker=dict(size=6, opacity=0.7, color="red"),
    text=scatter_df['country'],
    name="Current Selected Countries"
)

fig_pred = go.Figure(data=[heat, line30, scatter_pts])

with st.expander("Adjust sliders to predict CO₂ emissions", expanded=True):
    compare_pool = df_year[df_year['country'].isin(countries_selected)]
    if compare_pool.empty:
        compare_pool = df[df['country'].isin(countries_selected)]
    compare_options = sorted(compare_pool['country'].unique())

    country_for_diff = st.selectbox("Compare to country (current year)", compare_options)

    base_row = df_filtered[df_filtered['country'] == country_for_diff]
    if base_row.empty:
        re_default = float(compare_pool.loc[compare_pool['country']==country_for_diff,
                                            'renewables_share_pct'].median())
        ei_default = float(compare_pool.loc[compare_pool['country']==country_for_diff,
                                            'energy_intensity_mj_usd'].median())
    else:
        re_default = float(base_row['renewables_share_pct'].iloc[0])
        ei_default = float(base_row['energy_intensity_mj_usd'].iloc[0])

    c1, c2 = st.columns(2)
    with c1:
        re_user = st.slider("Renewables Share (%)", float(r_min), float(r_max), re_default, step=0.5)
    with c2:
        ei_user = st.slider("Energy Intensity (MJ/$)", float(ei_min), float(ei_max), ei_default, step=0.1)

    tip_user = 1 if re_user >= 30 else 0
    X_user = pd.DataFrame({
        'renewables_share_pct':[re_user],
        'energy_intensity_mj_usd':[ei_user],
        'tip30':[tip_user],
        're_x_ei':[re_user*ei_user],
        're_x_tip':[re_user*tip_user]
    })
    co2_pred = float(np.exp(model.predict(X_user))[0])

    base_val = (float(base_row['co2_per_capita_t'].mean())
                if not base_row.empty else
                float(compare_pool.loc[compare_pool['country']==country_for_diff,'co2_per_capita_t'].mean()))
    delta = co2_pred - base_val
    st.metric("Predicted CO₂ per Capita (t)", f"{co2_pred:,.2f}", f"Δ {delta:+.2f} t vs {country_for_diff}")

fig_pred.add_trace(go.Scatter(
    x=[re_user], y=[ei_user],
    mode="markers",
    marker=dict(symbol="star", size=16, color="white", line=dict(color="black", width=1)),
    name="Your scenario"
))

if not np.isnan(base_val):
    if base_row.empty:
        comp_r = float(compare_pool.loc[compare_pool['country']==country_for_diff,'renewables_share_pct'].iloc[0])
        comp_e = float(compare_pool.loc[compare_pool['country']==country_for_diff,'energy_intensity_mj_usd'].iloc[0])
    else:
        comp_r = float(base_row['renewables_share_pct'].iloc[0])
        comp_e = float(base_row['energy_intensity_mj_usd'].iloc[0])

    fig_pred.add_trace(go.Scatter(
        x=[comp_r], y=[comp_e],
        mode="markers",
        marker=dict(symbol="x", size=12, color="yellow", line=dict(width=1, color="black")),
        name=country_for_diff
    ))

fig_pred.add_shape(
    type="rect", x0=30, x1=r_max, y0=ei_min, y1=ei_max,
    line_width=0, fillcolor="rgba(255,255,255,0.04)", layer="below"
)
fig_pred.add_annotation(
    x=30, y=ei_max,
    text="H2: ≥30% renewables → faster CO₂ decline",
    showarrow=False,
    xanchor="left", yanchor="top",
    font=dict(size=11, color="white"),
    bgcolor="rgba(0,0,0,0.35)", borderpad=3
)
fig_pred.add_annotation(
    x=r_max*0.85, y=ei_min + (ei_max-ei_min)*0.15,
    ax=r_max*0.55, ay=ei_min + (ei_max-ei_min)*0.15,
    text="H1: ↑ Renewables  →  ↓ CO₂",
    showarrow=True, arrowhead=2, arrowsize=1.2, arrowcolor="white",
    font=dict(size=11, color="white"),
    bgcolor="rgba(0,0,0,0.35)"
)
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

fig_pred.data[0].showlegend = False
fig_pred.data[0].colorbar.update(x=1.14, len=0.75)

fig_pred.update_layout(
    legend=dict(
        orientation="h",
        x=0.5, y=-0.28,
        xanchor="center", yanchor="top",
        bgcolor="rgba(255,255,255,0.6)",
        borderwidth=0,
        font=dict(size=11)
    ),
    margin=dict(l=0, r=140, t=70, b=190),
    xaxis_title="Renewables Share (%)",
    yaxis_title="Energy Intensity (MJ/$)",
    title={"text": "Predicted CO₂ per Capita Surface (model-based)", "x": 0.02}
)

st.plotly_chart(fig_pred, use_container_width=True)
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)


