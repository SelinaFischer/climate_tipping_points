# streamlit dashboard was created with the help of chatgpt

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# -------------------------
# Data loading (NO widgets here)
# -------------------------
@st.cache_data
def load_data(data/cleaned/enhanced_energy_features_final.csv) -> pd.DataFrame:
    df = pd.read_csv(data/cleaned/enhanced_energy_features_final.csv)
    df = df.dropna(subset=['country', 'region', 'co2_per_capita_t'])

    # Normalize UK names
    df["country"] = df["country"].replace({
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "UK": "United Kingdom"
    })
    mask_uk = df["country"].str.contains(r"united\s*kingdom", case=False, na=False)
    df.loc[mask_uk, "country"] = "United Kingdom"
    return df

def build_geo_meta(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = ['region', 'subregion', 'country'] if 'subregion' in df.columns else ['region', 'country']
    geo_meta = df[geo_cols].drop_duplicates()

    # Ensure UK present
    if "United Kingdom" in df["country"].values and "United Kingdom" not in geo_meta["country"].values:
        uk_row = (df[df["country"] == "United Kingdom"][geo_cols].drop_duplicates().head(1))
        if uk_row.empty:
            uk_row = pd.DataFrame([{
                "region": "Europe",
                "subregion": "Northern Europe" if "subregion" in geo_cols else None,
                "country": "United Kingdom"
            }])
        geo_meta = pd.concat([geo_meta, uk_row], ignore_index=True).drop_duplicates()
    return geo_meta

# Tree helpers (no widgets)
def build_tree(geo_meta: pd.DataFrame):
    tree_data = {}
    for _, row in geo_meta.iterrows():
        region = row['region']
        subregion = row['subregion'] if 'subregion' in row else None
        country = row['country']
        if region not in tree_data:
            tree_data[region] = {}
        if subregion and subregion not in tree_data[region]:
            tree_data[region][subregion] = []
        if subregion:
            tree_data[region][subregion].append(country)
        else:
            tree_data[region].setdefault('countries', []).append(country)
    return tree_data

def to_nodes(tree_data):
    nodes = []
    for region, sub_data in tree_data.items():
        region_node = {'label': region, 'value': region, 'children': []}
        for subregion, countries in sub_data.items():
            if subregion != 'countries':
                sub_node = {
                    'label': subregion,
                    'value': subregion,
                    'children': [{'label': c, 'value': c} for c in countries]
                }
                region_node['children'].append(sub_node)
            else:
                region_node['children'].extend([{'label': c, 'value': c} for c in countries])
        nodes.append(region_node)
    return nodes

def safe_tree_select(nodes):
    try:
        from streamlit_tree_select import tree_select
        selected = tree_select(nodes, key="geo_tree")
        return selected.get('checked', [])
    except ImportError:
        st.warning("streamlit_tree_select not installed. Using default countries.")
        return ['Germany', 'France']  # Optional fallback

# -------------------------
# Main app
# -------------------------
def main():
    st.title("CO₂ Emissions Dashboard")

    # ---- Resolve CSV path ----
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "enhanced_energy_features_final.csv"

    if not DATA_PATH.exists():
        st.error(f"'enhanced_energy_features_final.csv' not found in {BASE_DIR}")
        uploaded = st.sidebar.file_uploader("Upload enhanced_energy_features_final.csv", type="csv")
        if uploaded is None:
            st.stop()
        df = pd.read_csv(uploaded)
        df = df.dropna(subset=['country', 'region', 'co2_per_capita_t'])
    else:
        df = load_data(DATA_PATH)

    # Build geo meta outside cache so we can debug with widgets safely
    geo_meta = build_geo_meta(df)

    # ---- Sidebar widgets (safe) ----
    st.sidebar.header("Filters")

    # Debug toggles
    show_uk_debug = st.sidebar.checkbox("🛠 Show UK debug", False)
    show_geo_debug = st.sidebar.checkbox("🛠 Show geo_meta debug", False)
    show_tree_debug = st.sidebar.checkbox("🛠 Show tree debug", False)

    # Year selector
    years = sorted(df['year'].dropna().unique().astype(int))
    if len(years) == 0:
        st.error("No 'year' values found in the dataset.")
        st.stop()
    if len(years) == 1:
        selected_year = years[0]
        st.sidebar.info(f"Only one year available: {selected_year}")
    else:
        selected_year = st.sidebar.slider("Select Year",
                                          min_value=years[0],
                                          max_value=years[-1],
                                          value=years[-1])

    # Geography tree selector
    tree_data = build_tree(geo_meta)
    nodes = to_nodes(tree_data)
    if show_tree_debug:
        st.write("Tree nodes:", nodes)

    countries_selected = safe_tree_select(nodes)

    # Force include UK if checkbox set
    force_uk = st.sidebar.checkbox("Force include United Kingdom", value=True, key="force_uk_cb")
    if force_uk and "United Kingdom" in df["country"].values and "United Kingdom" not in countries_selected:
        countries_selected = list(countries_selected) + ["United Kingdom"]

    # Sidebar debug output
    st.sidebar.write("Selected countries:", countries_selected)

    # ---- Debug prints (safe because not cached) ----
    if show_uk_debug:
        st.write("UK in dataset?", df['country'].eq("United Kingdom").any())
        if df['country'].eq("United Kingdom").any():
            st.write("UK data sample:", df[df['country'] == "United Kingdom"][
                ['country', 'year', 'renewables_share_pct', 'energy_intensity_mj_usd', 'co2_per_capita_t']
            ].head())

    if show_geo_debug:
        st.write("UK in geo_meta?", geo_meta['country'].eq("United Kingdom").any())
        st.write("geo_meta sample:", geo_meta[geo_meta['country'] == "United Kingdom"])

    # ---------------- Visual 10 -----------------
    st.subheader("Visual 10: What-If Explorer")

    df_year = df[df['year'] == selected_year]
    df_filtered = df_year[df_year['country'].isin(countries_selected)]

    # Debug: Check UK in filtered data
    st.write("UK in df_filtered?",
             df_filtered[df_filtered['country'] == "United Kingdom"][
                 ['country', 'year', 'renewables_share_pct', 'energy_intensity_mj_usd', 'co2_per_capita_t']
             ])

    # Failsafe for UK missing in selected year
    uk_data = df_year[df_year['country'] == "United Kingdom"]
    if "United Kingdom" in countries_selected and uk_data.empty and "United Kingdom" in df['country'].values:
        st.warning("UK data missing for selected year. Using latest available year.")
        uk_data = df[df['country'] == "United Kingdom"].sort_values('year', ascending=False).head(1)
        df_filtered = pd.concat([df_filtered, uk_data], ignore_index=True)

    # Create synthetic heatmap (replace with real model)
    x_range = np.linspace(0, 100, 50)  # Renewables share (%)
    y_range = np.linspace(0, 20, 50)   # Energy intensity (MJ/USD)
    Z = np.random.rand(50, 50) * 10    # Simulated CO₂ per capita

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Heatmap(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Predicted CO₂ per capita (t)")
    ))

    # Add scatter points
    df_filtered = df_filtered.dropna(subset=['renewables_share_pct', 'energy_intensity_mj_usd'])
    if not df_filtered.empty:
        fig_pred.add_trace(go.Scatter(
            x=df_filtered['renewables_share_pct'],
            y=df_filtered['energy_intensity_mj_usd'],
            mode="markers",
            marker=dict(size=6, opacity=0.7, color="red"),
            text=df_filtered['country'],
            name="Current countries"
        ))

    # Compare to country
    compare_pool = df_year[df_year['country'].isin(countries_selected)]
    if compare_pool.empty:
        compare_pool = df[df['country'].isin(countries_selected)]
    compare_options = sorted(compare_pool['country'].unique())
    st.write("Compare options:", compare_options)

    default_index = compare_options.index("United Kingdom") if "United Kingdom" in compare_options else 0
    compare_country = st.selectbox("Compare to country", compare_options, index=default_index)

    if compare_country:
        compare_data = df_filtered[df_filtered['country'] == compare_country]
        if not compare_data.empty:
            fig_pred.add_trace(go.Scatter(
                x=compare_data['renewables_share_pct'],
                y=compare_data['energy_intensity_mj_usd'],
                mode="markers",
                marker=dict(size=10, color="blue", symbol="star"),
                text=[compare_country],
                name=f"{compare_country} (selected)"
            ))

    fig_pred.update_layout(
        xaxis_title="Renewables Share (%)",
        yaxis_title="Energy Intensity (MJ/USD)",
        height=500
    )
    st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()
