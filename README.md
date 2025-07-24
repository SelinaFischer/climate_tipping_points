# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)


## 🌍 Climate Tipping Points: How Renewables & Efficiency Cut CO₂ for a Greener Future

### Introduction & Motivation

As climate urgency grows, the European Union has raised its renewable energy targets — from **32% by 2030** under the *Recast Renewable Energy Directive (2018/2001/EU)* to **at least 42.5%**, with an ambition to reach **45%**, under the *Revised Renewable Energy Directive (EU/2023/2413)*  
([source](https://energy.ec.europa.eu/topics/renewable-energy/renewable-energy-directive-targets-and-rules/renewable-energy-targets_en)).

But are these thresholds enough to trigger meaningful decarbonisation?

In this independent project, I investigate **climate tipping points** — moments when renewable energy adoption and efficiency improvements begin to produce measurable reductions in CO₂ emissions. Using global datasets, I apply statistical validation and interactive visualisation to identify key drivers accelerating or hindering the path to net zero.

The analysis explores:
- Whether exceeding a 30% renewables share represents a tipping point in emissions per capita,
- The role of energy efficiency and financial flows in influencing decarbonisation,
- And which regions are advancing — or lagging — in the energy transition.

The aim is to transform complex data into practical insights that highlight where climate action is taking hold — and where greater momentum is needed.


### Project Objectives

- Quantify how renewable energy adoption and energy efficiency relate to CO₂ emissions across countries and over time.
- Detect structural **tipping points** — thresholds where emission reductions begin to accelerate.
- Deliver actionable insights through an interactive dashboard designed to support policy decisions and further research.


### Business Case

Uneven global progress on decarbonisation highlights the need for data-driven insights to guide investments and policies where they can deliver the greatest impact.


### Project Overview

This project analyses and visualises the key drivers influencing sustainable energy transitions and climate outcomes across countries — identifying where and how tipping points for positive change can be reached. It combines data analytics, statistical validation, and an interactive Streamlit dashboard to equip decision-makers and the public with clear, actionable insights.


### Problem Statement

Despite global commitments through initiatives like the EU Green Deal and COP28, progress toward decarbonisation remains uneven. Key drivers such as GDP, population, economic development, energy policy, and energy efficiency vary significantly across countries and regions. This project uncovers these disparities and identifies tipping points where climate action begins to accelerate.


### Project Plan

**Tools & Workflow:**  
- GitHub repo with Kanban board for task tracking  
- Development and analysis in VS Code  
- Deployment via Streamlit Community Cloud  

**High-Level Agile Project Plan (5 Days)**

| Day   | Tasks                                                                 |
|-------|-----------------------------------------------------------------------|
| Day 1 | ETL: Clean, normalise, and merge variables; create threshold flag     |
| Day 2 | EDA: Summary statistics, correlation matrix, and initial visualisation |
| Day 3 | Statistical tests (H1, H3) and breakpoint analysis (H2)               |
| Day 4 | Build Streamlit components (plots + narrative), test, fix bugs, deploy |
| Day 5 | Documentation: Write README, summarise conclusions, update project board |

**Data Management Approach:**  
Data was collected from public sources (Kaggle, World Bank, UNSD) and harmonised across time and geography. Processing involved renaming columns, merging population data, handling missing values, creating per-capita and log-transformed features, and engineering indicators such as `above_30_pct`. Cleaned datasets were stored in GitHub and reused across both analysis and dashboard layers.

**Methodology Rationale:**  
The project combined descriptive analysis with statistical validation to balance clarity and rigour. Correlation analysis was used to explore associations, while OLS and segmented regression tested the strength and structure of relationships. This allowed both quantitative insight and policy-relevant interpretation, especially regarding the 30% renewables tipping point.



### MVP Deliverables

- Cleaned and enriched dataset enabling per-capita and regional analysis  
- Statistical validation of three climate-related hypotheses  
- Deployed Streamlit dashboard featuring at least four types of visualisations  
- Comprehensive README and annotated Jupyter Notebooks documenting the workflow


### Contingency Plan

- **Dashboard fallback:** If Streamlit deployment runs into issues, I’ll consult my AI buddies for alternatives and pivot to Tableau Public or Power BI if needed.  
- **Analysis support:** When stuck, I’ll lean on AI suggestions to explore alternative methods or visualisation ideas.


### Hypotheses & Deliverables

| Hypothesis | Rationale & Deliverables |
|-----------|---------------------------|
| **H1: Renewables Share vs CO₂**  
“Higher renewables share is associated with lower CO₂ per capita.” |  
- **Data work:** Extract `renewables_share_pct` and `co2_per_capita_t` from the cleaned dataset  
- **Stats:** Spearman correlation and OLS regression  
- **Dashboard:** Line chart and scatterplot with trendline to visualise the relationship |
| **H2: Renewables Tipping Point**  
“Above 30% renewables, CO₂ declines accelerate.” |  
- **Data work:** Use `above_30_pct` binary column to compare groups  
- **Stats:** Segmented regression with interaction term (`above_30_pct * renew_share`); Mann–Whitney U test (optional)  
- **EDA Visual:** 2020-only bar chart showing average CO₂ per capita below vs above 30%  
- **Narrative:** Tipping point hypothesis supported by structural difference in emissions |
| **H3: Energy Intensity vs CO₂**  
“Lower energy intensity (MJ per $ of GDP) correlates with lower CO₂ per capita.” |  
- **Data work:** Integrate `energy_intensity_mj_usd` and `gdp_pc_usd`  
- **Stats:** Spearman correlation and OLS regression controlling for GDP  
- **Dashboard:** Bubble or scatterplot by country, coloured by GDP |



### Dataset Content

This project integrates and transforms multiple global datasets to enable robust, cross-country analysis of energy transitions and emissions between 2000 and 2020.

#### Sources & Time Span

- **Global Sustainable Energy Dataset**  
  Country-year panel of sustainable energy indicators  
  **Source:** [Kaggle - Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy)  
  - Raw file: `global-data-on-sustainable-energy_raw.csv`  
  - Time span: 2000–2020  
  - Dimensions: 3,649 rows × 21 columns

- **World Bank Population (SP.POP.TOTL)**  
  Official population estimates for all countries  
  - Original coverage: 1960–2023  
  - Subset for this project: 2000–2020  
  - Used to calculate per capita CO₂ emissions and energy metrics  
  **Source:** [World Bank Data 360](https://data360.worldbank.org/en/indicator/WB_WDI_SP_POP_TOTL)

- **UNSD M49 Region Mapping**  
  Static country classification used for regional analysis  
  **Source:** [UNSD M49 Overview](https://unstats.un.org/unsd/methodology/m49/overview/)


#### Data Transformation Summary

| Step | Description |
|------|-------------|
| Rename columns | Standardised all column names for clarity and compatibility (e.g. `Access to electricity (%)` → `elec_access_pct`) |
| Normalisation | Merged in population data to create per capita metrics (e.g. `co2_per_capita_t`) |
| Regional enrichment | Mapped `region` and `subregion` using UNSD M49 classification |
| Derived features | Added engineered variables such as:  
&nbsp;&nbsp;– `log_co2_per_capita_t` (log-transformed emissions)  
&nbsp;&nbsp;– `above_30_pct` (binary indicator for tipping point analysis)  
&nbsp;&nbsp;– `year_offset` (used for trend-based modelling)  
&nbsp;&nbsp;– `renewables_3yr_avg` (3-year trailing average) |
| Missingness tracking | Added `_miss` columns to capture imputed or missing values for data quality checks |

#### Cleaned Dataset Overview

- File: `enhanced_energy_features_final.csv`  
- Dimensions: 3,649 rows × 37 columns  
- Format: Panel data (each row represents a country-year observation)  
- Key improvements:  
  - Per-capita emissions and energy indicators  
  - Log transformations for skewed variables  
  - Policy-relevant binary flag for ≥30% renewables tipping point

This enhanced dataset supports all statistical testing, hypothesis validation, and dashboard visualisations used throughout the project.


### Business Requirements

- Identify and visualise the key drivers of the global energy transition  
- Provide actionable insights to support decision-making by policymakers, analysts, and researchers  
- Enable scenario exploration by country, region, subregion, and year to reveal geographic and temporal patterns  
- Communicate complex findings in a clear, accessible way through data storytelling and interactive visualisation


## Hypotheses and How to Validate

This project is guided by three hypotheses related to the role of renewable energy and energy efficiency in reducing CO₂ emissions:

### H1: Renewables Share vs CO₂ per Capita  
**Hypothesis:** Countries with a higher share of renewable energy have lower CO₂ emissions per capita.  
**Validation Approach:**  
- Calculate correlations (Spearman and OLS regression) between `renewables_share_pct` and `co2_per_capita_t`  
- Visualise relationships using scatterplots with trendlines

### H2: Tipping Point at 30% Renewables  
**Hypothesis:** Countries with a renewables share above 30% experience a structural decline in CO₂ emissions.  
**Validation Approach:**  
- Create a binary flag (`above_30_pct`) to compare average emissions below and above the threshold  
- Apply segmented regression with interaction terms and visualise results  
- Support with a 2020-only bar chart comparing the two groups

### H3: Energy Intensity vs CO₂ per Capita  
**Hypothesis:** Countries with lower energy intensity (MJ per $ GDP) emit less CO₂ per capita.  
**Validation Approach:**  
- Test correlation between `energy_intensity_mj_usd` and `co2_per_capita_t`  
- Use OLS regression controlling for `gdp_pc_usd` to account for development level  
- Plot emissions vs energy intensity using scatterplots, coloured by GDP

Each hypothesis is tested using both statistical validation and visual storytelling to ensure interpretability and support evidence-based conclusions.


## The Rationale to Map the Business Requirements to the Data Visualisations

This project’s data visualisations were designed to directly support the core business requirements through meaningful, interpretable outputs. Below is a breakdown of each requirement and the rationale for the selected visual approach.

### 1. Identify and visualise key drivers of the energy transition  
**Visualisations used:**  
- Correlation matrix and scatterplots for key variable pairs  
- Time-series and trendlines showing how renewables share and energy intensity evolve  
**Rationale:**  
These visuals help isolate relationships between variables such as renewables share, energy intensity, and CO₂ emissions, allowing analysts to spot the most influential drivers.

### 2. Provide actionable insights for stakeholders  
**Visualisations used:**  
- Tipping point bar chart comparing CO₂ per capita above vs below 30% renewables  
- Country-level plots with GDP, energy metrics, and emissions  
**Rationale:**  
By visualising structural shifts and emissions performance across thresholds, the dashboard highlights where policy action has been most effective — enabling evidence-based recommendations.

### 3. Enable scenario exploration by country, region, subregion, and year  
**Visualisations used:**  
- Sidebar filters for country, region, and subregion  
- Line charts and scatterplots showing changes over time  
**Rationale:**  
Interactive filters and temporal visualisations allow users to explore emissions and energy patterns geographically and over the 2000–2020 period, enabling scenario analysis across space and time.


### 4. Communicate complex insights accessibly  
**Visualisations used:**  
- Cleanly annotated scatterplots, bar charts, and line graphs  
- Structured narrative embedded in the dashboard  
**Rationale:**  
The use of accessible visual formats and straightforward statistical overlays enables non-technical users to engage with insights without needing advanced analytical skills.

Each visual was chosen not just for analytical clarity but to support communication, exploration, and evidence-based decision-making aligned with stakeholder needs.


## Analysis techniques used
* List the data analysis methods used and explain limitations or alternative approaches.
* How did you structure the data analysis techniques. Justify your response.
* Did the data limit you, and did you use an alternative approach to meet these challenges?
* How did you use generative AI tools to help with ideation, design thinking and code optimisation?


## Analysis Techniques Used

This project combined descriptive statistics, hypothesis-driven regression, and visual analytics to validate the relationships between energy transition indicators and CO₂ emissions. The techniques were chosen to balance statistical rigour with interpretability for both technical and non-technical audiences.

### Analytical Methods

- **Descriptive Statistics & Correlation Analysis**  
  Used to summarise distributions and explore initial relationships between key variables such as `renewables_share_pct`, `energy_intensity_mj_usd`, and `co2_per_capita_t`. Spearman correlation was preferred due to the non-normal distribution of some variables.

- **Ordinary Least Squares (OLS) Regression**  
  Applied to test the strength of association between predictors and outcomes while controlling for GDP and other confounders. Log-transformed variables were used to reduce skew and improve model fit.

- **Segmented (Tipping Point) Regression**  
  Used to test Hypothesis 2 — whether a structural change in emissions occurs when renewables share exceeds 30%. This involved an interaction term (`above_30_pct * renew_share`) to capture changes in slope beyond the threshold.

- **Bar Chart Comparison (2020 Snapshot)**  
  To complement the regression analysis, a simple 2020-only bar chart was used to visually validate whether countries above the 30% renewables threshold had significantly lower emissions — supporting accessibility and visual impact.

### Data Structuring

The dataset was structured as a **panel (country-year)** format, allowing both cross-sectional and longitudinal analysis. Per capita indicators were derived by merging World Bank population data. Log-transformed features and a tipping point binary flag were added to support regression modelling. This structure allowed for flexible slicing across country, region, subregion, and time.

### Limitations & Alternative Approaches

- **Data Gaps:**  
  Some countries had missing population data or incomplete records across years. To address this, rows with critical missing values (e.g. emissions, renewables share) were dropped during cleaning. The impact was monitored using `_miss` indicator flags.

- **Statistical Constraints:**  
  Due to limited data points for some smaller countries or regions, the project did not use fixed-effects panel regression or clustering, which would require more consistent time-series coverage. Instead, analysis focused on aggregate patterns and cross-sectional validation (e.g. 2020 snapshot).

- **Causality:**  
  The analysis is observational and correlational. While relationships are statistically significant, causality cannot be inferred without experimental or time-lagged data.


### Use of Generative AI

Generative AI tools played a valuable supporting role across ideation, analysis, and delivery. They helped streamline workflows, improve clarity, and accelerate code development — while all analysis, validation, and decision-making were independently executed and manually verified.

- **ChatGPT** was used to:
  - Support ideation and framing of hypotheses, such as identifying the 30% renewables tipping point from policy benchmarks
  - Assist in design thinking for dashboard structure, interactive filters, and storytelling flow
  - Optimise and debug Python code, particularly in Streamlit and Plotly visualisations
  - Refine language and structure across documentation, markdown explanations, and README reporting

- **GitHub Copilot** assisted with:
  - Code auto-completion and pattern suggestions in VS Code
  - Fixing minor logic and syntax issues in data wrangling, feature engineering, and visualisation steps
  - Accelerating development during repetitive tasks (e.g. grouped transformations, plot annotation)

- **Abacus.AI** was used to:
  - Generate the project’s visual cover image for the README 
  - Contribute to early-stage ideation and planning of the project workflow

These tools served as valuable thought partners for improving efficiency, visual design, and clarity — while all analysis, validation, and interpretation were conducted independently to ensure the project remained both technically sound and professionally presented.


### Ethical Considerations

This project used publicly available, aggregate-level data from reputable sources including the World Bank, Kaggle, and the United Nations. As such, there were no individual or personally identifiable records involved, and no data privacy risks were present.

Potential fairness or bias concerns were primarily related to:
- **Data coverage and completeness** — Some countries had missing or inconsistent records, especially in earlier years or less economically developed regions.
- **Comparability across contexts** — Emissions and energy data vary widely by economic development level, infrastructure, and policy environment, which may affect the interpretation of cross-country comparisons.

These issues were addressed through:
- Subsetting the dataset to a consistent time period (2000–2020)  
- Using per-capita and log-transformed metrics to normalise for country size and economic scale  
- Visualising trends at regional and subregional levels to avoid overgeneralisation

No legal or intellectual property issues were encountered, as all data sources were open-access and properly cited. The analysis was conducted with the goal of supporting equitable and evidence-based climate planning, avoiding assumptions about causality, and presenting findings in a transparent and accessible manner.



## Dashboard Design

The dashboard was developed using **Streamlit** and evolved iteratively over 20+ design refinements. It provides an intuitive, interactive interface tailored for both technical and non-technical users to explore energy trends, validate hypotheses, and generate insights with clarity and control.

### Sidebar Filters

The sidebar supports dynamic filtering for user-driven exploration:

- **Geography Selector**  
  A nested region–subregion–country tree built using `streamlit-tree-select`. Enables filtering at multiple geographic levels.  

- **Year Slider**  
  Allows users to select a year between 2000 and 2020. All visuals dynamically update accordingly.

- **Metric Toggle**  
  Enables switching between:
  - `renewables_share_pct` – Renewables as a share of total energy
  - `co2_per_capita_t` – CO₂ emissions per capita (tonnes)

---

### Visual Components (11 in Total)

#### 1. Global Trends
- **Line Chart: Global CO₂ per Capita Over Time**
- **Line Chart: Global Renewables Share Over Time**
- **Line Chart: Global Energy Intensity Over Time**  
  _Illustrates long-term patterns to support Hypotheses H1 and H3._

#### 2. Country Leaders
- **Bar Chart: Top 10 Countries by Renewables Share**  
  Ranks countries by renewable energy share in the selected year.

- **Bar Chart: Top 10 CO₂ Reducers**  
  Highlights countries with the largest per-capita CO₂ reductions between the first and last available years.

#### 3. Geographic View
- **Choropleth Map**  
  Visualises CO₂ or renewables across the world using color-coded shading.

- **Heatmap: Metric by Country**  
  Displays values of energy indicators across countries in a matrix format.

#### 4. Metric Explorer
- **Bubble Chart**  
  Explores energy intensity vs. CO₂ per capita, with renewables share as the bubble size.

- **Time Series Breakdown**  
  Enables per-country or per-region tracking of key metrics over time.

#### 5. Tipping Point Test (Hypothesis H2)
- **Bar Chart: CO₂ Per Capita – Above vs. Below 30% Renewables**  
  Shows average emissions grouped by whether countries surpassed the 30% renewables threshold.

#### 6. Predictive Model
- **Interactive Prediction Tool**  
  Users can adjust sliders for `renewables_share_pct` and `energy_intensity` to generate a predicted CO₂ per capita value using an OLS regression model trained from the dataset.


### User Experience & Design Enhancements

- **Use of Expanders**  
  Collapse sections such as:
  - “How the Filters Work”
  - “Descriptive Analysis”
  - “Project Workflow & Methodology”  
  This helps reduce visual noise and focus the reader’s attention on key charts.

- **Colour Consistency & Label Validation**  
  Regional and subregional colors are maintained across visuals. Geographic inconsistencies (e.g., “UK” vs. “United Kingdom”) are corrected in the backend to avoid mislabeling.

- **Error Handling**  
  Empty filters or incompatible selections trigger warning messages and fallback logic to maintain app stability.


### Design Evolution

The dashboard began with basic trend charts and map visualisations. Through a trial-and-error approach, hands-on prototyping, and iterative testing, several key design improvements were introduced:

- Incorporated **expander widgets** to reduce clutter and allow users to focus on key visuals.
- Added **predictive sliders** to make the insights interactive and participatory.
- Introduced **metric toggles, tooltips,** and **segmented visual blocks** to support guided storytelling and exploration.
- Refined **layout, spacing, and legend placement** to improve readability across screen sizes and devices.


### Communicating Insights

- **For Non-Technical Audiences**:  
  Clear labels, interactive sliders, and visual cues make the data approachable. Expanders hide complexity unless needed.

- **For Technical Audiences**:  
  Metric definitions, trend directions, and chart configurations support further exploration and hypothesis testing. Visual outputs align with key statistical validations (e.g., segmented regression for tipping point testing).


The final dashboard communicates complex climate and energy dynamics through a carefully structured and visually accessible interface — enabling data-informed insight, scenario exploration, and public engagement on climate tipping points.


## Unfixed Bugs

### Subregion Misclassification

During post-deployment testing, a notable issue was identified in the **Geography** filter of the dashboard:

- When filtering for countries in **Northern Africa**, a country from **Western Europe** incorrectly appeared in the visualisation.![visual4 bug](image-1.png)
- This suggests a misclassification in the country-to-subregion mapping, despite defensive coding applied via the `FORCE_GEO` dictionary and validation logic.


### Why It Was Not Fixed

Due to time constraints, the issue was not fully investigated or resolved. It is considered a **known bug** and is documented for future improvement. The fix would likely involve:

- Strengthening validation logic in the `geo_meta` DataFrame
- Improving label harmonisation between the dataset and the sidebar filters


### Knowledge Gaps and How They Were Addressed

This issue surfaced several technical and practical gaps:

- **Region Metadata Normalisation:**  
  Handling mismatches in country names (e.g., "UK" vs "United Kingdom") required using a `FORCE_GEO` override and applying `enforce_geo_labels()` to consistently label geography columns.

> This section highlights the challenge of managing diverse global datasets and the importance of robust validation during the final deployment stage.



## Development Roadmap
* What challenges did you face, and what strategies were used to overcome these challenges?
* What new skills or tools do you plan to learn next based on your project experience? 

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. From the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people who provided support through this project.
