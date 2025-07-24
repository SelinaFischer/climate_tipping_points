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
| **H1: Renewables Share vs CO₂**<br>“Higher renewables share is associated with lower CO₂ per capita.” | - **Data work:** Extract renewables share and CO₂ per capita from cleaned dataset<br>- **Stats:** Pearson or Spearman correlation, plus OLS regression<br>- **Dashboard:** Line chart + scatterplot with trendline to visualise the relationship |
| **H2: Renewables Tipping Point**<br>“Above 30% renewables, CO₂ declines accelerate.” | - **Data work:** Flag years with ≥30% renewables share<br>- **Stats:** Segmented (breakpoint) regression with interaction term; Mann–Whitney U test as needed<br>- **Dashboard:** Before/after bar chart + tipping point alert |
| **H3: Energy Intensity vs CO₂**<br>“Lower energy intensity (MJ per $ of GDP) correlates with lower CO₂ per capita.” | - **Data work:** Integrate energy intensity metric and GDP<br>- **Stats:** Spearman correlation and OLS regression controlling for GDP<br>- **Dashboard:** Bubble or bar chart comparing countries |









## Dataset Content
* Describe your dataset. Choose a dataset of reasonable size to avoid exceeding the repository's maximum size of 100Gb.


## Business Requirements
* Describe your business requirements


## Hypothesis and how to validate?
* List here your project hypothesis(es) and how you envision validating it (them) 

## Project Plan
* Outline the high-level steps taken for the analysis.
* How was the data managed throughout the collection, processing, analysis and interpretation steps?
* Why did you choose the research methodologies you used?

## The rationale to map the business requirements to the Data Visualisations
* List your business requirements and a rationale to map them to the Data Visualisations

## Analysis techniques used
* List the data analysis methods used and explain limitations or alternative approaches.
* How did you structure the data analysis techniques. Justify your response.
* Did the data limit you, and did you use an alternative approach to meet these challenges?
* How did you use generative AI tools to help with ideation, design thinking and code optimisation?

## Ethical considerations
* Were there any data privacy, bias or fairness issues with the data?
* How did you overcome any legal or societal issues?

## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).
* How were data insights communicated to technical and non-technical audiences?
* Explain how the dashboard was designed to communicate complex data insights to different audiences. 

## Unfixed Bugs
* Please mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation are not valid reasons to leave bugs unfixed.
* Did you recognise gaps in your knowledge, and how did you address them?
* If applicable, include evidence of feedback received (from peers or instructors) and how it improved your approach or understanding.

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
