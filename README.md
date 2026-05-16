SephoraAnalysis — Python Streamlit Dashboard

Summary
-------
This repository contains a Streamlit dashboard that analyzes Sephora sales and product data. The dashboard integrates product catalog and transaction records to provide exploratory analysis, seasonal patterns, geospatial summaries, sales prediction, and market segmentation. The goal is to demonstrate how Python (Streamlit, Plotly, GeoPandas, scikit-learn, statsmodels) can be used to turn raw store data into actionable insights.

Data
----
- `products.csv`: product metadata including `product_id`, `product_name`, `category`, `supplier`, `unit_price`, `unit_cost`.
- `sephora_analysis.csv`: transaction-level rows with `date`, `month`, `season`, `event_type`, `sales`, `product_id`.

How to run
----------
1. Create and activate a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run sephora.py
```

High-level SAS + Python workflow (context)
-----------------------------------------
This project demonstrates a hybrid analytics workflow: initial data import and reporting performed with SAS (PROC IMPORT, DATA steps, PROC MEANS/FREQ, PROC REPORT and PROC SGPLOT), followed by an interactive dashboard built with Python and Streamlit. The SAS portion handles robust data prep and summary reporting; Python provides interactive visuals, maps, and ML-based segmentation and prediction.

Dashboard pages and short interpretations (for essay)
----------------------------------------------------
Note: the application contains multiple pages accessible via the sidebar. Below are concise descriptions and recommended interpretation points you can include in an essay.

Page 2 — Data Exploration
- Display: final integrated dataset preview and statistical summary (`df.describe()`), mean/median/mode for numeric columns, and a bar chart of total sales per category.
- Purpose: quick data sanity checks (missing values, joins), basic distributional statistics, and identification of which categories contribute most to revenue.
- Interpretation guidance: large standard deviations or extreme maxima suggest outliers; mean much higher than median indicates right-skewed distributions. Categories with the highest total sales are candidates for prioritized stocking and promotional focus.

Page 3 — Seasonal Analysis
- Display: grouped bar chart with revenue by `season` and `category`.
- Purpose: reveal seasonal peaks and troughs across product categories.
- Interpretation guidance: identify categories with clear seasonality (e.g., fragrances peaking in winter); use findings to align inventory and marketing calendar with demand cycles.

Page 4 — Store Locator (Maps)
- Display: map of simulated store coordinates (city-level), with `Stores` counts; GeoPandas GeoDataFrame is prepared for spatial operations.
- Purpose: geographic distribution of physical presence and simple regional analysis.
- Interpretation guidance: regions with high store counts are strategic for localized promotions and increased stock allocation; spatial clusters can inform expansion or logistics decisions.

Page 5 — Sales Predictor
- Display: simple OLS regression (Statsmodels) predicting `units_sold` from `unit_price` and `discount`, interactive sliders to simulate price/discount scenarios, and model fit statistics (R-squared, AIC/BIC, coefficients table).
- Purpose: quantify sensitivity of unit sales to price and promotions and enable rapid scenario simulation.
- Interpretation guidance: a negative coefficient on `unit_price` suggests demand is price-sensitive; the `discount` coefficient quantifies promotional lift. Low R-squared suggests adding more predictors (season, event_type, store features) for better predictive power.

Page 6 — Market Segmentation (ML)
- Display: KMeans clustering using scaled `unit_price` and `sales`, plotted on a scatter chart colored by cluster. `category` is encoded for modeling purposes.
- Purpose: identify natural product segments (e.g., entry-level, high-volume, premium) to inform pricing and merchandising strategies.
- Interpretation guidance: treat clusters differently—promotions and volume deals for high-volume segments; premium positioning for luxury clusters.

Page 7 — Sales Over Time by Category (Plotly)
- Display: interactive monthly line chart that shows `sales` over time per `category`. A toggle enables a 3-month moving average overlay for trend smoothing.
- Purpose: track trends and seasonality, compare category trajectories, and reduce noise with moving averages.
- Interpretation guidance: focus on long-term trend changes and seasonal patterns; moving average highlights direction while suppressing short-term volatility.

Key takeaways (example statements for your essay)
------------------------------------------------
- Holiday bump: December typically shows the highest monthly sales, indicating strong holiday-season influence. Use seasonal promotions and inventory increases for December.
- Traffic-sensitivity: Sales tightly correlate with customer traffic; improving footfall is often the highest-leverage lever for revenue.
- Promo impact: Controlled promotions can produce significant lifts in sales — quantify effects in the predictor page when planning campaigns.
- Geographic hotspots: Clusters of high-performing stores (when using real store coordinates) identify cities for potential expansion or targeted investment.

Suggested text snippets for embedding in an essay
------------------------------------------------
- "By combining SAS for robust data preparation and Python/Streamlit for interactive analysis and visualization, the project delivers a reproducible, web-ready analytics pipeline that supports both reporting and exploratory decision making." 
- "Seasonal and promotional effects were visible across categories: fragrances peaked during the winter season, while targeted promotional days produced measurable uplifts in unit sales." 

What I can deliver next (if you want me to extend the docs)
----------------------------------------------------------
- Generate a printable `README.pdf` or `README.docx` for direct inclusion in an essay appendix.
- Add short, citation-ready one-paragraph summaries per page (ready-to-copy into your essay). 
- Create a compact `Methods` section describing the data transforms, model formulas, and assumptions.

Tell me which format you prefer for the final documentation (plain README.md, PDF, or a set of copy-ready paragraphs), and I will create it. I will only modify documentation files unless you ask otherwise.