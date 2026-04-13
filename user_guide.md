# Procurement Intelligence Dashboard — User Guide

Welcome to the **Procurement Intelligence Dashboard**. This advanced analytics platform is designed to provide actionable insights into your procurement operations, inventory health, supplier performance, and financial planning.

---

## 🚀 Quick Start
To access the dashboard, run the following command in your terminal:
```bash
streamlit run streamlit_app.py
```
The dashboard will open in your default web browser at `http://localhost:8501`.

---

## 🎛️ Navigation & Filtering

### Sidebar Filters
The sidebar allows you to slice and dice the data to focus on specific areas:
- **Business Unit**: Filter by specific BU (e.g., Karachi, Lahore).
- **SKU Category**: Select specific product categories (e.g., Oil & Ghee, Beverages).
- **Manufacturer**: Focus on a single supplier's performance.
- **SKU Type**: Filter by internal classification (Alpha, Bravo, Charlie).
- **SKU Focus**: Filter "Core" SKUs (High Performance) vs "Non-Core".
- **OOS Filter**: Only see Out-of-Stock items for quick action.

---

## 📊 Dashboard Tabs Explained

### 1. 📊 Overview
The landing page provides a high-level summary of your procurement metrics.
- **KPI Cards**: Total Purchase Value, Live Inventory Value, MTD Sales vs Target, Gap Demand, OOS Rate, and Average Coverage.
- **Charts**: Revenue by Business Unit, Category Share (Pie Chart), and Treemaps for value distribution.

### 2. 🏭 Supplier Analysis
Evaluate and manage supplier risk and performance.
- **Performance Matrix**: Scatter plot comparing Purchase Value vs OOS Rate.
- **Supplier Funnel**: Top suppliers contributing to the majority of spend.
- **Scorecard**: A detailed table ranking manufacturers by SKU count, value, and target achievement.

### 3. 📦 Inventory Health
Deep dive into your stock levels and coverage.
- **Coverage Distribution**: See how many weeks of stock you have on average.
- **Stock Status**: Categorizes SKUs into **Critical**, **Low**, **Healthy**, or **Overstocked**.
- **AWS vs MSL**: Visual comparison of Available stock vs Average Weekly Sales (AWS) and Minimum Stock Level (MSL).

### 4. 🎯 Target Tracking
Monitor how well you are performing against set procurement targets.
- **MTD vs Target**: Actual Sales vs Target budget by Business Unit.
- **Achievement Distribution**: Breakdowns of SKUs meeting 50%, 80%, or 100%+ of their targets.

### 5. 🔬 SKU Deep Dive
For analysts who want to explore the raw data.
- **SKU Explorer**: A flexible scatter plot where you can choose X and Y axes (e.g., Price vs Quantity).
- **Pareto (80/20) Analysis**: Identifies the "Power SKUs" driving 80% of your business value.

### 6. ⚡ Alerts & Insights
Data-driven recommendations and automated flags.
- **Automated Alerts**: Flags "Critical" stock, high OOS suppliers, and achievement gaps.
- **Strategic Insights**: AI-driven summaries on supplier concentration and growth opportunities.

### 7. 📋 SKU Level Detail
The most granular view of your inventory.
- **Grouped Manufacturer View**: See performance rolled up to the supplier level.
- **Full SKU Grid**: Searchable, sortable, and color-coded table containing every field (Price, Cost, Gap, AWS, etc.).
- **Download**: Export your filtered view as a CSV for offline use.

### 8. 🔮 Zero SKU Demand Forecast
Predictive replenishment for SKUs that have had zero sales but remain essential.
- **Significance Filtering**: Uses historical importance (LTDS) to filter out noise and focus on critical demand.
- **Demand Blending**: Calculates "Capped Demand" to ensure you don't over-order based on targets.

### 9. 💼 Working Capital (WC)
Financial optimization and cash flow planning.
- **WC Requirement**: Calculated based on Target **DOC (Days of Cover)**.
- **Excess/Under Deployment**: Identifies where you have too much cash tied up in stock vs where you need to invest.
- **DOC Analysis**: Compares "Current DOC" vs "Target DOC" by category.

---

## 📏 Key Metrics & Terminology

| Metric | Definition |
| :--- | :--- |
| **AWS** | **Average Weekly Sales**: The volume typically sold in a 7-day period. |
| **MSL** | **Minimum Stock Level**: The threshold below which a SKU is considered "Critical". |
| **OOS** | **Out of Stock**: SKUs with zero available units. |
| **Coverage** | **Inventory Coverage**: (Available Units / AWS). Tells you how many weeks your current stock will last. |
| **MTD** | **Month to Date**: Sales or procurement volume from the start of the current month. |
| **DOC** | **Days of Cover**: How many days of sales your current inventory supports. |
| **Gap Demand** | The financial value required to bring inventory up to the target level. |

---

## 🛠️ Data Maintenance
The dashboard reads from `1776035327944_db_proc.csv`. To update the dashboard:
1. Replace the CSV with the latest export.
2. Ensure column headers remain consistent.
3. The app will automatically clean and recalculate all metrics upon the next load (or click **"R"** on your keyboard while in the app to refresh).

---
*Created for Procurement Intelligence Optimisation.*
