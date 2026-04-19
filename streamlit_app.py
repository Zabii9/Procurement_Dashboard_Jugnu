import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Intelligence Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0f1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252a3a 100%);
        border: 1px solid #2d3347;
        border-radius: 10px;
        padding: 12px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin: 0; line-height: 1.2; }
    .metric-label { font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }
    .metric-delta-pos { font-size: 0.75rem; color: #4ade80; font-weight: 600; }
    .metric-delta-neg { font-size: 0.75rem; color: #f87171; font-weight: 600; }
    
    .section-header {
        font-size: 0.95rem; font-weight: 600; color: #e2e8f0;
        border-left: 3px solid #6366f1;
        padding-left: 10px; margin: 12px 0 8px 0;
    }
    
    .badge-alpha { background:#4ade80;color:#052e16;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700; }
    .badge-bravo { background:#60a5fa;color:#0c1a2e;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700; }
    .badge-charlie { background:#f59e0b;color:#1c0a00;padding:2px 10px;border-radius:20px;font-size:0.75rem;font-weight:700; }
    
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #151929 100%);
        border-right: 1px solid #2d3347;
    }
    
    .stMultiSelect [data-baseweb="tag"] { background-color: #6366f1 !important; }
    
    .kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
    
    .alert-box {
        background: linear-gradient(135deg, #2d1515 0%, #3d1f1f 100%);
        border: 1px solid #7f1d1d; border-radius: 8px; padding: 12px 16px;
        color: #fca5a5; font-size: 0.85rem; margin: 8px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2035 0%, #1e2745 100%);
        border: 1px solid #3730a3; border-radius: 8px; padding: 12px 16px;
        color: #a5b4fc; font-size: 0.85rem; margin: 8px 0;
    }
    
    .stApp > header {
        background: linear-gradient(135deg, #1a1f2e 0%, #151929 100%) !important;
        border-bottom: 1px solid #2d3347;
    }
    
    .native-header-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 3.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000000;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "1776035327944_db_proc.csv")
    df = pd.read_csv(csv_path)
    # Clean & enrich
    df["Total buying Amount"] = pd.to_numeric(df["Total buying Amount"], errors="coerce").fillna(0)
    df["Live Total buying Amount"] = pd.to_numeric(df["Live Total buying Amount"], errors="coerce").fillna(0)
    df["Sale"] = pd.to_numeric(df["Sale"], errors="coerce").fillna(0)
    df["OdrValue"] = pd.to_numeric(df["OdrValue"], errors="coerce").fillna(0)
    df["MTD"] = pd.to_numeric(df["MTD"], errors="coerce").fillna(0)
    df["TGT"] = pd.to_numeric(df["TGT"], errors="coerce").fillna(0)
    df["OdrQty"] = pd.to_numeric(df["OdrQty"], errors="coerce").fillna(0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)
    df["MSL in Units"] = pd.to_numeric(df["MSL in Units"], errors="coerce").fillna(0)
    df["Available Total Units"] = pd.to_numeric(df["Available Total Units"], errors="coerce").fillna(0)
    df["AWS in Units"] = pd.to_numeric(df["AWS in Units"], errors="coerce").fillna(0)
    df["MRP"] = pd.to_numeric(df["MRP"], errors="coerce").fillna(0)
    df["LTDS_Amount"] = pd.to_numeric(df["LTDS_Amount"], errors="coerce").fillna(0)
    df["TopSKU"] = df["TopSKU"].fillna("Non-Core")
    df["type"] = df["type"].fillna("Unclassified")
    df["IsOOS"] = df["SubCatNew"].str.contains("OOS", case=False, na=False)

    # ── SKU-level target: split manufacturer TGT by each SKU's LTDS_Amount contribution ──
    # Step 1: true manufacturer-level TGT (max across SKUs per manu, handles repeated values)
    manu_tgt_true = df.groupby("ManufacturerName")["TGT"].max().rename("Manu_TGT_True")
    df = df.join(manu_tgt_true, on="ManufacturerName")

    # Step 2: total LTDS per manufacturer
    manu_ltds_total = df.groupby("ManufacturerName")["LTDS_Amount"].sum().rename("Manu_LTDS_Total")
    df = df.join(manu_ltds_total, on="ManufacturerName")

    # Step 3: LTDS contribution % of each SKU within its manufacturer
    df["LTDS_Contribution_Pct"] = np.where(
        df["Manu_LTDS_Total"] > 0,
        df["LTDS_Amount"] / df["Manu_LTDS_Total"],
        1 / df.groupby("ManufacturerName")["SKU Code"].transform("count")  # equal split fallback
    )

    # Step 4: SKU-level target = manu TGT × contribution %
    df["SKU_TGT"] = (df["Manu_TGT_True"] * df["LTDS_Contribution_Pct"]).round(2)

    # Keep original TGT column intact (manu-level) for reference; use SKU_TGT everywhere else
    df["TGT_Ach_Pct"] = np.where(df["SKU_TGT"] > 0, (df["MTD"] / df["SKU_TGT"] * 100).clip(0, 200), 0)
    df["Inventory_Coverage"] = np.where(df["AWS in Units"] > 0, df["Available Total Units"] / df["AWS in Units"], 0)
    df["Stock_Status"] = pd.cut(
        df["Inventory_Coverage"],
        bins=[-np.inf, 0.5, 1.5, 3, np.inf],
        labels=["Critical", "Low", "Healthy", "Overstocked"]
    )

    # ── Demand Value Calculation ─────────────────────────────────────────────
    # Unit Cost = AWSActiveAmt / AWSAtvPcs (piece-level; fallback to AWS in Units)
    df["Unit_Cost"] = np.where(
        df["AWSAtvPcs"] > 0,
        df["AWSActiveAmt"] / df["AWSAtvPcs"],
        np.where(df["AWS in Units"] > 0, df["AWSActiveAmt"] / df["AWS in Units"], 0)
    )

    # Gap Units = units short of AWS level right now
    df["Gap_Units"] = (df["AWS in Units"] - df["Available Total Units"]).clip(lower=0)

    # Full AWS Demand Value (reference — how much to fully stock to AWS)
    df["AWS_Demand_Value"] = (df["Unit_Cost"] * df["AWS in Units"]).round(2)

    # ── TGT-Implied Units: max units we should order within the SKU target budget ──
    # Demand can NEVER exceed the SKU target value
    df["TGT_Implied_Units"] = np.where(
        df["Unit_Cost"] > 0,
        df["SKU_TGT"] / df["Unit_Cost"],   # budget-constrained units
        df["Gap_Units"]                     # fallback if no unit cost
    )

    # Demand Units = min(gap needed, what target allows) — capped at target
    df["Demand_Units"] = np.minimum(df["Gap_Units"], df["TGT_Implied_Units"]).clip(lower=0)

    # Gap Demand Value = capped procurement cost (never exceeds SKU_TGT)
    df["Gap_Demand_Value"] = (df["Demand_Units"] * df["Unit_Cost"]).round(2)

    # ── Significance Filters ─────────────────────────────────────────────────
    # Only generate demand for SKUs that genuinely matter in their manufacturer
    # or category — avoids noise from tiny-contribution SKUs

    # 1. Contribution ratio: SKU's LTDS share vs equal share within manufacturer
    df["Manu_SKU_Count"] = df.groupby("ManufacturerName")["SKU Code"].transform("count")
    df["Contrib_Ratio"] = np.where(
        df["Manu_SKU_Count"] > 0,
        df["LTDS_Contribution_Pct"] * df["Manu_SKU_Count"],  # >1 = above average
        0
    )

    # 2. Category rank: top 30% by LTDS within category
    df["Cat_LTDS_Rank_Pct"] = df.groupby("SKU Category")["LTDS_Amount"].rank(
        pct=True, ascending=False
    )

    # SKU is significant if above-average in manufacturer OR top 30% in category
    df["Is_Significant"] = (df["Contrib_Ratio"] >= 1.0) | (df["Cat_LTDS_Rank_Pct"] <= 0.30)

    # Primary Demand_Value = Gap Demand Value (capped at target, significant SKUs only)
    df["Demand_Value"] = np.where(df["Is_Significant"], df["Gap_Demand_Value"], 0)

    return df

df_raw = load_data()


# ─── Sidebar Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.markdown("---")

    bus = ["All"] + sorted(df_raw["BusinessUnitName"].dropna().unique().tolist())
    sel_bu = st.selectbox("Business Unit", bus)

    cats = ["All"] + sorted(df_raw["SKU Category"].dropna().unique().tolist())
    sel_cat = st.selectbox("SKU Category", cats)

    manus = ["All"] + sorted(df_raw["ManufacturerName"].dropna().unique().tolist())
    sel_manu = st.selectbox("Manufacturer", manus)

    types = ["All"] + sorted(df_raw["type"].dropna().unique().tolist())
    sel_type = st.selectbox("SKU Type", types)

    sku_focus = st.selectbox("SKU Focus", ["All SKUs", "Core (Top SKUs)", "Non-Core"])
    oos_filter = st.checkbox("Show OOS SKUs only", value=False)

    st.markdown("---")
    st.markdown("### 💡 Quick Insights")
    total_skus = len(df_raw)
    oos_count = df_raw["IsOOS"].sum()
    st.info(f"**{oos_count}** of **{total_skus}** SKUs are OOS ({oos_count/total_skus*100:.1f}%)")

# Apply filters
df = df_raw.copy()
if sel_bu != "All":      df = df[df["BusinessUnitName"] == sel_bu]
if sel_cat != "All":     df = df[df["SKU Category"] == sel_cat]
if sel_manu != "All":    df = df[df["ManufacturerName"] == sel_manu]
if sel_type != "All":    df = df[df["type"] == sel_type]
if sku_focus == "Core (Top SKUs)":  df = df[df["TopSKU"] == "Core"]
if sku_focus == "Non-Core":         df = df[df["TopSKU"] == "Non-Core"]
if oos_filter:           df = df[df["IsOOS"]]


# ─── Header Overlay (Native Bar) ─────────────────────────────────────────────
st.markdown("""
<div class="native-header-overlay">
    <div style="display:flex; align-items:center; gap:12px; text-align:center;">
        <span style="font-size:1.3rem;">📦</span>
        <div style="display: flex; flex-direction: column; gap: 0px;">
            <h1 style="color:#e2e8f0; margin:0; font-size:1.1rem; font-weight:700; line-height:1.1;">
                Procurement Intelligence Dashboard
                <span id="active-tab-name" style="font-weight:800; font-size:1.35rem; color:#818cf8; margin-left:4px;"></span>
            </h1>
            <p style="color:#94a3b8; margin:-2px 0 0 0; font-size:0.65rem; letter-spacing:0.3px; line-height:1;">
                Advanced Analytics · Inventory Health · Supplier Performance
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── JavaScript Integration (Header Sync) ──────────────────────────────────
components.html(
    """
    <script>
    const updateTabName = () => {
        const doc = window.parent.document;
        // Streamlit uses 'button' with aria-selected="true" for tabs
        const activeTab = doc.querySelector('button[aria-selected="true"] p');
        const target = doc.querySelector('#active-tab-name');
        if (activeTab && target) {
            const newText = " — " + activeTab.innerText;
            if (target.innerText !== newText) {
                target.innerText = newText;
            }
        }
    };
    // Periodic check to handle Streamlit re-renders and lazy-loaded items
    setInterval(updateTabName, 500);
    </script>
    """,
    height=0
)

# Filter info
if df.shape[0] < df_raw.shape[0]:
    st.markdown(f"<p style='color:#94a3b8;font-size:0.85rem;'>Showing <b style='color:#6366f1'>{len(df):,}</b> of {len(df_raw):,} SKUs</p>", unsafe_allow_html=True)


# ─── KPI Cards ───────────────────────────────────────────────────────────────
total_buy       = df["Total buying Amount"].sum()
live_buy        = df["Live Total buying Amount"].sum()
total_mtd       = df["MTD"].sum()
total_tgt       = df["SKU_TGT"].sum()
tgt_ach         = (total_mtd / total_tgt * 100) if total_tgt > 0 else 0
oos_pct         = df["IsOOS"].mean() * 100
avg_cov         = df["Inventory_Coverage"].replace([np.inf], np.nan).median()
gap_demand_val  = df["Gap_Demand_Value"].sum()      # ₨ needed to fill all stock gaps
aws_demand_val  = df["AWS_Demand_Value"].sum()      # ₨ to stock everything to full AWS
gap_skus        = (df["Gap_Units"] > 0).sum()       # SKUs that need replenishment

def fmt_num(n):
    abs_n = abs(n)
    if abs_n >= 1e6: return f"₨{n/1e6:.2f}M"
    if abs_n >= 1e3: return f"₨{n/1e3:.1f}K"
    return f"₨{n:.0f}"

# (KPI calculations stay outside the tab to ensure data is available for other sections if needed)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview", "🏭 Supplier Analysis", "📦 Inventory Health",
    "🎯 Target Tracking", "🔬 SKU Deep Dive", "⚡ Alerts & Insights",
    "📋 SKU Level Detail", "🔮 Zero SKU Demand Forecast", "💼 Working Capital",
    "🔠 ABC Analysis", "⚙️ EOQ", "🛡️ Safety Stock"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # ─── Top KPI Cards ───────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
    k_cards = [
        (kc1, "💰", "Total Purchase Value",   fmt_num(total_buy),       "", ""),
        (kc2, "🟢", "Live Inventory Value",   fmt_num(live_buy),         "", ""),
        (kc3, "🎯", "MTD Sales",              fmt_num(total_mtd),        f"{tgt_ach:.1f}% of Target", tgt_ach >= 80),
        (kc4, "🛒", "Gap Demand Value",       fmt_num(gap_demand_val),   f"{gap_skus} SKUs need restock", gap_demand_val < aws_demand_val * 0.3),
        (kc5, "⚠️", "OOS Rate",              f"{oos_pct:.1f}%",         f"{df['IsOOS'].sum()} SKUs", oos_pct < 15),
        (kc6, "📦", "Avg Inventory Cover",   f"{avg_cov:.1f}x",         "vs AWS", avg_cov >= 1),
    ]
    for col, icon, label, val, delta, pos in k_cards:
        delta_html = ""
        if delta:
            cls = "metric-delta-pos" if pos else "metric-delta-neg"
            delta_html = f'<div class="{cls}">{delta}</div>'
        col.markdown(f"""
        <div class="metric-card">
          <div style="font-size:1.6rem; min-width:30px; display:flex; justify-content:center;">{icon}</div>
          <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            {delta_html}
          </div>
        </div>""", unsafe_allow_html=True)
    
    # ─── Main Overview Row (Revenue BU) ──────────────────────────────────────
    st.markdown('<div class="section-header">Revenue & Category Performance Mix</div>', unsafe_allow_html=True)
    r1_c1, r1_c2, r1_c3 = st.columns([2.5, 1.5, 2])

    with r1_c1:
        bu_grp = df.groupby("BusinessUnitName").agg(
            PurchaseValue=("Total buying Amount","sum"),
            LiveValue=("Live Total buying Amount","sum")
        ).reset_index().sort_values("PurchaseValue", ascending=False)
        fig = go.Figure()
        fig.add_bar(x=bu_grp["BusinessUnitName"], y=bu_grp["PurchaseValue"], name="Total Purchase", marker_color="#6366f1")
        fig.add_bar(x=bu_grp["BusinessUnitName"], y=bu_grp["LiveValue"], name="Live Inventory", marker_color="#22d3ee")
        fig.update_layout(barmode="group", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=240, legend=dict(orientation="h", y=1.1, font_size=10), margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with r1_c2:
        fig2 = px.pie(bu_grp, values="PurchaseValue", names="BusinessUnitName", hole=0.55,
                      color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig2.update_traces(textinfo="percent", textfont_size=10)
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=240, showlegend=False, 
                           margin=dict(l=0,r=0,t=40,b=10), title="BU Share", title_font_size=12)
        st.plotly_chart(fig2, use_container_width=True)

    with r1_c3:
        cat_grp = df.groupby("SKU Category").agg(TotalBuy=("Total buying Amount","sum")).reset_index().sort_values("TotalBuy").tail(10)
        fig3 = go.Figure(go.Bar(y=cat_grp["SKU Category"], x=cat_grp["TotalBuy"], orientation="h", marker_color="#818cf8",
                                text=[f"₨{v/1e6:.1f}M" for v in cat_grp["TotalBuy"]], textposition="outside"))
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                           height=240, margin=dict(l=0,r=50,t=40,b=0), title="Top Categories", title_font_size=12)
        st.plotly_chart(fig3, use_container_width=True)

    # ─── Bottom Insights Row (Treemap & SKU Detail) ──────────────────────────
    st.markdown('<div class="section-header">Distribution & SKU Type Deep Dive</div>', unsafe_allow_html=True)
    r2_c1, r2_c2 = st.columns([1.5, 1])

    with r2_c1:
        # Treemap
        tree_df = df.groupby(["BusinessUnitName","SKU Category"]).agg(Value=("Total buying Amount","sum")).reset_index()
        fig4 = px.treemap(tree_df, path=["BusinessUnitName","SKU Category"], values="Value",
                          color="Value", color_continuous_scale="Viridis")
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig4, use_container_width=True)

    with r2_c2:
        # Main Column for SKU Detail (Donut + Bars)
        sc1, sc2 = st.columns([1.1, 1])
        with sc1:
            type_vals = df["type"].value_counts().reset_index()
            type_vals.columns = ["Type","Count"]
            colors = {"Alpha":"#4ade80","Bravo":"#60a5fa","Charlie":"#f59e0b","Unclassified":"#94a3b8"}
            fig5 = px.pie(type_vals, values="Count", names="Type", hole=0.5, color="Type", color_discrete_map=colors)
            fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, 
                               showlegend=True, legend=dict(font_size=9, x=0, y=-0.2, orientation="h"),
                               margin=dict(l=0,r=0,t=40,b=0), title="SKU Type Mix", title_font_size=12)
            fig5.update_traces(textinfo="percent")
            st.plotly_chart(fig5, use_container_width=True)
        
        with sc2:
            # Stacked vertical column: Core vs Non-Core + Stock Status
            top_vals = df["TopSKU"].value_counts().reset_index()
            top_vals.columns = ["SKU_Focus","Count"]
            fig6 = px.bar(top_vals, y="SKU_Focus", x="Count", color="SKU_Focus", 
                          color_discrete_sequence=["#6366f1","#94a3b8"], text="Count", orientation="h")
            fig6.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                               height=140, showlegend=False, margin=dict(l=0,r=30,t=35,b=0), 
                               title="Core Focus", title_font_size=11, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig6, use_container_width=True)

            ss = df["Stock_Status"].value_counts().reset_index()
            ss.columns = ["Status","Count"]
            ss_colors = {"Critical":"#ef4444","Low":"#f59e0b","Healthy":"#4ade80","Overstocked":"#60a5fa"}
            fig7 = px.bar(ss, y="Status", x="Count", color="Status", color_discrete_map=ss_colors, text="Count", orientation="h")
            fig7.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                               height=180, showlegend=False, margin=dict(l=0,r=30,t=35,b=0), 
                               title="Stock Status", title_font_size=11, xaxis_title="", yaxis_title="",
                               yaxis=dict(tickfont_size=9))
            st.plotly_chart(fig7, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – SUPPLIER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Supplier Performance Matrix</div>', unsafe_allow_html=True)

    sup = df.groupby("ManufacturerName").agg(
        TotalBuy=("Total buying Amount","sum"),
        LiveValue=("Live Total buying Amount","sum"),
        SKUs=("SKU Code","count"),
        OOS_SKUs=("IsOOS","sum"),
        AvgCoverage=("Inventory_Coverage","mean"),
        OdrValue=("OdrValue","sum"),
        MTD=("MTD","sum"),
        TGT=("SKU_TGT","sum"),
    ).reset_index()
    sup["OOS_Rate"] = (sup["OOS_SKUs"] / sup["SKUs"] * 100).round(1)
    sup["TGT_Ach"] = np.where(sup["TGT"]>0, (sup["MTD"]/sup["TGT"]*100).clip(0,200), 0)
    sup = sup.sort_values("TotalBuy", ascending=False)

    # ─── Row 1: Performance Matrix (3 Columns) ───────────────────────────────
    r1_c1, r1_c2, r1_c3 = st.columns(3)

    with r1_c1:
        top_sup = sup.head(15)
        fig = go.Figure(go.Bar(
            x=top_sup["ManufacturerName"], y=top_sup["TotalBuy"],
            marker_color="#6366f1",
            text=[f"₨{v/1e6:.1f}M" for v in top_sup["TotalBuy"]], textposition="outside"
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300,
                          title="Top 15 Suppliers by Purchase Value",
                          margin=dict(l=0,r=0,t=40,b=80),
                          xaxis=dict(tickangle=-45, tickfont_size=9),
                          yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with r1_c2:
        fig2 = px.scatter(
            sup, x="TotalBuy", y="OOS_Rate",
            size="SKUs", color="TGT_Ach",
            hover_name="ManufacturerName",
            hover_data={"SKUs":True,"OOS_SKUs":True,"TGT_Ach":":.1f"},
            color_continuous_scale="RdYlGn",
            labels={"TotalBuy":"Value","OOS_Rate":"OOS %","TGT_Ach":"Target %"}
        )
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=300,
                           title="Supplier Risk Matrix (size=# SKUs, color=Target Achievement)",
                           margin=dict(l=0,r=0,t=40,b=40),
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with r1_c3:
        brand_grp = df.groupby("BrandName").agg(
            Value=("Total buying Amount","sum")
        ).reset_index().sort_values("Value", ascending=False).head(15)
        fig4 = px.bar(brand_grp, x="BrandName", y="Value",
                      color="Value", color_continuous_scale="Plasma",
                      text=[f"₨{v/1e6:.1f}M" for v in brand_grp["Value"]])
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=300,
                           title="Top 15 Brands by Value",
                           xaxis=dict(tickangle=-45, tickfont_size=9), showlegend=False,
                           margin=dict(l=0,r=0,t=40,b=80), coloraxis_showscale=False)
        fig4.update_traces(textposition="outside")
        st.plotly_chart(fig4, use_container_width=True)

    # ─── Row 2: Concentration & Scorecard ────────────────────────────────────
    st.markdown('<div class="section-header">Supplier Concentration & Scorecard</div>', unsafe_allow_html=True)
    r2_c1, r2_c2 = st.columns([1, 2.2])

    with r2_c1:
        top10_funnel = sup.head(10).sort_values("TotalBuy").copy()
        top10_funnel["Value_M"] = top10_funnel["TotalBuy"] / 1e6
        fig3 = px.funnel(top10_funnel, x="Value_M", y="ManufacturerName",
                         color_discrete_sequence=["#6366f1"])
        fig3.update_traces(texttemplate="%{x:.1f}M") 
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           height=320, title="Top 10 Supplier Funnel (₨ Millions)",
                           margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with r2_c2:
        scorecard = sup[["ManufacturerName","SKUs","TotalBuy","OOS_Rate","TGT_Ach","AvgCoverage"]].head(100).copy()
        scorecard.columns = ["Manufacturer","# SKUs","Purchase Val","OOS %","TGT %","Cov"]
        
        # Numeric values for styling calculation
        scorecard_styled = scorecard.style.format({
            "Purchase Val": "₨{:,.0f}",
            "OOS %": "{:.1f}%",
            "TGT %": "{:.1f}%",
            "Cov": "{:.2f}x"
        })
        st.dataframe(scorecard_styled, use_container_width=True, height=320)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – INVENTORY HEALTH
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    # ─── Row 1: Coverage & Stock Analysis (3 Columns) ────────────────────────
    r1_c1, r1_c2, r1_c3 = st.columns(3)

    with r1_c1:
        cov_df = df[df["Inventory_Coverage"] < 20]
        fig = px.histogram(cov_df, x="Inventory_Coverage", nbins=30,
                           color_discrete_sequence=["#6366f1"],
                           labels={"Inventory_Coverage":"Coverage (weeks)"})
        fig.add_vline(x=1, line_dash="dash", line_color="#ef4444")
        fig.add_vline(x=3, line_dash="dash", line_color="#f59e0b")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          title="Inventory Coverage Distribution", margin=dict(l=0,r=0,t=40,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with r1_c2:
        ss_cat = df.groupby(["SKU Category","Stock_Status"]).size().reset_index(name="Count")
        fig2 = px.bar(ss_cat, x="SKU Category", y="Count", color="Stock_Status",
                      color_discrete_map={"Critical":"#ef4444","Low":"#f59e0b","Healthy":"#4ade80","Overstocked":"#60a5fa"},
                      barmode="stack")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=280,
                           title="Stock Status by Category", xaxis=dict(tickangle=-45, tickfont_size=9),
                           margin=dict(l=0,r=0,t=40,b=60), showlegend=True,
                           legend=dict(orientation="h", y=1.1, font_size=8, title=dict(text="")))
        st.plotly_chart(fig2, use_container_width=True)

    with r1_c3:
        cat_inv = df.groupby("SKU Category").agg(
            Available=("Available Total Units","sum"),
            MSL=("MSL in Units","sum"),
            AWS=("AWS in Units","sum"),
        ).reset_index().sort_values("Available", ascending=False).head(12)
        fig3 = go.Figure()
        fig3.add_bar(x=cat_inv["SKU Category"], y=cat_inv["Available"], name="Avail", marker_color="#4ade80")
        fig3.add_bar(x=cat_inv["SKU Category"], y=cat_inv["MSL"], name="MSL", marker_color="#f59e0b")
        fig3.add_bar(x=cat_inv["SKU Category"], y=cat_inv["AWS"], name="AWS", marker_color="#6366f1")
        fig3.update_layout(barmode="group", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=280,
                           title="Available vs MSL vs AWS Analysis", xaxis=dict(tickangle=-45, tickfont_size=9),
                           legend=dict(orientation="h", y=1.1, font_size=8), margin=dict(l=0,r=0,t=40,b=60))
        st.plotly_chart(fig3, use_container_width=True)

    # ─── Row 2: Value & OOS Analysis (3 Columns) ─────────────────────────────
    st.markdown('<div class="section-header">Inventory Value & OOS Deep Dive</div>', unsafe_allow_html=True)
    r2_c1, r2_c2, r2_c3 = st.columns(3)

    with r2_c1:
        # Live vs Total scatter
        fig4 = px.scatter(df[df["Total buying Amount"]>0].sample(min(400,len(df))),
                          x="Total buying Amount", y="Live Total buying Amount",
                          color="BusinessUnitName", size="Available Total Units",
                          opacity=0.6)
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=280,
                           title="Live vs Total Inventory Value per SKU", showlegend=False,
                           margin=dict(l=0,r=0,t=40,b=40))
        st.plotly_chart(fig4, use_container_width=True)

    with r2_c2:
        oos_cat = df.groupby("SKU Category").agg(Total=("SKU Code","count"), OOS=("IsOOS","sum")).reset_index()
        oos_cat["OOS_Rate"] = (oos_cat["OOS"] / oos_cat["Total"] * 100).round(1)
        oos_cat = oos_cat.sort_values("OOS_Rate", ascending=False).head(12)
        fig5 = px.bar(oos_cat, x="SKU Category", y="OOS_Rate", color="OOS_Rate", color_continuous_scale="Reds", text="OOS_Rate")
        fig5.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=280,
                           title="OOS Rate by Category (%)", xaxis=dict(tickangle=-45, tickfont_size=9),
                           coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=60))
        st.plotly_chart(fig5, use_container_width=True)

    with r2_c3:
        oos_bu = df.groupby("BusinessUnitName").agg(Total=("SKU Code","count"), OOS=("IsOOS","sum")).reset_index()
        oos_bu["OOS_Rate"] = (oos_bu["OOS"] / oos_bu["Total"] * 100).round(1)
        fig6 = go.Figure()
        fig6.add_bar(x=oos_bu["BusinessUnitName"], y=oos_bu["OOS"], name="OOS Count", marker_color="#ef4444")
        fig6.add_scatter(x=oos_bu["BusinessUnitName"], y=oos_bu["OOS_Rate"], mode="lines+markers", name="OOS %", yaxis="y2",
                         marker=dict(color="#f59e0b", size=6))
        fig6.update_layout(
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, title="OOS by Business Unit", margin=dict(l=0,r=0,t=40,b=40),
            legend=dict(orientation="h", y=1.1, font_size=8), xaxis=dict(tickangle=-45, tickfont_size=9)
        )
        st.plotly_chart(fig6, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – TARGET TRACKING
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">MTD vs Target Achievement</div>', unsafe_allow_html=True)

    tgt_bu = df.groupby("BusinessUnitName").agg(
        MTD=("MTD","sum"), TGT=("SKU_TGT","sum"), Sale=("Sale","sum")
    ).reset_index()
    tgt_bu["Ach_Pct"] = np.where(tgt_bu["TGT"]>0, tgt_bu["MTD"]/tgt_bu["TGT"]*100, 0)
    tgt_bu["Gap"] = tgt_bu["TGT"] - tgt_bu["MTD"]

    # ─── Row 1: BU Target Performance (3 Columns) ────────────────────────────
    r1_c1, r1_c2, r1_c3 = st.columns(3)

    with r1_c1:
        fig = go.Figure()
        fig.add_bar(x=tgt_bu["BusinessUnitName"], y=tgt_bu["TGT"], name="Target",
                    marker_color="rgba(100,100,180,0.4)", marker_line_color="#6366f1", marker_line_width=1.5)
        fig.add_bar(x=tgt_bu["BusinessUnitName"], y=tgt_bu["MTD"], name="Actual", marker_color="#4ade80")
        fig.update_layout(barmode="overlay", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=280,
                          title="MTD vs Target by Business Unit", 
                          legend=dict(orientation="h", y=1.1, font_size=8), 
                          xaxis=dict(tickangle=-45, tickfont_size=9), margin=dict(l=0,r=0,t=40,b=40))
        st.plotly_chart(fig, use_container_width=True)

    with r1_c2:
        fig2 = px.bar(tgt_bu.sort_values("Ach_Pct"), x="Ach_Pct", y="BusinessUnitName",
                      orientation="h", color="Ach_Pct",
                      color_continuous_scale=["#ef4444","#f59e0b","#4ade80"],
                      range_color=[0,100], text="Ach_Pct")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.add_vline(x=80, line_dash="dash", line_color="white")
        fig2.add_vline(x=100, line_dash="dash", line_color="#4ade80")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=280,
                           title="Target Achievement % by Business Unit",
                           coloraxis_showscale=False, margin=dict(l=0,r=40,t=40,b=20),
                           yaxis=dict(tickfont_size=9))
        st.plotly_chart(fig2, use_container_width=True)

    with r1_c3:
        bins = pd.cut(df["TGT_Ach_Pct"], bins=[0,50,80,100,120,500],
                      labels=["<50%","50-80%","80-100%","100-120%",">120%"])
        bin_counts = bins.value_counts().sort_index().reset_index()
        bin_counts.columns = ["Range","Count"]
        colors = ["#ef4444","#f97316","#f59e0b","#4ade80","#22d3ee"]
        fig4 = px.pie(bin_counts, values="Count", names="Range", hole=0.5,
                      color_discrete_sequence=colors,
                      title="SKU Achievement Distribution")
        fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           height=280, margin=dict(l=0,r=0,t=40,b=0),
                           legend=dict(orientation="h", y=-0.1, font_size=8))
        fig4.update_traces(textinfo="percent")
        st.plotly_chart(fig4, use_container_width=True)

    # ─── Row 2: SKU Target Deep Dive (Stacked) ──────────────────────────────
    st.markdown('<div class="section-header">Target Compliance & Gap Analysis</div>', unsafe_allow_html=True)
    r2_c1, r2_c2 = st.columns([1, 2.2])

    with r2_c1:
        sku_tgt = df[df["SKU_TGT"]>0].copy()
        sku_tgt["Ach"] = (sku_tgt["MTD"] / sku_tgt["SKU_TGT"] * 100).clip(0, 200)
        fig3 = px.scatter(
            sku_tgt.sample(min(300, len(sku_tgt))),
            x="SKU_TGT", y="MTD", color="Ach",
            color_continuous_scale="RdYlGn", range_color=[0,150],
            size="SKU_TGT", opacity=0.6
        )
        fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                           title="SKU Target vs MTD Achievement",
                           margin=dict(l=0,r=0,t=40,b=40), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with r2_c2:
        gap_df = df[df["SKU_TGT"]>0].copy()
        gap_df["Gap"] = gap_df["SKU_TGT"] - gap_df["MTD"]
        gap_df["Ach_Pct"] = (gap_df["MTD"]/gap_df["SKU_TGT"]*100).round(1)
        gap_df = gap_df.sort_values("Gap", ascending=False).head(15)
        fig5 = px.bar(gap_df, y="SKU Description", x="Gap", orientation="h",
                      color="Ach_Pct", color_continuous_scale="Reds_r",
                      text=[f"₨{v/1e3:.0f}K" for v in gap_df["Gap"]])
        fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                           title="Top 15 SKUs by Target Gap (₨)",
                           margin=dict(l=0,r=50,t=40,b=0), coloraxis_showscale=False,
                           yaxis=dict(tickfont_size=9))
        st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 – SKU DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">SKU Explorer</div>', unsafe_allow_html=True)

    col_x = st.selectbox("X Axis", ["Total buying Amount","Sale","MTD","SKU_TGT","OdrValue","MRP","LTDS_Amount"], key="x_ax")
    col_y = st.selectbox("Y Axis", ["Live Total buying Amount","OdrQty","MTD","TGT_Ach_Pct","Inventory_Coverage"], key="y_ax")
    color_by = st.selectbox("Color By", ["BusinessUnitName","SKU Category","type","TopSKU","Stock_Status"], key="col_by")

    plot_df = df[(df[col_x]>0) & (df[col_y]>0)].copy()
    fig = px.scatter(plot_df.sample(min(600,len(plot_df))),
                     x=col_x, y=col_y, color=color_by,
                     hover_name="SKU Description",
                     hover_data={"ManufacturerName":True,"BrandName":True,col_x:True,col_y:True},
                     opacity=0.75, height=450,
                     title=f"{col_y} vs {col_x}")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", title_font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">SKU Pareto (80/20)</div>', unsafe_allow_html=True)
    pareto_df = df.groupby("SKU Description")["Total buying Amount"].sum().sort_values(ascending=False).reset_index()
    pareto_df["Cumulative"] = pareto_df["Total buying Amount"].cumsum()
    pareto_df["Cumulative_Pct"] = pareto_df["Cumulative"] / pareto_df["Total buying Amount"].sum() * 100
    pareto_df["SKU_Rank"] = range(1, len(pareto_df)+1)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_bar(x=pareto_df["SKU_Rank"].head(100), y=pareto_df["Total buying Amount"].head(100),
                 name="Purchase Value", marker_color="#6366f1")
    fig2.add_scatter(x=pareto_df["SKU_Rank"].head(100), y=pareto_df["Cumulative_Pct"].head(100),
                     mode="lines", name="Cumulative %", line=dict(color="#f59e0b",width=2), secondary_y=True)
    fig2.add_hline(y=80, line_dash="dash", line_color="white", secondary_y=True,
                   annotation_text="80% threshold")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=380,
                       title="Pareto Analysis – Top 100 SKUs", title_font_color="#e2e8f0",
                       legend=dict(orientation="h",y=1.1))
    fig2.update_yaxes(title_text="Purchase Value (₨)", secondary_y=False)
    fig2.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Full SKU Data Table</div>', unsafe_allow_html=True)
    cols_show = ["SKU Description","ManufacturerName","BrandName","SKU Category",
                 "BusinessUnitName","type","TopSKU","Total buying Amount",
                 "Live Total buying Amount","Available Total Units","MSL in Units",
                 "AWS in Units","Stock_Status","IsOOS","MTD","TGT","TGT_Ach_Pct"]
    table_df = df[cols_show].copy()
    table_df["Total buying Amount"] = table_df["Total buying Amount"].apply(lambda x: f"₨{x:,.0f}")
    table_df["Live Total buying Amount"] = table_df["Live Total buying Amount"].apply(lambda x: f"₨{x:,.0f}")
    table_df["TGT_Ach_Pct"] = table_df["TGT_Ach_Pct"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(table_df, use_container_width=True, height=400)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 – ALERTS & INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🚨 Automated Alerts</div>', unsafe_allow_html=True)

    # Critical inventory
    critical_inv = df[df["Stock_Status"]=="Critical"]
    # High OOS manufacturers
    oos_sup = df.groupby("ManufacturerName")["IsOOS"].agg(["sum","count"]).reset_index()
    oos_sup.columns = ["Manufacturer","OOS","Total"]
    oos_sup["Rate"] = oos_sup["OOS"]/oos_sup["Total"]*100
    high_oos = oos_sup[oos_sup["Rate"]>50].sort_values("Rate",ascending=False)
    # Target underachievers
    under = df[(df["SKU_TGT"]>0) & (df["TGT_Ach_Pct"]<50)]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="alert-box">
        ⚠️ <b>{len(critical_inv)} SKUs</b> in Critical stock status (coverage &lt; 0.5x AWS).
        Immediate replenishment required.
        </div>""", unsafe_allow_html=True)
        for _, row in critical_inv.head(5).iterrows():
            st.markdown(f"• `{row['SKU Description'][:40]}` — {row['ManufacturerName']}")

    with c2:
        st.markdown(f"""<div class="alert-box">
        🔴 <b>{len(high_oos)} suppliers</b> have OOS rate &gt;50%.
        Review supply chain partnerships.
        </div>""", unsafe_allow_html=True)
        for _, row in high_oos.head(5).iterrows():
            st.markdown(f"• `{row['Manufacturer']}` — {row['Rate']:.0f}% OOS")

    with c3:
        st.markdown(f"""<div class="alert-box">
        📉 <b>{len(under)} SKUs</b> below 50% target achievement.
        Urgent action needed.
        </div>""", unsafe_allow_html=True)
        for _, row in under.head(5).iterrows():
            st.markdown(f"• `{row['SKU Description'][:35]}` — {row['TGT_Ach_Pct']:.0f}%")

    st.markdown('<div class="section-header">💡 Strategic Insights</div>', unsafe_allow_html=True)

    # Top manu
    top_manu = df.groupby("ManufacturerName")["Total buying Amount"].sum().idxmax()
    top_manu_val = df.groupby("ManufacturerName")["Total buying Amount"].sum().max()
    top_cat = df.groupby("SKU Category")["Total buying Amount"].sum().idxmax()
    conc_pct = df.groupby("ManufacturerName")["Total buying Amount"].sum().nlargest(5).sum() / df["Total buying Amount"].sum() * 100
    healthy_pct = (df["Stock_Status"]=="Healthy").mean()*100
    core_ach = df[df["TopSKU"]=="Core"]["TGT_Ach_Pct"].mean()

    insights = [
        (f"Top supplier **{top_manu}** contributes ₨{top_manu_val/1e6:.2f}M — "
         f"top 5 suppliers drive **{conc_pct:.1f}%** of total purchase value. "
         "Consider diversifying to reduce concentration risk."),
        (f"**{healthy_pct:.1f}%** of SKUs have healthy inventory coverage. "
         f"**{(df['Stock_Status']=='Critical').mean()*100:.1f}%** are in critical status — "
         "prioritize restocking critical items before OOS materializes."),
        (f"Core SKUs (Top SKUs) average **{core_ach:.1f}%** target achievement. "
         "Focus sales push on core portfolio as it drives the highest value."),
        (f"**{top_cat}** is the highest-spend category. "
         f"OOS rate in this category is "
         f"**{df[df['SKU Category']==top_cat]['IsOOS'].mean()*100:.1f}%** — "
         "monitor closely for stock-outs."),
        (f"**{df[df['type']=='Alpha'].shape[0]} Alpha SKUs** represent premium portfolio. "
         f"Alpha type average target achievement: "
         f"**{df[df['type']=='Alpha']['TGT_Ach_Pct'].mean():.1f}%**.")
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-box">💡 {ins}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Advanced Analytics — Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["Total buying Amount","Live Total buying Amount","Available Total Units",
                "MSL in Units","AWS in Units","Sale","MTD","SKU_TGT","OdrValue","MRP",
                "Inventory_Coverage","TGT_Ach_Pct","LTDS_Amount"]
    corr = df[num_cols].replace([np.inf,-np.inf], np.nan).dropna().corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    aspect="auto", title="Feature Correlation Matrix",
                    zmin=-1, zmax=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      height=520, title_font_color="#e2e8f0",
                      margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">📊 Sub-Category Performance</div>', unsafe_allow_html=True)
    sub_df = df.groupby("SubCategoryName").agg(
        Value=("Total buying Amount","sum"),
        OOS=("IsOOS","sum"),
        SKUs=("SKU Code","count"),
        MTD=("MTD","sum"),
        TGT=("SKU_TGT","sum"),
    ).reset_index()
    sub_df["OOS_Rate"] = (sub_df["OOS"]/sub_df["SKUs"]*100).round(1)
    sub_df["Ach"] = np.where(sub_df["TGT"]>0, sub_df["MTD"]/sub_df["TGT"]*100, 0)
    sub_df = sub_df.sort_values("Value", ascending=False).head(25)

    fig2 = px.scatter(sub_df, x="Value", y="OOS_Rate", size="SKUs",
                      color="Ach", hover_name="SubCategoryName",
                      color_continuous_scale="RdYlGn", range_color=[0,120],
                      title="Sub-Category: Purchase Value vs OOS Rate (size=# SKUs, color=Target Ach%)",
                      labels={"Value":"Purchase Value","OOS_Rate":"OOS Rate %","Ach":"Target %"})
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=420,
                       title_font_color="#e2e8f0")
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 – SKU LEVEL DETAIL  (mirrors the grid view in the screenshot)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">SKU Level Detail — Full Grid View</div>', unsafe_allow_html=True)

    # ── Service Level KPI bar ────────────────────────────────────────────────
    total_skus_all   = len(df)
    short_skus       = df["IsOOS"].sum()
    svc_level        = round((1 - short_skus / total_skus_all) * 100) if total_skus_all else 0
    short_pct        = round(short_skus / total_skus_all * 100) if total_skus_all else 0

    s1, s2, s3, s4, s5 = st.columns(5)
    def mini_card(col, label, val, color="#6366f1"):
        col.markdown(f"""
        <div style="background:#1e2130;border:1px solid #2d3347;border-radius:10px;
                    padding:14px 16px;text-align:center;">
          <div style="font-size:1.5rem;font-weight:700;color:{color};">{val}</div>
          <div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;
                      letter-spacing:1px;margin-top:4px;">{label}</div>
        </div>""", unsafe_allow_html=True)

    mini_card(s1, "Total SKUs",   f"{total_skus_all:,}",   "#e2e8f0")
    mini_card(s2, "Short SKUs",   f"{int(short_skus):,}",   "#f87171")
    mini_card(s3, "Short %",      f"{short_pct}%",          "#f59e0b")
    mini_card(s4, "Service Level",f"{svc_level}%",
              "#4ade80" if svc_level >= 80 else "#f87171")
    mini_card(s5, "Live Stock Val",
              f"₨{df['Live Total buying Amount'].sum()/1e6:.2f}M", "#22d3ee")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Demand Value KPI row ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Demand Value Summary</div>', unsafe_allow_html=True)
    dv1, dv2, dv3, dv4, dv5 = st.columns(5)
    _gap_val   = df["Gap_Demand_Value"].sum()
    _aws_val   = df["AWS_Demand_Value"].sum()
    _gap_skus  = (df["Gap_Units"] > 0).sum()
    _avg_ucost = df[df["Unit_Cost"]>0]["Unit_Cost"].mean()
    _cov_gap   = df[df["Gap_Units"]>0]["Gap_Units"].sum()

    mini_card(dv1, "Gap Demand Value",
              fmt_num(_gap_val),
              "#f87171" if _gap_val > live_buy * 0.3 else "#f59e0b")
    mini_card(dv2, "Full AWS Demand Value",
              fmt_num(_aws_val), "#60a5fa")
    mini_card(dv3, "SKUs Needing Restock",
              f"{_gap_skus:,}", "#f59e0b")
    mini_card(dv4, "Avg Unit Cost",
              f"₨{_avg_ucost:,.0f}", "#a78bfa")
    mini_card(dv5, "Total Gap Units",
              f"{_cov_gap:,.0f} pcs", "#4ade80")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters row ──────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    manu_opts = ["All"] + sorted(df["ManufacturerName"].dropna().unique().tolist())
    cat_opts  = ["All"] + sorted(df["SKU Category"].dropna().unique().tolist())
    type_opts = ["All"] + sorted(df["type"].dropna().unique().tolist())

    f_manu  = fc1.selectbox("Manufacturer",  manu_opts, key="sd_manu")
    f_cat   = fc2.selectbox("Category",      cat_opts,  key="sd_cat")
    f_type  = fc3.selectbox("SKU Type",      type_opts, key="sd_type")
    f_oos   = fc4.selectbox("OOS / In-Stock",["All","OOS Only","In-Stock Only"], key="sd_oos")
    f_core  = fc5.selectbox("Core / Non-Core",["All","Core","Non-Core"], key="sd_core")

    sku_df = df.copy()
    if f_manu != "All":  sku_df = sku_df[sku_df["ManufacturerName"] == f_manu]
    if f_cat  != "All":  sku_df = sku_df[sku_df["SKU Category"]     == f_cat]
    if f_type != "All":  sku_df = sku_df[sku_df["type"]             == f_type]
    if f_oos  == "OOS Only":      sku_df = sku_df[sku_df["IsOOS"]]
    if f_oos  == "In-Stock Only": sku_df = sku_df[~sku_df["IsOOS"]]
    if f_core == "Core":     sku_df = sku_df[sku_df["TopSKU"] == "Core"]
    if f_core == "Non-Core": sku_df = sku_df[sku_df["TopSKU"] == "Non-Core"]

    # search box
    search = st.text_input("🔍  Search SKU / Brand / Description", key="sd_search")
    if search:
        mask = (
            sku_df["SKU Description"].str.contains(search, case=False, na=False) |
            sku_df["BrandName"].str.contains(search, case=False, na=False) |
            sku_df["ManufacturerName"].str.contains(search, case=False, na=False)
        )
        sku_df = sku_df[mask]

    st.markdown(f"<p style='color:#94a3b8;font-size:0.82rem;'>"
                f"Showing <b style='color:#6366f1'>{len(sku_df):,}</b> SKUs</p>",
                unsafe_allow_html=True)

    # ── Manufacturer-level summary (collapsible group header row) ────────────
    st.markdown('<div class="section-header">Manufacturer Summary (Group Level)</div>',
                unsafe_allow_html=True)

    def safe_mode(x):
        try:
            m = x.dropna()
            return m.mode().iloc[0] if not m.empty else "-"
        except Exception:
            return "-"

    def safe_first(x):
        try:
            m = x.dropna()
            return m.iloc[0] if not m.empty else "-"
        except Exception:
            return "-"

    manu_summary = sku_df.groupby("ManufacturerName").agg(
        T_SKU     =("SKU Code","count"),
        Shrt_SKU  =("IsOOS","sum"),
        OOS_Days  =("OOS","mean"),
        Allow_Pri =("AllowPri",  safe_mode),
        Top_SKU   =("TopSKU",   lambda x: (x == "Core").sum()),
        Tag       =("tags",      safe_first),
        Sales     =("Sale","sum"),
        LiveStk   =("Live Total buying Amount","sum"),
        SCD       =("SCD","sum"),
        LiveCtn   =("Live Cartons","sum"),
        AWS_Act   =("AWSActive","sum"),
        AWS_AdjCtn=("AWS Adjusted","sum"),
        AWS_ActAmt=("AWSActiveAmt","sum"),
        MTD       =("MTD","sum"),
        TGT       =("SKU_TGT","sum"),
    ).reset_index()

    manu_summary["Shrt_Pct"] = (
        manu_summary["Shrt_SKU"] / manu_summary["T_SKU"] * 100
    ).round(0).fillna(0).astype(int).astype(str) + "%"
    manu_summary["OOS_Days"] = manu_summary["OOS_Days"].round(0).fillna(0).astype(int)
    manu_summary["TGT_Ach"]  = np.where(
        manu_summary["TGT"] > 0,
        (manu_summary["MTD"] / manu_summary["TGT"] * 100).round(1),
        0
    )

    display_manu = manu_summary.rename(columns={
        "ManufacturerName":"Manufacturer",
        "T_SKU":"T.SKU","Shrt_SKU":"Shrt SKU","Shrt_Pct":"Sht %",
        "OOS_Days":"OOS Days","Allow_Pri":"Allow Pri","Top_SKU":"Top SKU",
        "Sales":"Sales (₨)","LiveStk":"Live Stock (₨)",
        "LiveCtn":"Live Ctn","AWS_Act":"AWS Active","AWS_AdjCtn":"AWS ADJ Ctn",
        "AWS_ActAmt":"AWS Active Amt (₨)","TGT":"SKU TGT (₨)","TGT_Ach":"TGT Ach %"
    })

    fmt_cols = ["Sales (₨)","Live Stock (₨)","AWS Active Amt (₨)","MTD","SKU TGT (₨)"]
    for c in fmt_cols:
        display_manu[c] = display_manu[c].apply(lambda x: f"₨{x:,.0f}")

    # colour rows by OOS
    def highlight_manu(row):
        if row["Shrt SKU"] == row["T.SKU"]:
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        elif int(str(row["Sht %"]).replace("%","")) >= 50:
            return ["background-color:#2d2215;color:#fcd34d"] * len(row)
        return [""] * len(row)

    styled_manu = display_manu[[
        "Manufacturer","T.SKU","Shrt SKU","Sht %","OOS Days",
        "Allow Pri","Top SKU","Tag","Sales (₨)","Live Stock (₨)",
        "SCD","Live Ctn","AWS Active","AWS ADJ Ctn",
        "AWS Active Amt (₨)","MTD","SKU TGT (₨)","TGT Ach %"
    ]].style.apply(highlight_manu, axis=1)

    st.dataframe(styled_manu, use_container_width=True, height=340)

    # ── Charts row ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">SKU-Level Visuals</div>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        fig = px.bar(
            manu_summary.sort_values("Sales", ascending=False).head(15),
            x="ManufacturerName", y="Sales",
            color="TGT_Ach", color_continuous_scale="RdYlGn", range_color=[0,120],
            text=manu_summary.sort_values("Sales",ascending=False).head(15)["TGT_Ach"].apply(
                lambda x: f"{x:.0f}%"),
            title="Sales by Manufacturer (color=TGT Ach%)"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=320,
                          title_font_color="#e2e8f0", coloraxis_showscale=False,
                          xaxis=dict(tickangle=-40), margin=dict(b=110,t=40))
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig2 = px.scatter(
            manu_summary, x="Sales", y="LiveStk",
            size="T_SKU", color="TGT_Ach",
            hover_name="ManufacturerName",
            color_continuous_scale="RdYlGn", range_color=[0,120],
            title="Sales vs Live Stock (size=# SKUs)"
        )
        fig2.add_shape(type="line", x0=0, y0=0,
                       x1=manu_summary["Sales"].max(),
                       y1=manu_summary["Sales"].max(),
                       line=dict(dash="dash",color="white",width=1))
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=320,
                           title_font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)

    with ch3:
        oos_dist = manu_summary[["ManufacturerName","T_SKU","Shrt_SKU"]].copy()
        oos_dist["In_Stock"] = oos_dist["T_SKU"] - oos_dist["Shrt_SKU"]
        oos_top = oos_dist.sort_values("Shrt_SKU", ascending=False).head(12)
        fig3 = go.Figure()
        fig3.add_bar(x=oos_top["ManufacturerName"], y=oos_top["In_Stock"],
                     name="In-Stock", marker_color="#4ade80")
        fig3.add_bar(x=oos_top["ManufacturerName"], y=oos_top["Shrt_SKU"],
                     name="OOS / Short", marker_color="#ef4444")
        fig3.update_layout(barmode="stack", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=320, title="SKU In-Stock vs OOS by Manufacturer",
                           title_font_color="#e2e8f0",
                           xaxis=dict(tickangle=-40), margin=dict(b=110,t=40),
                           legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Full SKU-level detail grid (matches screenshot columns) ──────────────
    st.markdown('<div class="section-header">Full SKU Grid — All Fields</div>',
                unsafe_allow_html=True)

    sku_grid = sku_df[[
        "ManufacturerName","BrandName","SKU Code","SKU Description",
        "type","TopSKU","AllowPri","tags",
        "IsOOS","OOS","Stock_Status",
        "Sale","Live Total buying Amount","Live Cartons","Live Units",
        "SCD","AWSActive","AWS Adjusted","AWSActiveAmt","AWSAtvPcs","AWS in Units",
        "MSL in Units","Available Total Units","Gap_Units",
        "Unit_Cost","Gap_Demand_Value","AWS_Demand_Value",
        "Total buying Amount",
        "MTD","SKU_TGT","TGT_Ach_Pct","Inventory_Coverage",
        "OdrQty","OdrValue","LTDS_Amount",
        "SubCategoryName","SKU Category","BusinessUnitName"
    ]].copy()

    # Format money cols
    for mc in ["Sale","Live Total buying Amount","Total buying Amount",
               "MTD","SKU_TGT","OdrValue","AWSActiveAmt",
               "Gap_Demand_Value","AWS_Demand_Value","LTDS_Amount"]:
        sku_grid[mc] = sku_grid[mc].apply(lambda x: f"₨{x:,.0f}")
    sku_grid["Unit_Cost"]       = sku_grid["Unit_Cost"].apply(lambda x: f"₨{x:,.2f}")
    sku_grid["TGT_Ach_Pct"]    = sku_grid["TGT_Ach_Pct"].apply(lambda x: f"{x:.1f}%")
    sku_grid["Inventory_Coverage"] = sku_grid["Inventory_Coverage"].apply(
        lambda x: f"{min(x,99):.2f}x" if np.isfinite(x) else "—")
    sku_grid["IsOOS"] = sku_grid["IsOOS"].map({True:"🔴 OOS", False:"🟢 OK"})

    sku_grid.rename(columns={
        "ManufacturerName":"Manufacturer","BrandName":"Brand",
        "SKU Code":"SKU Code","SKU Description":"Description",
        "type":"Type","TopSKU":"Core","AllowPri":"Allow Pri","tags":"Tag",
        "IsOOS":"Status","OOS":"OOS Days","Stock_Status":"Stock",
        "Sale":"Sales","Live Total buying Amount":"Live Stock (₨)",
        "Live Cartons":"Live Ctn","Live Units":"Live Units",
        "SCD":"SCD","AWSActive":"AWS Active Ctns","AWS Adjusted":"AWS ADJ Ctn",
        "AWSActiveAmt":"AWS Active Amt (₨)","AWSAtvPcs":"AWS Active Pcs",
        "AWS in Units":"AWS Units","MSL in Units":"MSL Units",
        "Available Total Units":"Avail Units","Gap_Units":"Gap Units",
        "Unit_Cost":"Unit Cost (₨)","Gap_Demand_Value":"Gap Demand (₨)",
        "AWS_Demand_Value":"Full AWS Demand (₨)",
        "Total buying Amount":"Total Buy Amt","MTD":"MTD","SKU_TGT":"SKU Target (₨)",
        "TGT_Ach_Pct":"Ach %","Inventory_Coverage":"Inv Coverage",
        "OdrQty":"Odr Qty","OdrValue":"Odr Value","LTDS_Amount":"LTDS Amt",
        "SubCategoryName":"Sub Category","SKU Category":"Category",
        "BusinessUnitName":"Business Unit"
    }, inplace=True)

    def style_sku_grid(row):
        if row["Status"] == "🔴 OOS":
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        if row["Stock"] == "Critical":
            return ["background-color:#2d1a00;color:#fcd34d"] * len(row)
        return [""] * len(row)

    styled_grid = sku_grid.style.apply(style_sku_grid, axis=1)
    st.dataframe(styled_grid, use_container_width=True, height=500)

    # download
    csv_bytes = sku_df.to_csv(index=False).encode()
    st.download_button("⬇️  Download Filtered SKU Data as CSV",
                       data=csv_bytes, file_name="sku_detail_export.csv",
                       mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 – ZERO SKU DEMAND FORECAST
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">🔮 Tentative Demand — Zero-Sale SKUs</div>',
                unsafe_allow_html=True)


    # ── Build zero-SKU demand frame ─────────────────────────────────────────
    # Start with ALL zero-sale SKUs from filtered df
    zero_all = df[(df["Sale"] == 0) | (df["MTD"] == 0)].copy()

    # Apply significance filter — only meaningful SKUs per manufacturer/category
    # Is_Significant already computed in load_data:
    #   Contrib_Ratio >= 1 (above-avg in manufacturer) OR top 30% LTDS in category
    zero_df = zero_all[zero_all["Is_Significant"]].copy()

    zero_df["MRP_n"] = pd.to_numeric(zero_df["MRP"], errors="coerce").fillna(0)

    # Manufacturer-level TGT for contribution % display
    manu_tgt_total = df.groupby("ManufacturerName")["Manu_TGT_True"].first().rename("Manu_TGT")
    zero_df = zero_df.join(manu_tgt_total, on="ManufacturerName")

    # Contribution % = SKU_TGT share of manufacturer target
    zero_df["Contribution_Pct"] = np.where(
        zero_df["Manu_TGT"] > 0,
        (zero_df["SKU_TGT"] / zero_df["Manu_TGT"] * 100).round(2),
        0
    )

    # ── Demand Units (already computed in load_data, use directly) ───────────
    # Gap_Units        = AWS - Available (actual shortfall)
    # TGT_Implied_Units = SKU_TGT / Unit_Cost (budget ceiling)
    # Demand_Units     = min(Gap, TGT_Implied) — CAPPED at target
    zero_df["AWS_Demand_Units"]     = zero_df["Gap_Units"]          # gap = AWS demand
    zero_df["TGT_Demand_Units"]     = zero_df["TGT_Implied_Units"]  # target ceiling
    zero_df["Blended_Demand_Units"] = zero_df["Demand_Units"]       # already capped
    zero_df["Demand_Value"]         = zero_df["Gap_Demand_Value"]   # cost of capped demand

    # ── Priority: based on Contrib_Ratio + Core flag + Gap size ─────────────
    def priority(row):
        if row["TopSKU"] == "Core" and row["Contrib_Ratio"] >= 2.0:
            return "🔴 P1 – Critical"
        elif row["Contrib_Ratio"] >= 1.5 or (row["TopSKU"] == "Core"):
            return "🟠 P2 – High"
        elif row["Contrib_Ratio"] >= 1.0:
            return "🟡 P3 – Medium"
        return "⚪ P4 – Low"
    zero_df["Priority"] = zero_df.apply(priority, axis=1)

    # ── Top-level KPIs ────────────────────────────────────────────────────────
    zk1,zk2,zk3,zk4,zk5 = st.columns(5)
    mini_card(zk1,"Significant Zero SKUs", f"{len(zero_df):,}",
              "#f87171")
    mini_card(zk2,"Filtered Out (Low Contrib)", f"{len(zero_all)-len(zero_df):,}",
              "#94a3b8")
    mini_card(zk3,"Gap Units (Demand)",
              f"{zero_df['AWS_Demand_Units'].sum():,.0f} pcs", "#60a5fa")
    mini_card(zk4,"Capped Demand Units",
              f"{zero_df['Blended_Demand_Units'].sum():,.0f} pcs", "#4ade80")
    mini_card(zk5,"Demand Value (₨ Cost)",
              fmt_num(zero_df["Demand_Value"].sum()), "#a78bfa")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Manufacturer-level grouped demand ────────────────────────────────────
    st.markdown('<div class="section-header">Demand by Manufacturer (Grouped)</div>',
                unsafe_allow_html=True)

    manu_demand = zero_df.groupby("ManufacturerName").agg(
        Zero_SKUs         =("SKU Code","count"),
        AWS_Demand        =("AWS_Demand_Units","sum"),
        TGT_Demand        =("TGT_Demand_Units","sum"),
        Blended_Demand    =("Blended_Demand_Units","sum"),
        Demand_Value      =("Demand_Value","sum"),
        Manu_TGT          =("Manu_TGT","first"),
        Avg_Contribution  =("Contribution_Pct","mean"),
        P1_SKUs           =("Priority", lambda x: (x=="🔴 P1 – Critical").sum()),
        P2_SKUs           =("Priority", lambda x: (x=="🟠 P2 – High").sum()),
    ).reset_index().sort_values("Demand_Value", ascending=False)

    manu_demand["TGT_Ach_Gap"] = (manu_demand["Manu_TGT"] - zero_df.groupby(
        "ManufacturerName")["MTD"].sum().reindex(
        manu_demand["ManufacturerName"]).values).clip(0)

    # ── Row 1: Integrated Demand Analytics (4 Columns) ──────────────────────
    dc1, dc2, dc3, dc4 = st.columns([1.2, 0.8, 1, 1])
    
    with dc1:
        fig_d1 = go.Figure()
        top_md = manu_demand.head(15) # Reduced to top 15 for narrow column
        fig_d1.add_bar(x=top_md["ManufacturerName"], y=top_md["AWS_Demand"],
                       name="AWS", marker_color="#60a5fa")
        fig_d1.add_bar(x=top_md["ManufacturerName"], y=top_md["TGT_Demand"],
                       name="TGT", marker_color="#f59e0b")
        fig_d1.update_layout(barmode="group", template="plotly_dark",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             height=250, title="Top 15 Manufacturers — Demand Signals",
                             legend=dict(orientation="h", y=1.2, font_size=8),
                             xaxis=dict(tickangle=-45, tickfont_size=9), margin=dict(l=0,r=0,t=40,b=40))
        st.plotly_chart(fig_d1, use_container_width=True)

    with dc2:
        fig_d2 = px.pie(manu_demand.head(10), values="Demand_Value",
                        names="ManufacturerName", hole=0.5,
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        title="Demand Value Share — Top 10 Manufacturers")
        fig_d2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             height=250, showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
        fig_d2.update_traces(textinfo="percent")
        st.plotly_chart(fig_d2, use_container_width=True)

    with dc3:
        fig_d3 = px.scatter(manu_demand, x="Avg_Contribution", y="Blended_Demand",
                            size="Zero_SKUs", color="Demand_Value",
                            color_continuous_scale="Viridis")
        fig_d3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=250,
                             title="Avg Contribution % vs Demand Value (size=# Zero SKUs)",
                             coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=20))
        st.plotly_chart(fig_d3, use_container_width=True)

    with dc4:
        pr_counts = zero_df["Priority"].value_counts().reset_index()
        pr_counts.columns = ["Priority","Count"]
        pr_colors = {"🔴 P1 – Critical":"#ef4444","🟠 P2 – High":"#f97316","🟡 P3 – Medium":"#f59e0b","⚪ P4 – Low":"#94a3b8"}
        fig_d4 = px.bar(pr_counts, x="Priority", y="Count", color="Priority", color_discrete_map=pr_colors, text="Count")
        fig_d4.update_traces(textposition="outside")
        fig_d4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", height=250,
                             title="SKU Priority Distribution", showlegend=False,
                             xaxis=dict(tickfont_size=9), margin=dict(l=0,r=0,t=40,b=20))
        st.plotly_chart(fig_d4, use_container_width=True)

    # ── Manufacturer summary table ───────────────────────────────────────────
    st.markdown('<div class="section-header">Manufacturer Demand Summary Table</div>',
                unsafe_allow_html=True)

    manu_disp = manu_demand.copy()
    manu_disp["Demand_Value"]   = manu_disp["Demand_Value"].apply(lambda x: f"₨{x:,.0f}")
    manu_disp["Manu_TGT"]       = manu_disp["Manu_TGT"].apply(lambda x: f"₨{x:,.0f}")
    manu_disp["TGT_Ach_Gap"]    = manu_disp["TGT_Ach_Gap"].apply(lambda x: f"₨{x:,.0f}")
    manu_disp["AWS_Demand"]     = manu_disp["AWS_Demand"].apply(lambda x: f"{x:,.0f}")
    manu_disp["TGT_Demand"]     = manu_disp["TGT_Demand"].apply(lambda x: f"{x:,.0f}")
    manu_disp["Blended_Demand"] = manu_disp["Blended_Demand"].apply(lambda x: f"{x:,.0f}")
    manu_disp["Avg_Contribution"]= manu_disp["Avg_Contribution"].apply(lambda x: f"{x:.2f}%")
    manu_disp.rename(columns={
        "ManufacturerName":"Manufacturer","Zero_SKUs":"Zero SKUs",
        "AWS_Demand":"AWS Demand (units)","TGT_Demand":"TGT Demand (units)",
        "Blended_Demand":"Blended Demand (units)","Demand_Value":"Demand Value (₨ Cost)",
        "Manu_TGT":"Manu Target","Avg_Contribution":"Avg Contribution %",
        "P1_SKUs":"P1 SKUs","P2_SKUs":"P2 SKUs","TGT_Ach_Gap":"Target Gap"
    }, inplace=True)

    def style_demand(row):
        if row["P1 SKUs"] > 0:
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        return [""] * len(row)

    styled_demand = manu_disp.style.apply(style_demand, axis=1)
    st.dataframe(styled_demand, use_container_width=True, height=220)

    # ── SKU-level demand drill-down ──────────────────────────────────────────
    st.markdown('<div class="section-header">SKU-Level Demand Drill-Down</div>',
                unsafe_allow_html=True)

    sel_manu_d = st.selectbox(
        "Select Manufacturer to drill into",
        ["All"] + sorted(zero_df["ManufacturerName"].dropna().unique().tolist()),
        key="drill_manu"
    )
    sku_demand_df = zero_df if sel_manu_d == "All" else zero_df[zero_df["ManufacturerName"]==sel_manu_d]

    sku_demand_disp = sku_demand_df[[
        "ManufacturerName","BrandName","SKU Description","SKU Category",
        "type","TopSKU","Priority","Contribution_Pct",
        "AWS in Units","Available Total Units","MSL in Units",
        "AWS_Demand_Units","TGT_Demand_Units","Blended_Demand_Units",
        "MRP_n","Demand_Value","SKU_TGT","MTD","IsOOS"
    ]].copy().sort_values("Blended_Demand_Units", ascending=False)

    sku_demand_disp["Demand_Value"]  = sku_demand_disp["Demand_Value"].apply(lambda x: f"₨{x:,.0f}")
    sku_demand_disp["SKU_TGT"]       = sku_demand_disp["SKU_TGT"].apply(lambda x: f"₨{x:,.0f}")
    sku_demand_disp["MTD"]           = sku_demand_disp["MTD"].apply(lambda x: f"₨{x:,.0f}")
    sku_demand_disp["Contribution_Pct"] = sku_demand_disp["Contribution_Pct"].apply(lambda x: f"{x:.2f}%")
    sku_demand_disp["MRP_n"]         = sku_demand_disp["MRP_n"].apply(lambda x: f"₨{x:,.0f}")
    sku_demand_disp["IsOOS"]         = sku_demand_disp["IsOOS"].map({True:"🔴 OOS",False:"🟢 OK"})
    sku_demand_disp["AWS_Demand_Units"]    = sku_demand_disp["AWS_Demand_Units"].apply(lambda x: f"{x:,.0f}")
    sku_demand_disp["TGT_Demand_Units"]    = sku_demand_disp["TGT_Demand_Units"].apply(lambda x: f"{x:,.0f}")
    sku_demand_disp["Blended_Demand_Units"]= sku_demand_disp["Blended_Demand_Units"].apply(lambda x: f"{x:,.0f}")

    sku_demand_disp.rename(columns={
        "ManufacturerName":"Manufacturer","BrandName":"Brand",
        "SKU Description":"Description","SKU Category":"Category",
        "type":"Type","TopSKU":"Core","Contribution_Pct":"Contribution %",
        "AWS in Units":"AWS Units","Available Total Units":"Avail Units",
        "MSL in Units":"MSL Units","AWS_Demand_Units":"AWS Demand",
        "TGT_Demand_Units":"TGT Demand","Blended_Demand_Units":"Blended Demand",
        "MRP_n":"MRP","Demand_Value":"Demand Val","IsOOS":"Status"
    }, inplace=True)

    def style_sku_demand(row):
        if row["Priority"] == "🔴 P1 – Critical":
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        if row["Priority"] == "🟠 P2 – High":
            return ["background-color:#2d1a00;color:#fcd34d"] * len(row)
        return [""] * len(row)

    st.dataframe(sku_demand_disp.style.apply(style_sku_demand, axis=1),
                 use_container_width=True, height=250)

    csv_z = zero_df.to_csv(index=False).encode()
    st.download_button("⬇️  Download Zero-SKU Demand Forecast as CSV",
                       data=csv_z, file_name="zero_sku_demand_forecast.csv",
                       mime="text/csv")




# ════════════════════════════════════════════════════════════════════════════════
# TAB 9 – WORKING CAPITAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="section-header">💼 Working Capital Analysis — Category Level</div>',
                unsafe_allow_html=True)

    # ── Target DOC lookup (from doc sheet in source Excel) ───────────────────
    TARGET_DOC = {
        'Atta': 8, 'Baby Care': 12, 'Beverages': 8, 'Breakfast': 8,
        'Chaawal': 10, 'Chai and Coffee': 12, 'Cheeni': 8, 'Daalein': 8,
        'Dairy': 10, 'Home Care': 13, 'Instant Foods': 8,
        'Ketchups/Sauces/Achaar': 8, 'Meetha': 8, 'Nicotine & Cigarettes': 8,
        'Nutrition': 8, 'Oil and Ghee': 8, 'Personal Care': 13,
        'Confectionary': 13, 'Snacks': 8, 'Spices': 8, 'Wallet Top Up': 8
    }
    WORKING_DAYS = 25   # standard working days per month

    # ── User can override Target DOC per category ────────────────────────────
    with st.expander("⚙️  Override Target DOC (days) per Category", expanded=False):
        st.markdown("Defaults loaded from the **doc sheet**. Edit to run scenarios.")
        doc_cols = st.columns(4)
        cats_sorted = sorted([c for c in TARGET_DOC if c != 'Wallet Top Up'])
        user_doc = {}
        for i, cat in enumerate(cats_sorted):
            user_doc[cat] = doc_cols[i % 4].number_input(
                cat, min_value=1, max_value=60,
                value=TARGET_DOC[cat], key=f"doc_{cat}"
            )
        user_doc['Wallet Top Up'] = 8
        wd_override = st.number_input("Working Days per Month", 20, 31, WORKING_DAYS, key="wc_wd")

    # ── Build WC dataframe ───────────────────────────────────────────────────
    wc = df[df['SKU Category'] != 'Wallet Top Up'].groupby('SKU Category').agg(
        TGT        =('SKU_TGT',              'sum'),
        Avail_Inv  =('Total buying Amount',  'sum'),
        MTD        =('MTD',                  'sum'),
        LTDS       =('LTDS_Amount',          'sum'),
        Live_Inv   =('Live Total buying Amount','sum'),
    ).reset_index()

    # Add Wallet Top Up row separately (no target, just show inventory)
    wt = df[df['SKU Category'] == 'Wallet Top Up'].agg({
        'SKU_TGT': 'sum',
        'Total buying Amount': 'sum',
        'MTD': 'sum',
        'LTDS_Amount': 'sum',
        'Live Total buying Amount': 'sum'
    })
    wt.index = ['TGT', 'Avail_Inv', 'MTD', 'LTDS', 'Live_Inv']
    wt_row = pd.DataFrame([['Wallet Top Up'] + wt.tolist()], columns=['SKU Category']+wt.index.tolist())
    wc = pd.concat([wc, wt_row], ignore_index=True)

    wc['Target_DOC']      = wc['SKU Category'].map(user_doc).fillna(8).astype(int)
    wc['Daily_Runrate']   = np.where(wc['TGT'] > 0, wc['TGT'] / wd_override, 0).round(0)
    wc['WC_Requirement']  = (wc['Daily_Runrate'] * wc['Target_DOC']).round(0)
    wc['Advances']        = 0   # placeholder — can be wired to actual advance data
    wc['WC_Total']        = wc['Avail_Inv'] + wc['Advances']
    wc['Excess_Under']    = (wc['WC_Total'] - wc['WC_Requirement']).round(0)
    wc['Util_Pct']        = np.where(wc['WC_Requirement'] > 0,
                                wc['Excess_Under'] / wc['WC_Requirement'] * 100, 0).round(1)
    wc['Current_DOC']     = np.where(wc['Daily_Runrate'] > 0,
                                wc['Avail_Inv'] / wc['Daily_Runrate'], 0).round(1)
    wc['TGT_Ach_Pct']     = np.where(wc['TGT'] > 0,
                                wc['MTD'] / wc['TGT'] * 100, 0).round(1)
    wc['DOC_Gap']         = (wc['Current_DOC'] - wc['Target_DOC']).round(1)

    # Grand total row
    gt = wc.sum(numeric_only=True)
    gt['SKU Category']  = 'Grand Total'
    gt['Target_DOC']    = round(wc['Target_DOC'].mean(), 0)
    gt['Current_DOC']   = round(wc['Current_DOC'].mean(), 1)
    gt['Util_Pct']      = np.where(gt['WC_Requirement'] > 0,
                              gt['Excess_Under'] / gt['WC_Requirement'] * 100, 0).round(1)
    gt['TGT_Ach_Pct']   = np.where(gt['TGT'] > 0, gt['MTD'] / gt['TGT'] * 100, 0).round(1)
    gt['DOC_Gap']       = round(gt['Current_DOC'] - gt['Target_DOC'], 1)

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    wk1,wk2,wk3,wk4,wk5,wk6 = st.columns(6)
    total_wc_req  = wc['WC_Requirement'].sum()
    total_avail   = wc['Avail_Inv'].sum()
    total_excess  = wc['Excess_Under'].sum()
    over_deployed = wc[wc['Excess_Under'] > 0]['Excess_Under'].sum()
    under_deployed= wc[wc['Excess_Under'] < 0]['Excess_Under'].sum()
    overall_util  = (total_excess / total_wc_req * 100) if total_wc_req > 0 else 0
    avg_curr_doc  = wc[wc['Daily_Runrate']>0]['Current_DOC'].mean()

    mini_card(wk1, "WC Requirement",    fmt_num(total_wc_req),       "#6366f1")
    mini_card(wk2, "Available Inventory",fmt_num(total_avail),       "#22d3ee")
    mini_card(wk3, "Net (Excess)/Under", fmt_num(total_excess),
              "#4ade80" if total_excess >= 0 else "#f87171")
    mini_card(wk4, "Over-Deployed",      fmt_num(over_deployed),     "#4ade80")
    mini_card(wk5, "Under-Deployed",     fmt_num(abs(under_deployed)),"#f87171")
    mini_card(wk6, "Avg Current DOC",   f"{avg_curr_doc:.1f} days",
              "#4ade80" if avg_curr_doc >= 8 else "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main WC Table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Category Working Capital Summary</div>',
                unsafe_allow_html=True)

    def style_wc_table(row):
        if row["Category"] == "Grand Total":
            return ["background-color:#1e2130;font-weight:700;color:#e2e8f0"] * len(row)
        if row["(E)/U Deploy"] > 0:
            color = "#14532d"
        else:
            color = "#2d1515"
        return [f"background-color:{color}"] * len(row)

    wc_disp = wc.copy()
    gt_row  = pd.DataFrame([gt])
    wc_full = pd.concat([wc_disp, gt_row], ignore_index=True)

    wc_full.rename(columns={
        'SKU Category':'Category', 'TGT':'TGT (₨)',
        'Daily_Runrate':'Daily Runrate', 'Target_DOC':'Target DOC',
        'WC_Requirement':'WC Requirement', 'Avail_Inv':'Available Inv',
        'Advances':'Advances', 'WC_Total':'WC Total',
        'Excess_Under':'(E)/U Deploy', 'Util_Pct':'(E)/U Util %',
        'Current_DOC':'Current DOC', 'TGT_Ach_Pct':'TGT Ach %',
        'DOC_Gap':'DOC Gap', 'MTD':'MTD (₨)', 'LTDS':'LTDS (₨)',
        'Live_Inv':'Live Inv (₨)'
    }, inplace=True)

    # Ensure numeric columns are numeric for styling
    money_cols = ['TGT (₨)','Daily Runrate','WC Requirement','Available Inv',
                  'Advances','WC Total','(E)/U Deploy', 'MTD (₨)', 'LTDS (₨)', 'Live Inv (₨)']
    other_num_cols = ['(E)/U Util %', 'Current DOC', 'TGT Ach %', 'DOC Gap']
    
    for c in money_cols + other_num_cols:
        if c in wc_full.columns:
            wc_full[c] = pd.to_numeric(wc_full[c], errors='coerce').fillna(0)

    display_cols = ['Category','TGT (₨)','Daily Runrate','Target DOC',
                    'WC Requirement','Available Inv','Advances','WC Total',
                    '(E)/U Deploy','(E)/U Util %','Current DOC','DOC Gap','TGT Ach %']

    # Create styled object (on numeric data)
    styled_wc = wc_full[display_cols].style.apply(style_wc_table, axis=1)

    # Apply formatting for display
    styled_wc = styled_wc.format({
        'TGT (₨)': "₨{:,.0f}", 'Daily Runrate': "₨{:,.0f}", 'WC Requirement': "₨{:,.0f}",
        'Available Inv': "₨{:,.0f}", 'Advances': "₨{:,.0f}", 'WC Total': "₨{:,.0f}",
        '(E)/U Deploy': "₨{:,.0f}", 'MTD (₨)': "₨{:,.0f}", 'LTDS (₨)': "₨{:,.0f}", 'Live Inv (₨)': "₨{:,.0f}",
        '(E)/U Util %': "{:.1f}%", 'TGT Ach %': "{:.1f}%", 'Current DOC': "{:.1f}",
        'DOC Gap': lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}"
    })
    st.dataframe(styled_wc, use_container_width=True, height=640)

    # ── Charts Row 1: WC Requirement vs Available Inventory ──────────────────
    st.markdown('<div class="section-header">WC Requirement vs Available Inventory</div>',
                unsafe_allow_html=True)
    cc1, cc2 = st.columns([3, 2])

    wc_chart = wc[wc['SKU Category'] != 'Wallet Top Up'].copy()

    with cc1:
        fig_wc1 = go.Figure()
        fig_wc1.add_bar(x=wc_chart['SKU Category'], y=wc_chart['WC_Requirement'],
                        name='WC Requirement', marker_color='rgba(99,102,241,0.5)',
                        marker_line_color='#6366f1', marker_line_width=1.5)
        fig_wc1.add_bar(x=wc_chart['SKU Category'], y=wc_chart['Avail_Inv'],
                        name='Available Inventory', marker_color='#22d3ee')
        fig_wc1.update_layout(barmode='overlay', template='plotly_dark',
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              height=380, title='WC Requirement vs Available Inventory',
                              title_font_color='#e2e8f0',
                              legend=dict(orientation='h', y=1.1),
                              xaxis=dict(tickangle=-40), margin=dict(b=120,t=40))
        st.plotly_chart(fig_wc1, use_container_width=True)

    with cc2:
        # Waterfall of excess/under by category
        wc_sorted = wc_chart.sort_values('Excess_Under')
        colors = ['#4ade80' if v >= 0 else '#f87171' for v in wc_sorted['Excess_Under']]
        fig_wc2 = go.Figure(go.Bar(
            x=wc_sorted['SKU Category'], y=wc_sorted['Excess_Under'],
            marker_color=colors,
            text=[f"₨{v/1e3:.0f}K" for v in wc_sorted['Excess_Under']],
            textposition='outside'
        ))
        fig_wc2.add_hline(y=0, line_color='white', line_width=1)
        fig_wc2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', height=380,
                              title='(Excess)/Under Deployment by Category',
                              title_font_color='#e2e8f0',
                              xaxis=dict(tickangle=-40), margin=dict(b=120,t=40,r=60))
        st.plotly_chart(fig_wc2, use_container_width=True)

    # ── Charts Row 2: Current DOC vs Target DOC + Utilisation heatmap ────────
    st.markdown('<div class="section-header">DOC Analysis & Utilisation</div>',
                unsafe_allow_html=True)
    cc3, cc4 = st.columns(2)

    with cc3:
        fig_doc = go.Figure()
        fig_doc.add_bar(x=wc_chart['SKU Category'], y=wc_chart['Target_DOC'],
                        name='Target DOC', marker_color='rgba(245,158,11,0.4)',
                        marker_line_color='#f59e0b', marker_line_width=1.5)
        fig_doc.add_scatter(x=wc_chart['SKU Category'], y=wc_chart['Current_DOC'],
                            mode='lines+markers+text', name='Current DOC',
                            line=dict(color='#4ade80', width=2),
                            marker=dict(size=8),
                            text=wc_chart['Current_DOC'].apply(lambda x: f"{x:.1f}"),
                            textposition='top center', textfont=dict(size=9))
        fig_doc.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', height=380,
                              title='Current DOC vs Target DOC by Category',
                              title_font_color='#e2e8f0',
                              legend=dict(orientation='h', y=1.1),
                              xaxis=dict(tickangle=-40), margin=dict(b=120,t=40))
        st.plotly_chart(fig_doc, use_container_width=True)

    with cc4:
        # Utilisation % heatmap-style bar
        util_sorted = wc_chart.sort_values('Util_Pct')
        bar_colors = ['#4ade80' if v >= 0 else '#f87171' for v in util_sorted['Util_Pct']]
        fig_util = go.Figure(go.Bar(
            x=util_sorted['Util_Pct'], y=util_sorted['SKU Category'],
            orientation='h', marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in util_sorted['Util_Pct']],
            textposition='outside'
        ))
        fig_util.add_vline(x=0, line_color='white', line_width=1, line_dash='dash')
        fig_util.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)', height=380,
                               title='(Excess)/Under Utilisation % by Category',
                               title_font_color='#e2e8f0',
                               margin=dict(l=0, r=80, t=40, b=0))
        st.plotly_chart(fig_util, use_container_width=True)

    # ── Charts Row 3: TGT Achievement + WC Allocation Treemap ────────────────
    st.markdown('<div class="section-header">Target Achievement vs WC Allocation</div>',
                unsafe_allow_html=True)
    cc5, cc6 = st.columns(2)

    with cc5:
        fig_ach = px.scatter(
            wc_chart, x='WC_Requirement', y='TGT_Ach_Pct',
            size='Avail_Inv', color='Util_Pct',
            hover_name='SKU Category',
            color_continuous_scale='RdYlGn', range_color=[-100, 100],
            text='SKU Category',
            title='WC Requirement vs TGT Achievement% (size=Available Inv)',
            labels={'WC_Requirement':'WC Requirement (₨)', 'TGT_Ach_Pct':'TGT Ach %'}
        )
        fig_ach.update_traces(textposition='top center', textfont_size=9)
        fig_ach.add_hline(y=100, line_dash='dash', line_color='#4ade80',
                          annotation_text='100% target')
        fig_ach.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', height=420,
                              title_font_color='#e2e8f0')
        st.plotly_chart(fig_ach, use_container_width=True)

    with cc6:
        fig_tree = px.treemap(
            wc_chart, path=['SKU Category'], values='WC_Requirement',
            color='Util_Pct',
            color_continuous_scale='RdYlGn', range_color=[-100, 100],
            title='WC Requirement Treemap (color = Util%)',
            hover_data={'Avail_Inv':True,'Target_DOC':True,'Current_DOC':True}
        )
        fig_tree.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                               height=420, title_font_color='#e2e8f0',
                               margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Under-deployed deep dive ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🔴 Under-Deployed Categories — Action Required</div>',
                unsafe_allow_html=True)
    under_cats = wc_chart[wc_chart['Excess_Under'] < 0].sort_values('Excess_Under')

    if len(under_cats) > 0:
        uc1, uc2 = st.columns(2)
        with uc1:
            for _, row in under_cats.iterrows():
                gap_pct = abs(row['Util_Pct'])
                color   = "#ef4444" if gap_pct > 50 else "#f59e0b"
                st.markdown(f"""
                <div style="background:#1e2130;border-left:3px solid {color};
                border-radius:6px;padding:10px 14px;margin:6px 0;">
                  <b style="color:#e2e8f0">{row['SKU Category']}</b>
                  <span style="color:{color};float:right;font-weight:700">
                    ₨{abs(row['Excess_Under']):,.0f} under</span><br>
                  <span style="color:#94a3b8;font-size:0.8rem">
                    Current DOC: {row['Current_DOC']:.1f}d &nbsp;|&nbsp;
                    Target DOC: {row['Target_DOC']}d &nbsp;|&nbsp;
                    Gap: {row['DOC_Gap']:.1f}d &nbsp;|&nbsp;
                    Utilisation: {row['Util_Pct']:.1f}%
                  </span>
                </div>""", unsafe_allow_html=True)
        with uc2:
            fig_ug = px.funnel(
                under_cats.sort_values('Excess_Under'),
                x='Avail_Inv', y='SKU Category',
                title='Available Inventory — Under-Deployed Categories',
                color_discrete_sequence=['#f87171']
            )
            fig_ug.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                 height=400, title_font_color='#e2e8f0')
            st.plotly_chart(fig_ug, use_container_width=True)
    else:
        st.success("✅ All categories are adequately deployed!")

    # ── Download ──────────────────────────────────────────────────────────────
    csv_wc = wc_full[display_cols].to_csv(index=False).encode()
    st.download_button("⬇️  Download Working Capital Analysis as CSV",
                       data=csv_wc, file_name="working_capital_analysis.csv",
                       mime="text/csv")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 10 – ABC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="section-header">🔠 ABC Analysis — Inventory Value Classification</div>', unsafe_allow_html=True)

    # ── ABC Classification ───────────────────────────────────────────────────
    abc_df = df.groupby(["SKU Code","SKU Description","ManufacturerName","BrandName","SKU Category","BusinessUnitName"]).agg(
        Annual_Value=("Total buying Amount","sum"),
        LTDS=("LTDS_Amount","sum"),
        MTD=("MTD","sum"),
        Available_Units=("Available Total Units","sum"),
        AWS_Units=("AWS in Units","sum"),
        OOS=("IsOOS","max"),
    ).reset_index()

    # Use LTDS as proxy for annual demand value (better than purchase amt for classification)
    abc_df["Value_Proxy"] = np.where(abc_df["LTDS"] > 0, abc_df["LTDS"], abc_df["Annual_Value"])
    abc_df = abc_df[abc_df["Value_Proxy"] > 0].copy()
    abc_df = abc_df.sort_values("Value_Proxy", ascending=False).reset_index(drop=True)
    abc_df["Cumulative_Value"] = abc_df["Value_Proxy"].cumsum()
    abc_df["Cumulative_Pct"]  = abc_df["Cumulative_Value"] / abc_df["Value_Proxy"].sum() * 100
    abc_df["SKU_Pct"]         = (abc_df.index + 1) / len(abc_df) * 100

    # Standard ABC thresholds
    with st.expander("⚙️  ABC Thresholds (adjust to customise classification)", expanded=False):
        th_col1, th_col2 = st.columns(2)
        abc_a_thresh = th_col1.slider("Class A — Top Value Cumulative %", 50, 90, 80, 5, key="abc_a")
        abc_b_thresh = th_col2.slider("Class B — Mid Value Cumulative %", abc_a_thresh+1, 99, 95, 1, key="abc_b")

    def classify_abc(pct):
        if pct <= abc_a_thresh:  return "A"
        elif pct <= abc_b_thresh: return "B"
        return "C"

    abc_df["ABC_Class"] = abc_df["Cumulative_Pct"].apply(classify_abc)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    a_skus = abc_df[abc_df["ABC_Class"]=="A"]
    b_skus = abc_df[abc_df["ABC_Class"]=="B"]
    c_skus = abc_df[abc_df["ABC_Class"]=="C"]

    ak1,ak2,ak3,ak4,ak5,ak6 = st.columns(6)
    mini_card(ak1, "Total SKUs Classified",    f"{len(abc_df):,}",          "#e2e8f0")
    mini_card(ak2, "Class A SKUs",             f"{len(a_skus):,}",          "#4ade80")
    mini_card(ak3, "Class B SKUs",             f"{len(b_skus):,}",          "#60a5fa")
    mini_card(ak4, "Class C SKUs",             f"{len(c_skus):,}",          "#f59e0b")
    mini_card(ak5, "Class A Value Share",      f"{a_skus['Value_Proxy'].sum()/abc_df['Value_Proxy'].sum()*100:.1f}%", "#4ade80")
    mini_card(ak6, "Class A SKU Share",        f"{len(a_skus)/len(abc_df)*100:.1f}%", "#94a3b8")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        # Pareto Curve
        fig_abc1 = make_subplots(specs=[[{"secondary_y": True}]])
        color_map = {"A":"#4ade80","B":"#60a5fa","C":"#f59e0b"}
        for cls in ["A","B","C"]:
            sub = abc_df[abc_df["ABC_Class"]==cls]
            fig_abc1.add_bar(x=sub.index, y=sub["Value_Proxy"], name=f"Class {cls}",
                             marker_color=color_map[cls])
        fig_abc1.add_scatter(x=abc_df.index, y=abc_df["Cumulative_Pct"],
                             mode="lines", name="Cumulative %",
                             line=dict(color="#f8fafc",width=2), secondary_y=True)
        fig_abc1.add_hline(y=abc_a_thresh, line_dash="dash", line_color="#4ade80",
                           secondary_y=True, annotation_text=f"A: {abc_a_thresh}%")
        fig_abc1.add_hline(y=abc_b_thresh, line_dash="dash", line_color="#60a5fa",
                           secondary_y=True, annotation_text=f"B: {abc_b_thresh}%")
        fig_abc1.update_layout(barmode="stack", template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=320, title="Pareto Curve — ABC Classification",
                               legend=dict(orientation="h",y=1.15,font_size=9),
                               margin=dict(l=0,r=0,t=50,b=20))
        fig_abc1.update_yaxes(title_text="Value (₨)", secondary_y=False)
        fig_abc1.update_yaxes(title_text="Cumulative %", secondary_y=True)
        st.plotly_chart(fig_abc1, use_container_width=True)

    with ac2:
        # Class breakdown donut
        abc_summary = abc_df.groupby("ABC_Class").agg(
            SKUs=("SKU Code","count"),
            Value=("Value_Proxy","sum")
        ).reset_index()
        abc_summary["Pct"] = (abc_summary["Value"]/abc_summary["Value"].sum()*100).round(1)
        fig_abc2 = px.pie(abc_summary, values="Value", names="ABC_Class", hole=0.55,
                          color="ABC_Class", color_discrete_map=color_map,
                          title="Value Share by ABC Class")
        fig_abc2.update_traces(textinfo="label+percent", textfont_size=12)
        fig_abc2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               height=320, showlegend=False,
                               margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_abc2, use_container_width=True)

    with ac3:
        # Category breakdown stacked bar
        cat_abc = abc_df.groupby(["SKU Category","ABC_Class"]).size().reset_index(name="Count")
        fig_abc3 = px.bar(cat_abc, x="SKU Category", y="Count", color="ABC_Class",
                          color_discrete_map=color_map, barmode="stack",
                          title="ABC Class Distribution by Category")
        fig_abc3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=320,
                               xaxis=dict(tickangle=-40,tickfont_size=9),
                               legend=dict(orientation="h",y=1.15,font_size=9),
                               margin=dict(l=0,r=0,t=50,b=80))
        st.plotly_chart(fig_abc3, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    ac4, ac5 = st.columns([1.5,1])

    with ac4:
        # Manufacturer ABC scatter
        manu_abc = abc_df.groupby(["ManufacturerName","ABC_Class"]).agg(
            SKUs=("SKU Code","count"), Value=("Value_Proxy","sum")
        ).reset_index()
        manu_abc_pivot = manu_abc.pivot_table(index="ManufacturerName", columns="ABC_Class",
                                              values="SKUs", fill_value=0).reset_index()
        manu_abc_val   = abc_df.groupby("ManufacturerName")["Value_Proxy"].sum().reset_index()
        manu_abc_pivot = manu_abc_pivot.merge(manu_abc_val, on="ManufacturerName")
        manu_abc_pivot = manu_abc_pivot.sort_values("Value_Proxy", ascending=False).head(15)
        fig_abc4 = go.Figure()
        for cls in ["A","B","C"]:
            if cls in manu_abc_pivot.columns:
                fig_abc4.add_bar(x=manu_abc_pivot["ManufacturerName"], y=manu_abc_pivot[cls],
                                 name=f"Class {cls}", marker_color=color_map[cls])
        fig_abc4.update_layout(barmode="stack", template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=320, title="Top 15 Manufacturers — ABC SKU Mix",
                               xaxis=dict(tickangle=-40,tickfont_size=9),
                               legend=dict(orientation="h",y=1.15,font_size=9),
                               margin=dict(l=0,r=0,t=50,b=100))
        st.plotly_chart(fig_abc4, use_container_width=True)

    with ac5:
        # OOS rate by class
        oos_by_class = abc_df.groupby("ABC_Class").agg(
            Total=("SKU Code","count"), OOS=("OOS","sum")
        ).reset_index()
        oos_by_class["OOS_Rate"] = (oos_by_class["OOS"]/oos_by_class["Total"]*100).round(1)
        fig_abc5 = px.bar(oos_by_class, x="ABC_Class", y="OOS_Rate",
                          color="ABC_Class", color_discrete_map=color_map,
                          text="OOS_Rate", title="OOS Rate by ABC Class (%)")
        fig_abc5.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_abc5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=320,
                               showlegend=False, margin=dict(l=0,r=0,t=50,b=20))
        st.plotly_chart(fig_abc5, use_container_width=True)

    # ── ABC Detail Table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ABC Classified SKU Table</div>', unsafe_allow_html=True)
    f_abc_class = st.multiselect("Filter by Class", ["A","B","C"], default=["A","B","C"], key="abc_class_filter")
    abc_show = abc_df[abc_df["ABC_Class"].isin(f_abc_class)].copy()
    abc_show["Value_Proxy"] = abc_show["Value_Proxy"].apply(lambda x: f"₨{x:,.0f}")
    abc_show["Cumulative_Pct"] = abc_show["Cumulative_Pct"].apply(lambda x: f"{x:.1f}%")
    abc_show["SKU_Pct"]      = abc_show["SKU_Pct"].apply(lambda x: f"{x:.1f}%")
    abc_show["OOS"]          = abc_show["OOS"].map({True:"🔴 OOS",False:"🟢 OK"})
    abc_show = abc_show[["ABC_Class","SKU Description","ManufacturerName","BrandName",
                          "SKU Category","BusinessUnitName","Value_Proxy","Cumulative_Pct","SKU_Pct","OOS"]]
    abc_show.rename(columns={"ABC_Class":"Class","SKU Description":"Description",
                              "ManufacturerName":"Manufacturer","BrandName":"Brand",
                              "SKU Category":"Category","BusinessUnitName":"BU",
                              "Value_Proxy":"Value (₨)","Cumulative_Pct":"Cumulative %",
                              "SKU_Pct":"SKU %","OOS":"Status"}, inplace=True)

    def style_abc(row):
        c = {"A":"#052e16","B":"#0c1a2e","C":"#1c0a00"}.get(row["Class"],"")
        return [f"background-color:{c}"] * len(row) if c else [""] * len(row)

    st.dataframe(abc_show.style.apply(style_abc, axis=1), use_container_width=True, height=420)
    csv_abc = abc_df.to_csv(index=False).encode()
    st.download_button("⬇️  Download ABC Classification as CSV", data=csv_abc,
                       file_name="abc_analysis.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 11 – EOQ (ECONOMIC ORDER QUANTITY)
# ════════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.markdown('<div class="section-header">⚙️ Economic Order Quantity (EOQ) Analysis</div>', unsafe_allow_html=True)

    # ── Global EOQ parameters ────────────────────────────────────────────────
    with st.expander("⚙️  EOQ Parameters (adjust assumptions)", expanded=True):
        ep1, ep2, ep3 = st.columns(3)
        holding_pct  = ep1.slider("Annual Holding Cost (% of unit cost)", 5, 40, 20, 1, key="eoq_h") / 100
        order_cost   = ep2.number_input("Ordering Cost per Order (₨)", min_value=100, max_value=50000,
                                        value=2500, step=100, key="eoq_o")
        lead_days    = ep3.slider("Average Lead Time (days)", 1, 60, 7, 1, key="eoq_l")
        working_days_eoq = ep1.number_input("Working Days per Year", 200, 365, 300, key="eoq_wd")

    # ── Build EOQ dataframe ──────────────────────────────────────────────────
    eoq_df = df.groupby(["SKU Code","SKU Description","ManufacturerName",
                          "BrandName","SKU Category","BusinessUnitName"]).agg(
        Annual_Demand_Units=("AWS in Units","sum"),   # AWS as annual demand proxy
        Unit_Cost=("Unit_Cost","mean"),
        Available_Units=("Available Total Units","sum"),
        MTD=("MTD","sum"),
        LTDS=("LTDS_Amount","sum"),
    ).reset_index()

    # Scale AWS weekly units → annual (AWS is weekly stock level; demand ≈ AWS/week × working weeks)
    weeks_per_year  = working_days_eoq / 5
    eoq_df["Annual_Demand_Units"] = eoq_df["Annual_Demand_Units"] * weeks_per_year

    # Filter to SKUs with valid cost & demand
    eoq_df = eoq_df[(eoq_df["Unit_Cost"] > 0) & (eoq_df["Annual_Demand_Units"] > 0)].copy()

    # EOQ = sqrt(2 * D * S / H)
    eoq_df["H"]          = eoq_df["Unit_Cost"] * holding_pct        # holding cost per unit per year
    eoq_df["EOQ"]        = np.sqrt(2 * eoq_df["Annual_Demand_Units"] * order_cost / eoq_df["H"]).round(0)
    eoq_df["Orders_Per_Year"] = (eoq_df["Annual_Demand_Units"] / eoq_df["EOQ"]).round(1)
    eoq_df["Cycle_Days"] = np.where(eoq_df["Orders_Per_Year"] > 0,
                                    working_days_eoq / eoq_df["Orders_Per_Year"], 0).round(1)
    eoq_df["Total_Annual_Cost"] = (
        (eoq_df["Annual_Demand_Units"] / eoq_df["EOQ"]) * order_cost +
        (eoq_df["EOQ"] / 2) * eoq_df["H"]
    ).round(2)
    eoq_df["Current_Stock_Coverage_Days"] = np.where(
        eoq_df["Annual_Demand_Units"] > 0,
        eoq_df["Available_Units"] / (eoq_df["Annual_Demand_Units"] / working_days_eoq), 0
    ).round(1)
    eoq_df["Reorder_Point"] = (
        (eoq_df["Annual_Demand_Units"] / working_days_eoq) * lead_days
    ).round(0)
    eoq_df["Needs_Reorder"] = eoq_df["Available_Units"] <= eoq_df["Reorder_Point"]

    # ── KPI Cards ────────────────────────────────────────────────────────────
    ek1,ek2,ek3,ek4,ek5 = st.columns(5)
    reorder_needed = eoq_df["Needs_Reorder"].sum()
    avg_eoq        = eoq_df["EOQ"].median()
    avg_cycle      = eoq_df["Cycle_Days"].median()
    total_opt_cost = eoq_df["Total_Annual_Cost"].sum()
    avg_orders_yr  = eoq_df["Orders_Per_Year"].median()

    mini_card(ek1, "SKUs Analysed",       f"{len(eoq_df):,}",               "#e2e8f0")
    mini_card(ek2, "SKUs at Reorder Point",f"{int(reorder_needed):,}",      "#f87171")
    mini_card(ek3, "Median EOQ (units)",  f"{avg_eoq:,.0f} pcs",           "#60a5fa")
    mini_card(ek4, "Median Order Cycle",  f"{avg_cycle:.0f} days",          "#4ade80")
    mini_card(ek5, "Total Opt. Cost/yr",  fmt_num(total_opt_cost),           "#a78bfa")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        # EOQ distribution histogram
        fig_eoq1 = px.histogram(eoq_df[eoq_df["EOQ"] < eoq_df["EOQ"].quantile(0.95)],
                                x="EOQ", nbins=40, color_discrete_sequence=["#6366f1"],
                                title="EOQ Distribution (units)")
        fig_eoq1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=300,
                               margin=dict(l=0,r=0,t=50,b=30))
        st.plotly_chart(fig_eoq1, use_container_width=True)

    with ec2:
        # Reorder status by category
        reorder_cat = eoq_df.groupby("SKU Category").agg(
            Total=("SKU Code","count"), Reorder=("Needs_Reorder","sum")
        ).reset_index()
        reorder_cat["OK"] = reorder_cat["Total"] - reorder_cat["Reorder"]
        reorder_cat["Reorder_Rate"] = (reorder_cat["Reorder"]/reorder_cat["Total"]*100).round(1)
        fig_eoq2 = go.Figure()
        fig_eoq2.add_bar(x=reorder_cat["SKU Category"], y=reorder_cat["OK"],
                         name="OK", marker_color="#4ade80")
        fig_eoq2.add_bar(x=reorder_cat["SKU Category"], y=reorder_cat["Reorder"],
                         name="At Reorder Point", marker_color="#ef4444")
        fig_eoq2.update_layout(barmode="stack", template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=300, title="Reorder Status by Category",
                               xaxis=dict(tickangle=-40,tickfont_size=9),
                               legend=dict(orientation="h",y=1.15,font_size=9),
                               margin=dict(l=0,r=0,t=50,b=90))
        st.plotly_chart(fig_eoq2, use_container_width=True)

    with ec3:
        # EOQ vs current stock scatter
        plot_eoq = eoq_df.sample(min(400, len(eoq_df)))
        fig_eoq3 = px.scatter(plot_eoq, x="EOQ", y="Available_Units",
                              color="Needs_Reorder",
                              color_discrete_map={True:"#ef4444", False:"#4ade80"},
                              hover_name="SKU Description",
                              title="EOQ vs Current Available Stock")
        fig_eoq3.add_shape(type="line", x0=0, y0=0,
                           x1=plot_eoq["EOQ"].max(), y1=plot_eoq["EOQ"].max(),
                           line=dict(dash="dash", color="white", width=1))
        fig_eoq3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=300,
                               legend_title_text="At Reorder?",
                               margin=dict(l=0,r=0,t=50,b=20))
        st.plotly_chart(fig_eoq3, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    ec4, ec5 = st.columns(2)

    with ec4:
        # Order cycle by category box
        fig_eoq4 = px.box(eoq_df[eoq_df["Cycle_Days"]<365], x="SKU Category", y="Cycle_Days",
                          color="SKU Category", title="Order Cycle Days by Category")
        fig_eoq4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=320,
                               showlegend=False, xaxis=dict(tickangle=-40,tickfont_size=9),
                               margin=dict(l=0,r=0,t=50,b=90))
        st.plotly_chart(fig_eoq4, use_container_width=True)

    with ec5:
        # Top SKUs needing reorder
        reorder_skus = eoq_df[eoq_df["Needs_Reorder"]].sort_values("LTDS", ascending=False).head(20)
        fig_eoq5 = px.bar(reorder_skus, x="Annual_Demand_Units", y="SKU Description",
                          orientation="h", color="Current_Stock_Coverage_Days",
                          color_continuous_scale="RdYlGn", range_color=[0, lead_days*2],
                          title=f"Top 20 SKUs at Reorder Point (color=coverage days)",
                          labels={"Annual_Demand_Units":"Annual Demand (units)",
                                  "Current_Stock_Coverage_Days":"Coverage Days"})
        fig_eoq5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=320,
                               yaxis=dict(tickfont_size=9),
                               margin=dict(l=0,r=0,t=50,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_eoq5, use_container_width=True)

    # ── EOQ Detail Table ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">EOQ SKU Detail Table</div>', unsafe_allow_html=True)
    show_reorder_only = st.checkbox("Show only SKUs at Reorder Point", key="eoq_reorder_only")
    eoq_show = eoq_df[eoq_df["Needs_Reorder"]].copy() if show_reorder_only else eoq_df.copy()
    eoq_show = eoq_show.sort_values("LTDS", ascending=False)

    eoq_disp = eoq_show[["SKU Description","ManufacturerName","SKU Category",
                           "Annual_Demand_Units","Unit_Cost","EOQ","Reorder_Point",
                           "Available_Units","Needs_Reorder","Orders_Per_Year",
                           "Cycle_Days","Total_Annual_Cost","Current_Stock_Coverage_Days"]].copy()
    eoq_disp["Unit_Cost"]    = eoq_disp["Unit_Cost"].apply(lambda x: f"₨{x:,.2f}")
    eoq_disp["Total_Annual_Cost"] = eoq_disp["Total_Annual_Cost"].apply(lambda x: f"₨{x:,.0f}")
    eoq_disp["Annual_Demand_Units"] = eoq_disp["Annual_Demand_Units"].apply(lambda x: f"{x:,.0f}")
    eoq_disp["EOQ"] = eoq_disp["EOQ"].apply(lambda x: f"{x:,.0f}")
    eoq_disp["Reorder_Point"] = eoq_disp["Reorder_Point"].apply(lambda x: f"{x:,.0f}")
    eoq_disp["Needs_Reorder"] = eoq_disp["Needs_Reorder"].map({True:"🔴 YES",False:"🟢 NO"})
    eoq_disp.rename(columns={
        "SKU Description":"Description","ManufacturerName":"Manufacturer","SKU Category":"Category",
        "Annual_Demand_Units":"Annual Demand","Unit_Cost":"Unit Cost","EOQ":"EOQ (units)",
        "Reorder_Point":"Reorder Point","Available_Units":"Available","Needs_Reorder":"Need Reorder",
        "Orders_Per_Year":"Orders/Year","Cycle_Days":"Cycle Days",
        "Total_Annual_Cost":"Opt. Annual Cost","Current_Stock_Coverage_Days":"Coverage Days"
    }, inplace=True)

    def style_eoq(row):
        if row["Need Reorder"] == "🔴 YES":
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        return [""] * len(row)

    st.dataframe(eoq_disp.style.apply(style_eoq, axis=1), use_container_width=True, height=420)
    csv_eoq = eoq_show.to_csv(index=False).encode()
    st.download_button("⬇️  Download EOQ Analysis as CSV", data=csv_eoq,
                       file_name="eoq_analysis.csv", mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 12 – SAFETY STOCK
# ════════════════════════════════════════════════════════════════════════════════
with tabs[11]:
    st.markdown('<div class="section-header">🛡️ Safety Stock Analysis — Buffer Inventory Planning</div>', unsafe_allow_html=True)

    # ── Safety Stock Parameters ──────────────────────────────────────────────
    with st.expander("⚙️  Safety Stock Parameters", expanded=True):
        sp1, sp2, sp3, sp4 = st.columns(4)
        service_level_z = sp1.selectbox("Service Level (Z-Score)",
                                        ["90% (Z=1.28)","95% (Z=1.65)","97% (Z=1.88)","99% (Z=2.33)"],
                                        index=1, key="ss_z")
        z_map = {"90% (Z=1.28)":1.28,"95% (Z=1.65)":1.65,"97% (Z=1.88)":1.88,"99% (Z=2.33)":2.33}
        Z = z_map[service_level_z]
        lead_time_ss  = sp2.slider("Lead Time (days)", 1, 60, 7, key="ss_lt")
        demand_var_pct = sp3.slider("Demand Variability (% std dev of daily demand)", 5, 80, 25, 5, key="ss_dv") / 100
        lead_var_pct  = sp4.slider("Lead Time Variability (% std dev)", 0, 50, 15, 5, key="ss_lv") / 100
        wdays_ss = sp1.number_input("Working Days per Year", 200, 365, 300, key="ss_wd")

    # ── Safety Stock calculation ─────────────────────────────────────────────
    ss_df = df.groupby(["SKU Code","SKU Description","ManufacturerName",
                         "BrandName","SKU Category","BusinessUnitName"]).agg(
        AWS_Units=("AWS in Units","sum"),
        Available_Units=("Available Total Units","sum"),
        MSL_Units=("MSL in Units","sum"),
        Unit_Cost=("Unit_Cost","mean"),
        OOS=("IsOOS","max"),
        Gap_Units=("Gap_Units","sum"),
        LTDS=("LTDS_Amount","sum"),
    ).reset_index()

    ss_df = ss_df[ss_df["AWS_Units"] > 0].copy()

    # Daily demand = AWS weekly units / 5 working days
    ss_df["Daily_Demand"]      = ss_df["AWS_Units"] / 5
    ss_df["Demand_Std"]        = ss_df["Daily_Demand"] * demand_var_pct
    ss_df["Lead_Time_Std"]     = lead_time_ss * lead_var_pct

    # Safety Stock formula: SS = Z * sqrt(LT * σd² + d² * σLT²)
    ss_df["Safety_Stock"] = (
        Z * np.sqrt(
            lead_time_ss * ss_df["Demand_Std"]**2 +
            ss_df["Daily_Demand"]**2 * ss_df["Lead_Time_Std"]**2
        )
    ).round(0)

    ss_df["Reorder_Point_SS"] = (ss_df["Daily_Demand"] * lead_time_ss + ss_df["Safety_Stock"]).round(0)
    ss_df["Max_Stock"]        = (ss_df["Reorder_Point_SS"] + ss_df["AWS_Units"]).round(0)
    ss_df["Current_vs_SS"]   = ss_df["Available_Units"] - ss_df["Safety_Stock"]
    ss_df["SS_Status"]       = pd.cut(
        ss_df["Current_vs_SS"],
        bins=[-np.inf, 0, ss_df["Safety_Stock"].median(), np.inf],
        labels=["Below SS","Near SS","Above SS"]
    )
    ss_df["SS_Value"]        = (ss_df["Safety_Stock"] * ss_df["Unit_Cost"]).round(2)
    ss_df["SS_Coverage_Days"] = np.where(
        ss_df["Daily_Demand"] > 0,
        ss_df["Safety_Stock"] / ss_df["Daily_Demand"], 0
    ).round(1)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    sk1,sk2,sk3,sk4,sk5,sk6 = st.columns(6)
    below_ss  = (ss_df["SS_Status"]=="Below SS").sum()
    total_ss_val = ss_df["SS_Value"].sum()
    avg_ss_units = ss_df["Safety_Stock"].median()
    avg_ss_days  = ss_df["SS_Coverage_Days"].median()
    pct_ok       = (ss_df["Current_vs_SS"] >= 0).mean() * 100

    mini_card(sk1, "SKUs Analysed",        f"{len(ss_df):,}",              "#e2e8f0")
    mini_card(sk2, "Below Safety Stock",   f"{int(below_ss):,}",           "#f87171")
    mini_card(sk3, "Total SS Value (₨)",   fmt_num(total_ss_val),           "#a78bfa")
    mini_card(sk4, "Median SS (units)",    f"{avg_ss_units:,.0f} pcs",     "#60a5fa")
    mini_card(sk5, "Median SS Coverage",   f"{avg_ss_days:.1f} days",      "#4ade80")
    mini_card(sk6, "SKUs Above SS (%)",    f"{pct_ok:.1f}%",               "#4ade80" if pct_ok>=70 else "#f87171")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ─────────────────────────────────────────────────────────
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        # Safety stock status donut
        ss_status = ss_df["SS_Status"].value_counts().reset_index()
        ss_status.columns = ["Status","Count"]
        ss_colors_map = {"Below SS":"#ef4444","Near SS":"#f59e0b","Above SS":"#4ade80"}
        fig_ss1 = px.pie(ss_status, values="Count", names="Status", hole=0.55,
                         color="Status", color_discrete_map=ss_colors_map,
                         title="SKUs vs Safety Stock Level")
        fig_ss1.update_traces(textinfo="label+percent", textfont_size=11)
        fig_ss1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              height=320, showlegend=False, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_ss1, use_container_width=True)

    with sc2:
        # Safety stock by category bar
        ss_cat = ss_df.groupby("SKU Category").agg(
            Avg_SS=("Safety_Stock","mean"),
            Avg_Available=("Available_Units","mean"),
            Below_SS=("SS_Status", lambda x: (x=="Below SS").sum())
        ).reset_index().sort_values("Avg_SS", ascending=False).head(15)
        fig_ss2 = go.Figure()
        fig_ss2.add_bar(x=ss_cat["SKU Category"], y=ss_cat["Avg_SS"],
                        name="Avg Safety Stock", marker_color="#a78bfa")
        fig_ss2.add_bar(x=ss_cat["SKU Category"], y=ss_cat["Avg_Available"],
                        name="Avg Available", marker_color="#22d3ee")
        fig_ss2.update_layout(barmode="group", template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=320, title="Safety Stock vs Available Units by Category",
                              xaxis=dict(tickangle=-40,tickfont_size=9),
                              legend=dict(orientation="h",y=1.15,font_size=9),
                              margin=dict(l=0,r=0,t=50,b=90))
        st.plotly_chart(fig_ss2, use_container_width=True)

    with sc3:
        # SS Coverage Days distribution
        fig_ss3 = px.histogram(
            ss_df[ss_df["SS_Coverage_Days"] < ss_df["SS_Coverage_Days"].quantile(0.95)],
            x="SS_Coverage_Days", nbins=35, color_discrete_sequence=["#a78bfa"],
            title="Safety Stock Coverage Days Distribution"
        )
        fig_ss3.add_vline(x=lead_time_ss, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"Lead Time ({lead_time_ss}d)")
        fig_ss3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=320,
                              margin=dict(l=0,r=0,t=50,b=30))
        st.plotly_chart(fig_ss3, use_container_width=True)

    # ── Charts Row 2 ─────────────────────────────────────────────────────────
    sc4, sc5 = st.columns(2)

    with sc4:
        # Reorder point vs current stock scatter
        plot_ss = ss_df.sample(min(500, len(ss_df)))
        fig_ss4 = px.scatter(
            plot_ss, x="Reorder_Point_SS", y="Available_Units",
            color="SS_Status", color_discrete_map=ss_colors_map,
            hover_name="SKU Description",
            hover_data={"ManufacturerName":True,"Safety_Stock":":.0f","Daily_Demand":":.1f"},
            title="Reorder Point vs Current Available Stock",
            labels={"Reorder_Point_SS":"Reorder Point (units)","Available_Units":"Available (units)"}
        )
        # Diagonal line = reorder point == available
        max_val = max(plot_ss["Reorder_Point_SS"].max(), plot_ss["Available_Units"].max())
        fig_ss4.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(dash="dash",color="white",width=1))
        fig_ss4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=360,
                              margin=dict(l=0,r=0,t=50,b=20))
        st.plotly_chart(fig_ss4, use_container_width=True)

    with sc5:
        # Top SKUs below safety stock
        below_ss_df = ss_df[ss_df["SS_Status"]=="Below SS"].sort_values("Current_vs_SS").head(20)
        fig_ss5 = px.bar(
            below_ss_df, x="Current_vs_SS", y="SKU Description",
            orientation="h",
            color="Current_vs_SS", color_continuous_scale="Reds_r",
            title="Top 20 SKUs Below Safety Stock (units short)",
            labels={"Current_vs_SS":"Units Below Safety Stock"}
        )
        fig_ss5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=360,
                              yaxis=dict(tickfont_size=9), coloraxis_showscale=False,
                              margin=dict(l=0,r=0,t=50,b=20))
        st.plotly_chart(fig_ss5, use_container_width=True)

    # ── Manufacturer Safety Stock summary ────────────────────────────────────
    st.markdown('<div class="section-header">Safety Stock by Manufacturer</div>', unsafe_allow_html=True)
    ss_manu = ss_df.groupby("ManufacturerName").agg(
        SKUs=("SKU Code","count"),
        Total_SS=("Safety_Stock","sum"),
        Total_SS_Value=("SS_Value","sum"),
        Below_SS=("SS_Status", lambda x: (x=="Below SS").sum()),
        Total_Available=("Available_Units","sum"),
        Avg_Coverage=("SS_Coverage_Days","mean"),
    ).reset_index().sort_values("Total_SS_Value", ascending=False).head(20)

    ss_manu["Below_SS_Pct"] = (ss_manu["Below_SS"]/ss_manu["SKUs"]*100).round(1)
    ss_manu["Total_SS_Value_fmt"] = ss_manu["Total_SS_Value"].apply(lambda x: f"₨{x:,.0f}")
    ss_manu_disp = ss_manu[["ManufacturerName","SKUs","Total_SS","Total_SS_Value_fmt",
                              "Below_SS","Below_SS_Pct","Total_Available","Avg_Coverage"]].copy()
    ss_manu_disp.rename(columns={
        "ManufacturerName":"Manufacturer","Total_SS":"Total Safety Stock (units)",
        "Total_SS_Value_fmt":"SS Value (₨)","Below_SS":"Below SS",
        "Below_SS_Pct":"Below SS %","Total_Available":"Available Units",
        "Avg_Coverage":"Avg Coverage (days)"
    }, inplace=True)
    ss_manu_disp["Total Safety Stock (units)"] = ss_manu_disp["Total Safety Stock (units)"].apply(lambda x: f"{x:,.0f}")
    ss_manu_disp["Available Units"] = ss_manu_disp["Available Units"].apply(lambda x: f"{x:,.0f}")
    ss_manu_disp["Below SS %"]       = ss_manu_disp["Below SS %"].apply(lambda x: f"{x:.1f}%")
    ss_manu_disp["Avg Coverage (days)"] = ss_manu_disp["Avg Coverage (days)"].apply(lambda x: f"{x:.1f}")

    def style_ss_manu(row):
        pct_val = float(str(row["Below SS %"]).replace("%",""))
        if pct_val >= 50:
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        elif pct_val >= 25:
            return ["background-color:#2d1a00;color:#fcd34d"] * len(row)
        return [""] * len(row)

    st.dataframe(ss_manu_disp.style.apply(style_ss_manu, axis=1), use_container_width=True, height=400)

    # ── Full SKU Detail ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Safety Stock SKU Detail Table</div>', unsafe_allow_html=True)
    ss_show = ss_df.copy().sort_values("Current_vs_SS")
    ss_show_disp = ss_show[["SKU Description","ManufacturerName","SKU Category",
                             "Daily_Demand","Safety_Stock","Reorder_Point_SS",
                             "Available_Units","Current_vs_SS","SS_Status",
                             "SS_Coverage_Days","SS_Value","OOS"]].copy()
    ss_show_disp["Safety_Stock"]    = ss_show_disp["Safety_Stock"].apply(lambda x: f"{x:,.0f}")
    ss_show_disp["Reorder_Point_SS"]= ss_show_disp["Reorder_Point_SS"].apply(lambda x: f"{x:,.0f}")
    ss_show_disp["Available_Units"] = ss_show_disp["Available_Units"].apply(lambda x: f"{x:,.0f}")
    ss_show_disp["Current_vs_SS"]   = ss_show_disp["Current_vs_SS"].apply(lambda x: f"{x:,.0f}")
    ss_show_disp["Daily_Demand"]    = ss_show_disp["Daily_Demand"].apply(lambda x: f"{x:.1f}")
    ss_show_disp["SS_Coverage_Days"]= ss_show_disp["SS_Coverage_Days"].apply(lambda x: f"{x:.1f}")
    ss_show_disp["SS_Value"]        = ss_show_disp["SS_Value"].apply(lambda x: f"₨{x:,.0f}")
    ss_show_disp["OOS"]             = ss_show_disp["OOS"].map({True:"🔴 OOS",False:"🟢 OK"})
    ss_show_disp.rename(columns={
        "SKU Description":"Description","ManufacturerName":"Manufacturer","SKU Category":"Category",
        "Daily_Demand":"Daily Demand","Safety_Stock":"Safety Stock",
        "Reorder_Point_SS":"Reorder Point","Available_Units":"Available",
        "Current_vs_SS":"Vs Safety Stock","SS_Status":"SS Status",
        "SS_Coverage_Days":"SS Coverage (days)","SS_Value":"SS Value (₨)","OOS":"OOS Status"
    }, inplace=True)

    def style_ss_sku(row):
        if row["SS Status"] == "Below SS":
            return ["background-color:#2d1515;color:#fca5a5"] * len(row)
        if row["SS Status"] == "Near SS":
            return ["background-color:#2d1a00;color:#fcd34d"] * len(row)
        return [""] * len(row)

    st.dataframe(ss_show_disp.style.apply(style_ss_sku, axis=1), use_container_width=True, height=440)
    csv_ss = ss_show.to_csv(index=False).encode()
    st.download_button("⬇️  Download Safety Stock Analysis as CSV", data=csv_ss,
                       file_name="safety_stock_analysis.csv", mime="text/csv")

# Footer
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.75rem;padding:24px 0 8px 0;
border-top:1px solid #1e2130;margin-top:32px;">
    📦 Procurement Intelligence Dashboard · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)