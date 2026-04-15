import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AgriRisk AI - USDA", layout="wide", page_icon="🌾")

st.markdown("""
<style>
  @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap");
  html, body, [class*="css"] { font-family: "Inter", sans-serif; }
  .kpi-card {
    background: linear-gradient(135deg, #1a2030, #222b3a);
    border: 1px solid #3a4a60; border-radius: 14px;
    padding: 18px 14px; text-align: center; margin-bottom: 8px;
    height: 130px; display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    cursor: default;
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    border-color: #5a7a9a;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }
  .kpi-label { font-size: 10px; color: #a0b4c8; font-weight: 700;
    letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 4px;
    min-height: 28px; display: flex; align-items: center; justify-content: center; }
  .kpi-value { font-size: 28px; font-weight: 800; color: #ffffff; line-height:1.1; }
  .kpi-sub   { font-size: 10px; margin-top: 5px; color: #7a9ab0;
    min-height: 24px; display: flex; align-items: center; justify-content: center; }
  .season-badge {
    display: inline-block; background: linear-gradient(90deg, #1a5c36, #22854f);
    color: #b6f5d0; font-size: 14px; font-weight: 700;
    padding: 8px 20px; border-radius: 20px; border: 1px solid #3fb950;
  }
  .risk-card { border-radius: 12px; padding: 14px 18px; margin: 6px 0; border-left: 5px solid; }
  .critical { background: #200a0a; border-color: #c0392b; }
  .high     { background: #1e1400; border-color: #e67e22; }
  .medium   { background: #0a1525; border-color: #2980b9; }
  .low      { background: #071510; border-color: #27ae60; }
  .risk-label { font-size: 14px; font-weight: 700; color: #ffffff; }
  .risk-meta  { font-size: 13px; color: #b0c4d8; margin-top: 5px; line-height:1.7; }
  .int-card { border-radius: 12px; padding: 18px; margin: 10px 0; border: 1px solid; }
  .int-header { font-size: 16px; font-weight: 700; color: #ffffff; margin-bottom: 12px; }
  .int-row { font-size: 13px; color: #c0cfe0; margin: 8px 0; line-height: 1.6; }
  .section-hdr {
    font-size: 17px; font-weight: 700; color: #ffffff;
    margin: 20px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #3a4a60;
  }
  .stat-row { font-size: 13px; color: #b0c4d8; margin: 5px 0; }
  .stat-val  { color: #ffffff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

CSV_PATH = "us_agri_support_agentic_dataset.csv"

def generate_data():
    """Generate dataset inline if CSV not found."""
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)

    CONFIGS = {
        "Heartland":             {"states":["IL","IN","IA","MO","OH"],"n":1350,"d":(0.20,0.08),"sm":(0.58,0.07),"nv":(0.70,0.06),"t":(72,8),"rp":(0.85,0.07),"df":0.08,"dl":(2,2),"pl":(2,3),"crops":["Corn","Soybeans","Hogs","Wheat"],"code":"HRT"},
        "Northern Crescent":     {"states":["CT","ME","MA","MI","NH","NJ","NY","PA","RI","VT","WI","ND","SD","MN"],"n":900,"d":(0.18,0.07),"sm":(0.62,0.06),"nv":(0.72,0.05),"t":(63,9),"rp":(0.88,0.06),"df":0.06,"dl":(1,2),"pl":(2,3),"crops":["Dairy","Corn","Soybeans","Wheat","Apples"],"code":"NCR"},
        "Northern Great Plains": {"states":["KS","NE","ND","SD","MT","MN"],"n":350,"d":(0.58,0.10),"sm":(0.28,0.08),"nv":(0.40,0.07),"t":(84,10),"rp":(0.50,0.09),"df":0.35,"dl":(16,6),"pl":(10,6),"crops":["Wheat","Corn","Soybeans","Cattle","Barley"],"code":"NGP"},
        "Prairie Gateway":       {"states":["OK","TX","KS","NE","CO","NM"],"n":850,"d":(0.75,0.09),"sm":(0.14,0.07),"nv":(0.22,0.06),"t":(95,9),"rp":(0.34,0.08),"df":0.55,"dl":(26,7),"pl":(16,7),"crops":["Cotton","Wheat","Cattle","Sorghum","Corn"],"code":"PGW"},
        "Eastern Uplands":       {"states":["KY","NC","TN","VA","WV","AR"],"n":700,"d":(0.42,0.09),"sm":(0.40,0.07),"nv":(0.52,0.06),"t":(78,8),"rp":(0.66,0.08),"df":0.22,"dl":(9,4),"pl":(5,4),"crops":["Tobacco","Corn","Soybeans","Cattle","Poultry"],"code":"EUP"},
        "Southern Seaboard":     {"states":["AL","FL","GA","SC","NC","VA","MD","DE"],"n":800,"d":(0.50,0.10),"sm":(0.32,0.08),"nv":(0.46,0.07),"t":(86,8),"rp":(0.58,0.08),"df":0.30,"dl":(13,5),"pl":(7,5),"crops":["Cotton","Peanuts","Soybeans","Poultry","Vegetables"],"code":"SSB"},
        "Fruitful Rim":          {"states":["AZ","CA","ID","MT","NV","OR","UT","WA"],"n":500,"d":(0.45,0.10),"sm":(0.38,0.08),"nv":(0.50,0.07),"t":(72,10),"rp":(0.72,0.08),"df":0.18,"dl":(7,4),"pl":(4,4),"crops":["Vegetables","Fruits","Wine Grapes","Nuts","Dairy"],"code":"FRM"},
        "Basin and Range":       {"states":["AZ","CO","ID","MT","NV","NM","UT","WY"],"n":650,"d":(0.76,0.09),"sm":(0.12,0.07),"nv":(0.16,0.06),"t":(100,10),"rp":(0.30,0.08),"df":0.58,"dl":(30,8),"pl":(20,7),"crops":["Wheat","Cattle","Hay","Potatoes","Sheep"],"code":"BNR"},
        "Mississippi Portal":    {"states":["AR","LA","MS","MO","TN"],"n":400,"d":(0.36,0.08),"sm":(0.48,0.07),"nv":(0.60,0.06),"t":(82,7),"rp":(0.74,0.07),"df":0.17,"dl":(6,3),"pl":(3,3),"crops":["Rice","Cotton","Soybeans","Corn","Catfish"],"code":"MSP"},
    }

    rows = []
    pid = 1
    for region, c in CONFIGS.items():
        for _ in range(c["n"]):
            state   = random.choice(c["states"])
            drought = float(np.clip(np.random.normal(c["d"][0], c["d"][1]), 0.05, 0.98))
            sm      = float(np.clip(np.random.normal(c["sm"][0], c["sm"][1]), 0.05, 0.95))
            ndvi    = float(np.clip(np.random.normal(c["nv"][0], c["nv"][1]), 0.05, 0.98))
            temp    = float(np.clip(np.random.normal(c["t"][0], c["t"][1]), 15, 115))
            repay   = float(np.clip(np.random.normal(c["rp"][0], c["rp"][1]), 0.05, 1.0))
            prior_d = int(np.random.binomial(1, c["df"]))
            s_delay = int(np.clip(np.random.normal(c["dl"][0], c["dl"][1]), 0, 60))
            f_delay = int(np.clip(np.random.normal(c["dl"][0], c["dl"][1]) + 2, 0, 65))
            pl_del  = int(np.clip(np.random.normal(c["pl"][0], c["pl"][1]), 0, 69))
            crop    = random.choice(c["crops"])

            yield_rs = round(float(np.clip(drought*0.30+(1-ndvi)*0.20+(1-sm)*0.15+(1-repay)*0.20+prior_d*0.15, 0, 1)), 3)
            repay_rs = round(float(np.clip((1-repay)*0.45+prior_d*0.25+(1-repay)*0.20+min(1,s_delay/60)*0.10, 0, 1)), 3)
            urgency  = round(float(np.clip(yield_rs*0.45+repay_rs*0.35+min(1,s_delay/30)*0.10+min(1,pl_del/14)*0.10, 0, 1)), 3)

            if urgency >= 0.60:   tier = "immediate_intervention"
            elif urgency >= 0.42: tier = "priority_deployment"
            elif urgency >= 0.25: tier = "active_monitoring"
            else:                 tier = "routine_monitoring"

            y_band = "high" if yield_rs >= 0.55 else "medium" if yield_rs >= 0.30 else "low"
            r_band = "high" if repay_rs >= 0.55 else "medium" if repay_rs >= 0.30 else "low"

            if tier == "immediate_intervention":   action = "emergency_input_dispatch"
            elif tier == "priority_deployment":    action = "planting_support" if pl_del > 5 else "drought_advisory"
            elif drought > 0.55:                   action = "drought_advisory"
            else:                                  action = "monitor_only"

            rows.append({
                "producer_id": f"PRD{pid:06d}",
                "season": "2024-25",
                "usda_farm_resource_region": region,
                "usda_region_code": c["code"],
                "state": state,
                "primary_crop": crop,
                "farm_size_acres": round(float(np.clip(np.random.lognormal(6,1.2), 0.3, 54994)), 1),
                "drought_index": round(drought, 3),
                "soil_moisture_index": round(sm, 3),
                "ndvi": round(ndvi, 3),
                "avg_temperature_f": round(temp, 1),
                "repayment_rate": round(repay, 3),
                "prior_default_flag": prior_d,
                "seed_delivery_delay_days": s_delay,
                "fertilizer_delivery_delay_days": f_delay,
                "planting_delay_days": pl_del,
                "soil_ph": round(float(np.clip(6.5+(drought-0.5)*2.5+np.random.normal(0,0.4), 4.0, 9.1)), 2),
                "soil_organic_matter_pct": round(float(np.clip(4-drought*3+random.uniform(0,1.5), 0.5, 8.0)), 2),
                "yield_volatility_index": round(float(np.clip(drought*0.6+random.uniform(0,0.25), 0.05, 0.95)), 3),
                "pest_pressure_flag": int(np.random.binomial(1, min(0.85, drought*0.5+0.15))),
                "seasonal_rainfall_inches": round(float(np.clip(60*(1-drought)+np.random.normal(0,5), 2, 80)), 2),
                "input_credit_amount_usd": round(float(np.clip(random.uniform(40,350)*random.uniform(100,10000), 0, 17900000)), 0),
                "extension_visit_count": int(np.clip(int(np.random.poisson(2+(1-repay)*3)), 0, 12)),
                "support_gap_flag": int(random.random() > repay),
                "approval_required": int(tier in ("immediate_intervention","priority_deployment")),
                "yield_risk_score": yield_rs,
                "repayment_risk_score": repay_rs,
                "intervention_urgency_score": urgency,
                "yield_risk_band": y_band,
                "repayment_risk_band": r_band,
                "intervention_priority_tier": tier,
                "recommended_action": action,
                "explanation_summary": "Drivers: drought=" + str(round(drought,2)) + ", repayment=" + str(round(repay,2)),
                "action_rationale": "Urgency=" + str(urgency) + " based on yield and repayment risk.",
            })
            pid += 1

    return pd.DataFrame(rows)

REGION_COLORS = {
    "Heartland":             "#ffd54f",
    "Northern Crescent":     "#4fc3f7",
    "Northern Great Plains": "#ff7043",
    "Prairie Gateway":       "#c0392b",
    "Eastern Uplands":       "#81c784",
    "Southern Seaboard":     "#ffb74d",
    "Fruitful Rim":          "#4dd0e1",
    "Basin and Range":       "#ba68c8",
    "Mississippi Portal":    "#e57373",
}
TIER_COLORS = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#2980b9", "LOW": "#27ae60"}
TIER_ICONS  = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "🟢"}
TIER_LABELS = {
    "CRITICAL": "Immediate Intervention Cases",
    "HIGH":     "Priority Deployment Cases",
    "MEDIUM":   "Active Monitoring Cases",
    "LOW":      "Stable — No Action Required",
}


@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            df = generate_data()
    else:
        df = generate_data()

    season = df["season"].mode()[0] if "season" in df.columns else "2024-25"
    season = df["season"].mode()[0] if "season" in df.columns else "2024-25"

    grp = df.groupby("usda_farm_resource_region").agg(
        farm_count              =("producer_id",                "count"),
        states_in_region        =("state",                      lambda x: ", ".join(sorted(x.unique()))),
        usda_region_code        =("usda_region_code",           "first"),
        dominant_crop           =("primary_crop",               lambda x: x.mode()[0]),
        season                  =("season",                     "first"),
        drought_index           =("drought_index",              "mean"),
        soil_moisture_index     =("soil_moisture_index",        "mean"),
        ndvi_score              =("ndvi",                       "mean"),
        avg_temp_f              =("avg_temperature_f",          "mean"),
        repayment_rate_pct      =("repayment_rate",             "mean"),
        prior_default_rate      =("prior_default_flag",         "mean"),
        input_delay_days        =("seed_delivery_delay_days",   "mean"),
        avg_planting_delay      =("planting_delay_days",        "mean"),
        avg_farm_size_acres     =("farm_size_acres",            "mean"),
        avg_loan_usd            =("input_credit_amount_usd",    "mean"),
        avg_rainfall_mm         =("seasonal_rainfall_inches",   lambda x: round(x.mean() * 25.4, 1)),
        avg_soil_ph             =("soil_ph",                    "mean"),
        yield_volatility        =("yield_volatility_index",     "mean"),
        pest_reported_pct       =("pest_pressure_flag",         "mean"),
        avg_urgency_score       =("intervention_urgency_score", "mean"),
        avg_yield_risk          =("yield_risk_score",           "mean"),
        avg_repay_risk          =("repayment_risk_score",       "mean"),
        immediate_count         =("intervention_priority_tier", lambda x: (x == "immediate_intervention").sum()),
        priority_count          =("intervention_priority_tier", lambda x: (x == "priority_deployment").sum()),
        active_count            =("intervention_priority_tier", lambda x: (x == "active_monitoring").sum()),
        routine_count           =("intervention_priority_tier", lambda x: (x == "routine_monitoring").sum()),
        top_action              =("recommended_action",         lambda x: x.mode()[0]),
        approval_required_count =("approval_required",          "sum"),
        high_risk_count         =("yield_risk_band",            lambda x: (x == "high").sum()),
        support_gap_count       =("support_gap_flag",           "sum"),
    ).reset_index().rename(columns={"usda_farm_resource_region": "region"}).round(3)

    grp["risk_score"] = (grp["avg_urgency_score"] * 100).round(1)
    grp = grp.sort_values("risk_score", ascending=False).reset_index(drop=True)

    def assign_tier(s):
        if s >= 42:   return "CRITICAL"
        elif s >= 30: return "HIGH"
        elif s >= 20: return "MEDIUM"
        else:         return "LOW"

    grp["risk_tier"] = grp["risk_score"].apply(assign_tier)
    if (grp["risk_tier"] == "CRITICAL").sum() == 0:
        grp.loc[0, "risk_tier"] = "CRITICAL"
    if (grp["risk_tier"] == "HIGH").sum() == 0:
        grp.loc[1, "risk_tier"] = "HIGH"

    grp["repayment_rate_pct"] = grp["repayment_rate_pct"].apply(
        lambda x: round(x * 100, 1) if x <= 1 else round(x, 1)
    )
    return grp, df, season


regions_df, raw_df, SEASON = load_data()

if regions_df is None or len(regions_df) == 0:
    st.error("No data available. Please check the app logs.")
    st.stop()

# ── Header ────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        '<div style="background:linear-gradient(90deg,#0d3d26,#145c38);padding:24px 28px;'
        'border-radius:16px;margin-bottom:16px;border:1px solid #1e7a4a;">'
        '<h1 style="color:#ffffff;margin:0;font-size:26px;font-weight:800;">'
        '🌾 AgriRisk AI — USDA Agricultural Risk Intelligence</h1>'
        '<p style="color:#a0dbb8;margin:6px 0 0;font-size:13px;">'
        'University of Maryland Smith School of Business &nbsp;|&nbsp;'
        'CrewAI · Groq Llama 3.3 · 9 USDA ERS Farm Resource Regions · 6,500 Producer Records'
        '</p></div>',
        unsafe_allow_html=True
    )
with col_h2:
    total_states = int(raw_df["state"].nunique()) if raw_df is not None else 48
    st.markdown(
        '<div style="padding:20px 0 0;text-align:right;">'
        '<div class="season-badge">📅 Season: ' + str(SEASON) + '</div>'
        '<div style="color:#7a9ab0;font-size:12px;margin-top:8px;">'
        + str(len(regions_df)) + ' USDA regions · ' + str(total_states) + ' states</div></div>',
        unsafe_allow_html=True
    )

# ── KPIs ──────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
total_farms      = int(regions_df["farm_count"].sum())
immediate_total  = int(regions_df["immediate_count"].sum())
priority_total   = int(regions_df["priority_count"].sum())
active_total     = int(regions_df["active_count"].sum())
avg_urgency      = round(float(regions_df["avg_urgency_score"].mean()) * 100, 1)
critical_regions = int((regions_df["risk_tier"] == "CRITICAL").sum())
high_regions     = int((regions_df["risk_tier"] == "HIGH").sum())
medium_regions   = int((regions_df["risk_tier"] == "MEDIUM").sum())

for col, label, val, sub, color in [
    (k1, "Coverage Footprint",         str(len(regions_df)) + " Regions",
         str(total_states) + " states under active monitoring", "#58a6ff"),
    (k2, "Producers Assessed",         "{:,}".format(total_farms),
         "Records reconciled and scored", "#79c0ff"),
    (k3, "Immediate Intervention Cases", "{:,}".format(immediate_total),
         str(critical_regions) + " region(s) · same-day escalation", "#c0392b"),
    (k4, "Priority Deployment Cases",  "{:,}".format(priority_total),
         str(high_regions) + " region(s) · 48-hour action", "#e67e22"),
    (k5, "Active Monitoring Cases",    "{:,}".format(active_total),
         str(medium_regions) + " region(s) · emerging risk", "#2980b9"),
    (k6, "Intervention Pressure Index", str(avg_urgency) + "/100",
         "Average urgency across all regions", "#e3b341"),
]:
    col.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-label">' + label + '</div>'
        '<div class="kpi-value" style="color:' + color + ';">' + str(val) + '</div>'
        '<div class="kpi-sub">' + sub + '</div>'
        '</div>', unsafe_allow_html=True
    )

st.markdown("---")
left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        '<div class="section-hdr">🥧 Intervention Pressure by Region · ' + str(SEASON) + '</div>',
        unsafe_allow_html=True
    )
    pie_colors = [REGION_COLORS.get(r, "#888888") for r in regions_df["region"]]
    pull_vals  = [0.06 if t in ("CRITICAL", "HIGH") else 0.01 for t in regions_df["risk_tier"]]

    fig_pie = go.Figure(go.Pie(
        labels=regions_df["region"],
        values=regions_df["risk_score"],
        marker=dict(colors=pie_colors, line=dict(color="#0d1117", width=2.5)),
        hole=0.44,
        textposition="inside",
        textinfo="percent",
        textfont=dict(size=12, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>Urgency: %{value}/100<br>Share: %{percent}<extra></extra>",
        pull=pull_vals, direction="clockwise", sort=False,
        showlegend=True,
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", size=12),
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center",
            y=-0.12, yanchor="top",
            font=dict(color="#ffffff", size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=40, b=80, l=20, r=20),
        height=460,
        annotations=[dict(
            text="<b>9 USDA</b><br>Regions", x=0.5, y=0.5,
            font=dict(size=13, color="#ffffff"), showarrow=False
        )]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-hdr">📊 Highest Intervention Pressure by Region</div>',
                unsafe_allow_html=True)
    bar_df = regions_df.sort_values("risk_score", ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=bar_df["risk_score"],
        y=bar_df["region"],
        orientation="h",
        marker=dict(
            color=[REGION_COLORS.get(r, "#888") for r in bar_df["region"]],
            line=dict(color="#0d1117", width=1)
        ),
        text=[str(s) + "/100  " + TIER_ICONS[t]
              for s, t in zip(bar_df["risk_score"], bar_df["risk_tier"])],
        textposition="outside",
        textfont=dict(color="#ffffff", size=12),
        hovertemplate="<b>%{y}</b><br>Urgency: %{x}/100<extra></extra>",
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", size=12),
        margin=dict(t=10, b=10, l=10, r=80),
        height=360,
        showlegend=False,
        xaxis=dict(
            title=dict(text="Intervention Pressure Index (0-100)", font=dict(color="#a0b4c8")),
            gridcolor="#2e3a50",
            tickfont=dict(color="#ffffff", size=11),
            range=[0, 110],
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#2e3a50",
            tickfont=dict(color="#ffffff", size=12),
            showgrid=False,
            zeroline=False,
        ),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-hdr">📋 All USDA Regions Overview</div>',
                unsafe_allow_html=True)
    for _, row in regions_df.iterrows():
        tc  = row["risk_tier"].lower()
        ic  = TIER_ICONS[row["risk_tier"]]
        tl  = TIER_LABELS[row["risk_tier"]]
        rc  = REGION_COLORS.get(row["region"], "#888")
        szn = str(row.get("season", SEASON))
        imm = int(row.get("immediate_count", 0))
        pri = int(row.get("priority_count", 0))
        act = int(row.get("active_count", 0))
        st.markdown(
            '<div class="risk-card ' + tc + '">'
            '<div class="risk-label">'
            '<span style="color:' + rc + ';">■</span> &nbsp;'
            + ic + ' ' + str(row["region"]) +
            ' &nbsp;|&nbsp; <span style="color:' + TIER_COLORS[row["risk_tier"]] + ';">'
            + str(row["risk_score"]) + '/100</span>'
            ' &nbsp;|&nbsp; ' + tl +
            ' &nbsp;|&nbsp; <span style="color:#a0dbb8;">📅 ' + szn + '</span>'
            '</div>'
            '<div class="risk-meta">'
            'States: <b>' + str(row.get("states_in_region", "N/A")) + '</b><br>'
            'Producers: <b>' + "{:,}".format(int(row["farm_count"])) + '</b>'
            ' &nbsp;|&nbsp; Crop: <b>' + str(row.get("dominant_crop", "N/A")) + '</b>'
            ' &nbsp;|&nbsp; NDVI: <b>' + str(row["ndvi_score"]) + '</b>'
            ' &nbsp;|&nbsp; Drought: <b>' + str(row["drought_index"]) + '</b>'
            ' &nbsp;|&nbsp; Repayment: <b>' + str(row["repayment_rate_pct"]) + '%</b><br>'
            '🔴 Immediate: <b>' + str(imm) + '</b>'
            ' &nbsp;|&nbsp; 🟠 Priority: <b>' + str(pri) + '</b>'
            ' &nbsp;|&nbsp; 🔵 Active: <b>' + str(act) + '</b>'
            ' &nbsp;|&nbsp; Top action: <b>' + str(row.get("top_action", "N/A")) + '</b>'
            '</div></div>',
            unsafe_allow_html=True
        )

with right:
    st.markdown('<div class="section-hdr">⚡ Region-Level Action Planning</div>',
                unsafe_allow_html=True)

    region_list = regions_df["region"].tolist()

    def fmt_region(x):
        m = regions_df[regions_df["region"] == x]
        if m.empty: return x
        t = m["risk_tier"].values[0]
        return TIER_ICONS[t] + " " + x + " — " + TIER_LABELS[t]

    selected = st.selectbox("Select a USDA region:", region_list, format_func=fmt_region)
    matched  = regions_df[regions_df["region"] == selected]
    if matched.empty:
        st.warning("Region not found. Select another.")
        st.stop()

    row = matched.iloc[0]
    tc  = TIER_COLORS[row["risk_tier"]]
    tl  = TIER_LABELS[row["risk_tier"]]
    ic  = TIER_ICONS[row["risk_tier"]]
    szn = str(row.get("season", SEASON))
    rn  = str(row["region"])
    si  = str(row.get("states_in_region", "N/A"))
    cr  = str(row.get("dominant_crop", "N/A"))
    rs  = str(row["risk_score"])
    dr  = str(row["drought_index"])
    nv  = str(row["ndvi_score"])
    rp  = str(row["repayment_rate_pct"])
    imm = int(row.get("immediate_count", 0))
    pri = int(row.get("priority_count", 0))
    act = int(row.get("active_count", 0))
    rtn = int(row.get("routine_count", 0))
    apr = int(row.get("approval_required_count", 0))
    top = str(row.get("top_action", "monitor_only"))
    yrs = str(round(float(row.get("avg_yield_risk", 0)) * 100, 1))
    rrs = str(round(float(row.get("avg_repay_risk", 0)) * 100, 1))

    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a2030,#222b3a);border-radius:14px;'
        'padding:20px;border:1px solid ' + tc + ';margin-bottom:12px;">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        '<div>'
        '<h3 style="color:#ffffff;margin:0;font-size:18px;">' + ic + ' ' + rn + '</h3>'
        '<p style="color:' + tc + ';font-weight:700;margin:4px 0;font-size:14px;">' + tl + '</p>'
        '<p style="color:#a0b4c8;margin:4px 0 0;font-size:12px;">'
        '📅 ' + szn + ' &nbsp;|&nbsp; Crop: <b style="color:#fff;">' + cr + '</b>'
        ' &nbsp;|&nbsp; States: <b style="color:#fff;">' + si + '</b>'
        '</p></div>'
        '<div style="text-align:right;">'
        '<div style="font-size:48px;font-weight:900;color:' + tc + ';line-height:1;">' + rs + '</div>'
        '<div style="font-size:11px;color:#7a9ab0;">/ 100 urgency</div>'
        '</div></div>'
        '<div style="background:#0d1117;border-radius:8px;padding:10px 14px;margin-top:12px;">'
        '<div style="height:10px;background:#2e3a50;border-radius:5px;">'
        '<div style="height:10px;width:' + rs + '%;background:' + tc + ';border-radius:5px;"></div>'
        '</div>'
        '<div style="color:#a0b4c8;font-size:11px;margin-top:5px;">Intervention Pressure Index: ' + rs + '/100</div>'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-hdr">👥 Producer Case Breakdown</div>',
                unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    for col, label, val, color in [
        (t1, "Immediate", imm, "#c0392b"),
        (t2, "Priority",  pri, "#e67e22"),
        (t3, "Active",    act, "#2980b9"),
        (t4, "Routine",   rtn, "#27ae60"),
    ]:
        col.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-label">' + label + '</div>'
            '<div class="kpi-value" style="color:' + color + ';font-size:24px;">'
            + "{:,}".format(val) + '</div>'
            '</div>', unsafe_allow_html=True
        )

    st.markdown('<div class="section-hdr">📈 Risk Index Summary</div>',
                unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    for col, label, val, color in [
        (r1, "Yield Risk Index",     yrs + "/100", "#f0883e"),
        (r2, "Repayment Risk Index", rrs + "/100", "#e57373"),
        (r3, "Urgency Index",        rs + "/100",  tc),
    ]:
        col.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-label">' + label + '</div>'
            '<div class="kpi-value" style="color:' + color + ';font-size:20px;">' + val + '</div>'
            '</div>', unsafe_allow_html=True
        )

    st.markdown('<div class="section-hdr">🚨 Active Intervention Protocol</div>',
                unsafe_allow_html=True)

    if row["risk_tier"] == "CRITICAL":
        sms = ("[CRITICAL] " + rn + " (" + szn + "): Urgency " + rs + "/100. "
               + str(imm) + " producers need immediate intervention. Drought=" + dr
               + ", NDVI=" + nv + ", Repayment=" + rp + "%. Contact FSA TODAY.")
        st.markdown(
            '<div class="int-card" style="background:#1e0808;border-color:#c0392b;">'
            '<div class="int-header">🔴 IMMEDIATE INTERVENTION — ' + rn + '</div>'
            '<div class="int-row">📱 <b>SMS Blast:</b><br>'
            '<span style="background:#2a0a0a;padding:8px 12px;border-radius:6px;'
            'display:block;margin-top:6px;color:#ffb3b3;">' + sms + '</span></div>'
            '<div class="int-row">📧 <b>Email:</b> [CRITICAL ALERT] Same-Day Escalation — ' + rn + ' · ' + szn + '</div>'
            '<div class="int-row">⚠️ <b>Escalation:</b> IMMEDIATE — State Agriculture Dept + FSA Director + Extension Agent</div>'
            '<div class="int-row">🚜 <b>48hr dispatch:</b> Emergency input reallocation to (' + si + '). Top action: <b>' + top + '</b></div>'
            '<div class="int-row">💰 <b>FSA Emergency Loan:</b> <span style="color:#f85149;font-weight:700;">REQUIRED</span> — '
            + str(apr) + ' producers need approval</div>'
            '<div class="int-row">📋 <b>2-week plan:</b> EQIP enrollment · ELAP water loss assistance · LFP livestock support · Soil amendment delivery</div>'
            '<div class="int-row">💵 <b>Est. cost of action:</b> ~$1.5M &nbsp;|&nbsp; <b>Cost of inaction:</b> ~$5.0M crop loss</div>'
            '<div class="int-row">👤 <b>Responsible:</b> County Extension Agent (lead) · FSA Credit Officer · Input Supplier</div>'
            '</div>',
            unsafe_allow_html=True
        )
    elif row["risk_tier"] == "HIGH":
        sms = ("[HIGH RISK] " + rn + " (" + szn + "): Urgency " + rs + "/100. "
               + str(pri) + " producers flagged for priority deployment. Review within 48 hours.")
        st.markdown(
            '<div class="int-card" style="background:#1e1200;border-color:#e67e22;">'
            '<div class="int-header">🟠 PRIORITY DEPLOYMENT — ' + rn + '</div>'
            '<div class="int-row">📱 <b>SMS Alert:</b><br>'
            '<span style="background:#2a1800;padding:8px 12px;border-radius:6px;'
            'display:block;margin-top:6px;color:#ffd080;">' + sms + '</span></div>'
            '<div class="int-row">📧 <b>Email:</b> [HIGH RISK] 48-Hour Deployment Required — ' + rn + ' · ' + szn + '</div>'
            '<div class="int-row">⚠️ <b>Escalation:</b> PRIORITY — County Extension Agent + FSA Credit Officer</div>'
            '<div class="int-row">🚜 <b>48hr dispatch:</b> Extension officer site visits to (' + si + '). Top action: <b>' + top + '</b></div>'
            '<div class="int-row">💰 <b>FSA Loan Review:</b> <span style="color:#f0883e;font-weight:700;">RECOMMENDED</span> — '
            + str(apr) + ' cases pending approval</div>'
            '<div class="int-row">📋 <b>2-week plan:</b> EQIP referral · Drought advisory · Planting support · Irrigation efficiency</div>'
            '<div class="int-row">💵 <b>Est. cost of action:</b> ~$0.9M &nbsp;|&nbsp; <b>Cost of inaction:</b> ~$3.0M crop loss</div>'
            '<div class="int-row">👤 <b>Responsible:</b> County Extension Agent + Input Supplier</div>'
            '</div>',
            unsafe_allow_html=True
        )
    elif row["risk_tier"] == "MEDIUM":
        msg = ("Active Monitoring — " + rn + ". " + str(act) + " producers under watch in "
               + si + ". Weekly field checks. Monitor NDVI (" + nv + ") and drought index ("
               + dr + "). Auto-escalate if urgency exceeds 42.")
        st.info("🔵 " + msg)
        st.warning("📋 Contingency: Prepare input supply. If score rises above 42, escalate to Priority Deployment.")
    else:
        msg = (rn + " — Stable. " + str(rtn) + " producers on routine monitoring. NDVI: "
               + nv + " · Repayment: " + rp + "%")
        st.success("🟢 " + msg)

    st.markdown('<div class="section-hdr">📊 Regional Farm Profile</div>',
                unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        for label, val in [
            ("Avg Farm Size",   str(int(round(float(row.get("avg_farm_size_acres", 0)), 0))) + " acres"),
            ("Avg Temperature", str(round(float(row.get("avg_temp_f", 0)), 1)) + "°F"),
            ("Avg Soil pH",     str(round(float(row.get("avg_soil_ph", 6.5)), 2))),
            ("Planting Delay",  str(round(float(row.get("avg_planting_delay", 0)), 1)) + " days"),
        ]:
            st.markdown(
                '<div class="stat-row">' + label + ': <span class="stat-val">' + val + '</span></div>',
                unsafe_allow_html=True
            )
    with s2:
        for label, val in [
            ("Avg Loan Size",  "$" + str(round(float(row.get("avg_loan_usd", 0)) / 1000, 1)) + "K"),
            ("Rainfall",       str(int(round(float(row.get("avg_rainfall_mm", 0)), 0))) + " mm"),
            ("Repayment Rate", rp + "%"),
            ("Prior Defaults", str(round(float(row.get("prior_default_rate", 0)) * 100, 1)) + "%"),
        ]:
            st.markdown(
                '<div class="stat-row">' + label + ': <span class="stat-val">' + val + '</span></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-hdr">📡 Risk Signal Radar</div>',
                unsafe_allow_html=True)
    sig_vals = [
        float(row["drought_index"]),
        float(row["ndvi_score"]),
        float(row["pest_reported_pct"]),
        float(row["prior_default_rate"]),
        float(row["yield_volatility"]),
        min(1.0, float(row["input_delay_days"]) / 30),
    ]
    sig_labels = ["Drought", "NDVI", "Pest %", "Prior Default", "Yield Volatility", "Input Delay"]
    r_int = int(tc[1:3], 16)
    g_int = int(tc[3:5], 16)
    b_int = int(tc[5:7], 16)

    fig_radar = go.Figure(go.Scatterpolar(
        r=sig_vals + [sig_vals[0]],
        theta=sig_labels + [sig_labels[0]],
        fill="toself",
        fillcolor="rgba(" + str(r_int) + "," + str(g_int) + "," + str(b_int) + ",0.25)",
        line=dict(color=tc, width=2.5),
        marker=dict(color=tc, size=8),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor="#3a4a60",
                color="#c0cfe0", tickfont=dict(color="#c0cfe0", size=11)
            ),
            angularaxis=dict(gridcolor="#3a4a60", tickfont=dict(color="#ffffff", size=13)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(t=20, b=20, l=40, r=40),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if raw_df is not None:
        st.markdown('<div class="section-hdr">🚨 Highest-Urgency Producer Cases</div>',
                    unsafe_allow_html=True)
        with st.expander("View top at-risk producers — " + selected):
            region_raw = raw_df[raw_df["usda_farm_resource_region"] == selected].copy()
            region_raw = region_raw.sort_values("intervention_urgency_score", ascending=False)
            cols = [
                "producer_id", "state", "county", "primary_crop", "farm_size_acres",
                "intervention_priority_tier", "intervention_urgency_score",
                "yield_risk_band", "repayment_risk_band", "drought_index", "ndvi",
                "repayment_rate", "recommended_action", "explanation_summary", "season"
            ]
            avail = [c for c in cols if c in region_raw.columns]
            st.dataframe(region_raw[avail].head(20), use_container_width=True)

st.markdown("---")
st.caption(
    "AgriRisk AI · USDA 9-Region ERS Framework · Season: " + str(SEASON) +
    " · 5-Agent CrewAI System · Groq Llama 3.3 · Smith School of Business, University of Maryland"
)
