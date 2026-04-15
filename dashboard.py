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
    if not os.path.exists(CSV_PATH):
        return None, None, "2024-25"

    df = pd.read_csv(CSV_PATH)
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
    st.error("CSV not found. Make sure us_agri_support_agentic_dataset.csv is in your GitHub repo.")
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
