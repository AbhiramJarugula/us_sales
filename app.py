import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import time
import warnings
import traceback
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Clinical Demand Analytics | Weight Loss Drug Market Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Color Palette
COLORS = {
    'primary': '#1a237e',
    'secondary': '#283593',
    'accent': '#1976d2',
    'success': '#2e7d32',
    'warning': '#ed6c02',
    'danger': '#d32f2f',
    'info': '#0288d1',
    'light': '#f5f5f5',
    'dark': '#212121',
    'white': '#ffffff',
    'critical': '#d32f2f',
    'high': '#ed6c02',
    'moderate': '#0288d1',
    'low': '#2e7d32',
    'purple': '#7b1fa2',
    'teal': '#00695c'
}

# Custom CSS
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    .dashboard-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}

    .dashboard-title {{
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }}

    .dashboard-subtitle {{
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: white;
    }}

    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
        transition: transform 0.2s;
        color: white;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}

    .metric-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: rgba(255,255,255,0.8);
        margin-bottom: 0.5rem;
    }}

    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }}

    .section-header {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['primary']};
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['accent']};
    }}

    .status-success {{
        background-color: #e8f5e9;
        border-left: 4px solid {COLORS['success']};
        color: #1b5e20;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}

    .status-info {{
        background-color: #e3f2fd;
        border-left: 4px solid {COLORS['accent']};
        color: #0d47a1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }}

    .insight-box {{
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-top: 3px solid {COLORS['accent']};
        color: {COLORS['dark']};
    }}

    .ready-badge {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: {COLORS['success']};
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }}

    .stButton > button {{
        background-color: {COLORS['accent']};
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s;
    }}

    .stButton > button:hover {{
        background-color: {COLORS['primary']};
        color: white;
        transform: translateY(-1px);
    }}
</style>
""", unsafe_allow_html=True)


class DemandDashboard:
    def __init__(self):
        self.scored_data = None
        self.data_loaded = False

    def load_data(self):
        try:
            from config import DATASET_PATH
            from dataloading import dataloading
            from datapreprocessing import preprocess_data

            with st.spinner("Loading data..."):
                dataset = dataloading(DATASET_PATH)
                processed_data, scored_data = preprocess_data(dataset)

                if 'CountyFIPS' in scored_data.columns:
                    scored_data = scored_data.drop_duplicates(subset=['CountyFIPS'], keep='first')
                elif 'CountyName' in scored_data.columns and 'StateAbbr' in scored_data.columns:
                    scored_data = scored_data.drop_duplicates(subset=['CountyName', 'StateAbbr'], keep='first')

                if 'deployment_priority' in scored_data.columns:
                    scored_data = scored_data.drop(columns=['deployment_priority'])

                return scored_data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def plot_demand_distribution_by_tier(self,df):
        fig = go.Figure()
        tiers = ['Critical', 'High', 'Moderate', 'Low']
        colors = [COLORS['critical'], COLORS['high'], COLORS['moderate'], COLORS['low']]

        for tier, color in zip(tiers, colors):
            tier_data = df[df['demand_tier'] == tier]['demand_score_normalized']
            if len(tier_data) > 0:
                fig.add_trace(go.Violin(
                    y=tier_data,
                    name=tier,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color,
                    line_color='white',
                    opacity=0.7
                ))

        fig.update_layout(
            title="Demand Score Distribution by Tier",
            yaxis_title="Demand Score",
            xaxis_title="Demand Tier",
            height=450,
            template="plotly_white",
            showlegend=False
        )
        return fig


    def plot_top_states_heatmap(self, df):
        state_stats = df.groupby('StateAbbr').agg({
            'demand_score_normalized': ['mean', 'count'],
            'demand_tier': lambda x: (x == 'Critical').sum()
        }).reset_index()
        state_stats.columns = ['State', 'Avg_Score', 'County_Count', 'Critical_Count']
        top_states = state_stats.nlargest(10, 'Avg_Score')

        fig = px.treemap(
            top_states,
            path=['State'],
            values='County_Count',
            color='Avg_Score',
            color_continuous_scale='RdYlGn_r',
            title="Top 10 States - Demand Heatmap",
            hover_data={'Avg_Score': ':.1f', 'Critical_Count': True}
        )
        fig.update_layout(height=450)
        return fig

    def plot_correlation_heatmap(self, df):
        metrics = ['demand_score_normalized', 'positive_contributions', 'negative_contributions', 'num_measures_used']
        measure_cols = ['Obesity among adults', 'High blood pressure among adults',
                        'High cholesterol among adults who have ever been screened',
                        'Diagnosed diabetes among adults', 'Depression among adults']

        for col in measure_cols:
            if col in df.columns:
                metrics.append(col)

        available_metrics = [m for m in metrics if m in df.columns]
        corr_matrix = df[available_metrics].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 9}
        ))
        fig.update_layout(
            title="Correlation Matrix - Key Health Metrics",
            height=550,
            width=650,
            xaxis={'tickangle': 45, 'tickfont': {'size': 10}},
            yaxis={'tickfont': {'size': 10}}
        )
        return fig


    def plot_scatter_demand_vs_measures(self, df):
        fig = go.Figure()

        for tier in ['Critical', 'High', 'Moderate', 'Low']:
            tier_data = df[df['demand_tier'] == tier]
            color = {'Critical': COLORS['critical'], 'High': COLORS['high'],
                     'Moderate': COLORS['moderate'], 'Low': COLORS['low']}[tier]

            fig.add_trace(go.Scatter(
                x=tier_data['num_measures_used'],
                y=tier_data['demand_score_normalized'],
                mode='markers',
                name=tier,
                marker=dict(color=color, size=8, opacity=0.6),
                text=tier_data['CountyName'],
                hovertemplate='<b>%{text}</b><br>Measures: %{x}<br>Score: %{y:.1f}<extra></extra>'
            ))

        fig.update_layout(
            title="Demand Score vs Number of Health Measures",
            xaxis_title="Number of Health Measures Available",
            yaxis_title="Demand Score",
            height=450,
            template="plotly_white"
        )
        return fig

    def plot_radar_chart(self, df):
        categories = {
            'Metabolic': ['Obesity among adults', 'High blood pressure among adults'],
            'Cardiovascular': ['Coronary heart disease among adults', 'Stroke among adults'],
            'Mental Health': ['Depression among adults', 'Frequent mental distress among adults'],
            'Lifestyle': ['No leisure-time physical activity among adults', 'Binge drinking among adults']
        }

        category_scores = {}
        for cat, measures in categories.items():
            values = []
            for measure in measures:
                if measure in df.columns:
                    values.extend(df[measure].dropna().values)
            if values:
                category_scores[cat] = np.mean(values)

        if category_scores:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(category_scores.values()),
                theta=list(category_scores.keys()),
                fill='toself',
                name='Health Categories',
                line_color=COLORS['accent'],
                fillcolor=f'rgba(25, 118, 210, 0.3)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Health Category Profile - Radar Chart",
                height=450,
                showlegend=False
            )
            return fig
        return None

    # Plot 6: Sunburst Chart
    def plot_sunburst_chart(self, df):
        df['Demand_Level'] = pd.cut(df['demand_score_normalized'],
                                    bins=[0, 25, 50, 75, 100],
                                    labels=['Low', 'Moderate', 'High', 'Critical'])

        hierarchy_data = df.groupby(['StateAbbr', 'Demand_Level']).size().reset_index(name='count')

        fig = px.sunburst(
            hierarchy_data,
            path=['StateAbbr', 'Demand_Level'],
            values='count',
            color='Demand_Level',
            color_discrete_map={
                'Critical': COLORS['critical'],
                'High': COLORS['high'],
                'Moderate': COLORS['moderate'],
                'Low': COLORS['low']
            },
            title="Demand Hierarchy by State",
            height=500
        )
        return fig

    def plot_boxplot_states(self, df):
        top_states = df.groupby('StateAbbr')['demand_score_normalized'].mean().nlargest(15).index
        plot_df = df[df['StateAbbr'].isin(top_states)]

        fig = go.Figure()
        for state in top_states:
            state_data = plot_df[plot_df['StateAbbr'] == state]['demand_score_normalized']
            fig.add_trace(go.Box(
                y=state_data,
                name=state,
                boxmean='sd'
            ))

        fig.update_layout(
            title="Demand Score Distribution - Top 15 States",
            yaxis_title="Demand Score",
            xaxis_title="State",
            height=500,
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        return fig


    def plot_cumulative_distribution(self, df):
        sorted_scores = df['demand_score_normalized'].sort_values()
        cumulative_pct = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_scores,
            y=cumulative_pct,
            mode='lines',
            fill='tozeroy',
            name='Cumulative %',
            line=dict(color=COLORS['accent'], width=2),
            fillcolor=f'rgba(25, 118, 210, 0.3)'
        ))

        fig.add_hline(y=50, line_dash="dash", line_color=COLORS['warning'],
                      annotation_text="50% of counties")
        fig.add_hline(y=80, line_dash="dash", line_color=COLORS['danger'],
                      annotation_text="80% of counties")

        fig.update_layout(
            title="Cumulative Demand Distribution",
            xaxis_title="Demand Score",
            yaxis_title="Cumulative Percentage of Counties",
            height=450,
            template="plotly_white"
        )
        return fig

    def plot_3d_scatter(self, df):
        plot_df = df[['demand_score_normalized', 'positive_contributions',
                      'negative_contributions', 'demand_tier']].copy()
        plot_df = plot_df.head(500)

        fig = go.Figure(data=[go.Scatter3d(
            x=plot_df['positive_contributions'],
            y=plot_df['negative_contributions'],
            z=plot_df['demand_score_normalized'],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_df['demand_score_normalized'],
                colorscale='Viridis',
                showscale=True
            ),
            text=df.loc[plot_df.index, 'CountyName'] if 'CountyName' in df.columns else None
        )])

        fig.update_layout(
            title="3D Analysis: Demand Drivers",
            scene=dict(
                xaxis_title="Positive Contributions",
                yaxis_title="Negative Contributions",
                zaxis_title="Demand Score"
            ),
            height=500
        )
        return fig


    def plot_funnel_analysis(self, df):
        stages = ['Total Counties', 'Moderate Demand', 'High Demand', 'Critical Demand']
        counts = [
            len(df),
            len(df[df['demand_tier'].isin(['Moderate', 'High', 'Critical'])]),
            len(df[df['demand_tier'].isin(['High', 'Critical'])]),
            len(df[df['demand_tier'] == 'Critical'])
        ]

        fig = go.Figure(go.Funnel(
            y=stages,
            x=counts,
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=[COLORS['low'], COLORS['moderate'], COLORS['high'], COLORS['critical']])
        ))

        fig.update_layout(
            title="Demand Pipeline Funnel",
            height=450,
            template="plotly_white"
        )
        return fig


    def plot_waterfall(self, df):
        avg_positive = df['positive_contributions'].mean()
        avg_negative = df['negative_contributions'].mean()
        net_score = avg_positive + avg_negative

        fig = go.Figure(go.Waterfall(
            name="Demand Components",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Positive Contributions", "Negative Impact", "Net Demand Score"],
            y=[avg_positive, avg_negative, net_score],
            text=[f"{avg_positive:.2f}", f"{avg_negative:.2f}", f"{net_score:.2f}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": COLORS['success']}},
            decreasing={"marker": {"color": COLORS['danger']}},
            totals={"marker": {"color": COLORS['accent']}}
        ))

        fig.update_layout(
            title="Demand Score Components - Waterfall Analysis",
            height=450,
            template="plotly_white"
        )
        return fig


    def plot_gauge_dashboard(self, df):
        critical_pct = len(df[df['demand_tier'] == 'Critical']) / len(df) * 100
        high_pct = len(df[df['demand_tier'] == 'High']) / len(df) * 100
        avg_score = df['demand_score_normalized'].mean()

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Critical Demand %", "High Demand %", "Average Score"),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
        )

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=critical_pct,
            title={'text': "% Critical"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['critical']}}
        ), row=1, col=1)

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=high_pct,
            title={'text': "% High"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['high']}}
        ), row=1, col=2)

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={'text': "Avg Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['accent']}}
        ), row=1, col=3)

        fig.update_layout(height=300)
        return fig


    def plot_geographic_map(self, df):
        if 'latitude' in df.columns and 'longitude' in df.columns:
            map_data = df.dropna(subset=['latitude', 'longitude'])

            fig = px.density_mapbox(
                map_data,
                lat='latitude',
                lon='longitude',
                z='demand_score_normalized',
                radius=10,
                center=dict(lat=39.8283, lon=-98.5795),
                zoom=3,
                mapbox_style="carto-positron",
                title="Geographic Demand Intensity Map",
                color_continuous_scale="Viridis",
                hover_name='CountyName' if 'CountyName' in map_data.columns else None
            )
            fig.update_layout(height=500)
            return fig
        return None

    def plot_bubble_chart(self, df):
        state_metrics = df.groupby('StateAbbr').agg({
            'demand_score_normalized': 'mean',
            'positive_contributions': 'mean',
            'num_measures_used': 'count'
        }).reset_index()

        fig = px.scatter(
            state_metrics,
            x='demand_score_normalized',
            y='positive_contributions',
            size='num_measures_used',
            color='demand_score_normalized',
            text='StateAbbr',
            title="Market Opportunity Bubble Chart",
            labels={
                'demand_score_normalized': 'Average Demand Score',
                'positive_contributions': 'Positive Contributions',
                'num_measures_used': 'Number of Measures'
            },
            color_continuous_scale='Viridis',
            size_max=50
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        return fig


    def plot_ridge_plot(self, df):
        regions = {
            'Northeast': ['NY', 'NJ', 'PA', 'MA', 'CT', 'RI', 'NH', 'VT', 'ME'],
            'South': ['TX', 'FL', 'GA', 'NC', 'SC', 'VA', 'TN', 'AL', 'MS', 'LA', 'AR', 'KY', 'WV'],
            'Midwest': ['IL', 'OH', 'MI', 'IN', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
            'West': ['CA', 'WA', 'OR', 'AZ', 'CO', 'UT', 'NV', 'NM', 'ID', 'MT', 'WY']
        }

        fig = go.Figure()
        colors = [COLORS['accent'], COLORS['success'], COLORS['warning'], COLORS['critical']]

        for i, (region, states) in enumerate(regions.items()):
            region_data = df[df['StateAbbr'].isin(states)]['demand_score_normalized']
            if len(region_data) > 0:
                fig.add_trace(go.Violin(
                    y=region_data,
                    name=region,
                    box_visible=True,
                    line_color=colors[i],
                    fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.5)',
                    opacity=0.7
                ))

        fig.update_layout(
            title="Demand Distribution by US Region",
            yaxis_title="Demand Score",
            height=500,
            template="plotly_white"
        )
        return fig


    def plot_insights_chart(self, df):
        def safe_count(column_name, threshold):
            #Safely count rows above threshold for a column
            if column_name in df.columns:
                return len(df[df[column_name].fillna(0) > threshold])
            return 0

        insights = {
            'Critical Need Counties': len(df[df['demand_tier'] == 'Critical']),
            'High Obesity Rate': safe_count('Obesity among adults', 35),
            'High Hypertension': safe_count('High blood pressure among adults', 35),
            'High Diabetes': safe_count('Diagnosed diabetes among adults', 15),
            'Mental Health Concern': safe_count('Depression among adults', 25),
            'Physical Inactivity': safe_count('No leisure-time physical activity among adults', 30)
        }

        fig = go.Figure(data=[go.Bar(
            x=list(insights.values()),
            y=list(insights.keys()),
            orientation='h',
            marker_color=COLORS['accent'],
            text=list(insights.values()),
            textposition='outside'
        )])

        fig.update_layout(
            title="Key Health Insights - County Counts",
            xaxis_title="Number of Counties",
            yaxis_title="Insight Category",
            height=400,
            template="plotly_white"
        )
        return fig


    def plot_percentile_analysis(self, df):
        df['Percentile'] = pd.qcut(df['demand_score_normalized'], q=10,
                                   labels=[f'{i * 10}-{(i + 1) * 10}%' for i in range(10)])

        percentile_stats = df.groupby('Percentile').agg({
            'demand_score_normalized': 'mean',
            'positive_contributions': 'mean',
            'num_measures_used': 'count'
        }).reset_index()

        fig = px.scatter(
            percentile_stats,
            x='demand_score_normalized',
            y='positive_contributions',
            size='num_measures_used',
            color='Percentile',
            title="Demand Score Percentile Analysis",
            labels={'demand_score_normalized': 'Average Demand Score',
                    'positive_contributions': 'Positive Contributions'},
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        return fig

    def plot_2d_density(self, df):
        fig = go.Figure()

        fig.add_trace(go.Histogram2dContour(
            x=df['positive_contributions'],
            y=df['demand_score_normalized'],
            colorscale='Viridis',
            contours=dict(coloring='heatmap'),
            showscale=True
        ))

        fig.update_layout(
            title="2D Density: Positive Contributions vs Demand Score",
            xaxis_title="Positive Contributions",
            yaxis_title="Demand Score",
            height=450,
            template="plotly_white"
        )
        return fig

    def run(self):
        """Main dashboard runner"""

        # Header
        st.markdown(f"""
        <div class="dashboard-header">
            <div class="dashboard-title">🏥 Clinical Demand Analytics - Market Intelligence Dashboard</div>
            <div class="dashboard-subtitle">Weight Loss Drug Market Opportunity Analysis | Real-time County-Level Insights</div>
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.scored_data = None


        with st.sidebar:
            st.markdown("### 🎛️ Control Panel")
            st.markdown("---")

            if not st.session_state.data_loaded:
                if st.button("Initialize Dashboard", use_container_width=True):
                    with st.spinner("Loading and processing data..."):
                        data = self.load_data()
                        if data is not None:
                            st.session_state.scored_data = data
                            st.session_state.data_loaded = True
                            st.success(f"✅ Loaded {len(data):,} unique counties!")
                            time.sleep(1)
                            st.rerun()

            if st.session_state.data_loaded and st.session_state.scored_data is not None:
                st.markdown(f"""
                <div class="status-success">
                    ✅ Dashboard Active<br>
                    {len(st.session_state.scored_data):,} counties loaded
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🎯 Filters")

                tier_filter = st.selectbox("Demand Tier", ['All', 'Critical', 'High', 'Moderate', 'Low'])
                state_filter = st.selectbox("State",
                                            ['All'] + sorted(st.session_state.scored_data['StateAbbr'].unique()))

                min_score = float(st.session_state.scored_data['demand_score_normalized'].min())
                max_score = float(st.session_state.scored_data['demand_score_normalized'].max())
                score_range = st.slider("Demand Score Range", min_score, max_score, (min_score, max_score))

        # Main content
        if st.session_state.data_loaded and st.session_state.scored_data is not None:
            # Apply filters
            df = st.session_state.scored_data.copy()

            if tier_filter != 'All':
                df = df[df['demand_tier'] == tier_filter]
            if state_filter != 'All':
                df = df[df['StateAbbr'] == state_filter]
            df = df[(df['demand_score_normalized'] >= score_range[0]) &
                    (df['demand_score_normalized'] <= score_range[1])]

            if len(df) == 0:
                st.warning("No data matches the filters. Please adjust your criteria.")
                return

            # Key Metrics Row
            st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Counties</div>
                    <div class="metric-value">{len(df):,}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                critical_pct = len(df[df['demand_tier'] == 'Critical']) / len(df) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Critical Demand</div>
                    <div class="metric-value">{critical_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                avg_score = df['demand_score_normalized'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Demand Score</div>
                    <div class="metric-value">{avg_score:.1f}/100</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                high_risk = len(df[df['demand_tier'].isin(['Critical', 'High'])])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Priority Counties</div>
                    <div class="metric-value">{high_risk:,}</div>
                </div>
                """, unsafe_allow_html=True)


            st.markdown('<div class="section-header">📈 Distribution & Correlation Analysis</div>',
                        unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                fig1 = self.plot_demand_distribution_by_tier(df)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = self.plot_top_states_heatmap(df)
                st.plotly_chart(fig2, use_container_width=True)

            col1, col2 = st.columns([1.2, 0.8])
            with col1:
                fig3 = self.plot_correlation_heatmap(df)
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                fig4 = self.plot_scatter_demand_vs_measures(df)
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown('<div class="section-header">🔬 Advanced Analytics</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                fig5 = self.plot_radar_chart(df)
                if fig5:
                    st.plotly_chart(fig5, use_container_width=True)

                fig6 = self.plot_sunburst_chart(df)
                st.plotly_chart(fig6, use_container_width=True)

            with col2:
                fig7 = self.plot_boxplot_states(df)
                st.plotly_chart(fig7, use_container_width=True)

                fig8 = self.plot_cumulative_distribution(df)
                st.plotly_chart(fig8, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig9 = self.plot_3d_scatter(df)
                st.plotly_chart(fig9, use_container_width=True)

                fig10 = self.plot_funnel_analysis(df)
                st.plotly_chart(fig10, use_container_width=True)

            with col2:
                fig11 = self.plot_waterfall(df)
                st.plotly_chart(fig11, use_container_width=True)

                fig12 = self.plot_gauge_dashboard(df)
                st.plotly_chart(fig12, use_container_width=True)

            # Row 5: Geographic Intelligence
            st.markdown('<div class="section-header">🗺️ Geographic Intelligence</div>', unsafe_allow_html=True)

            fig13 = self.plot_geographic_map(df)
            if fig13:
                st.plotly_chart(fig13, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig14 = self.plot_bubble_chart(df)
                st.plotly_chart(fig14, use_container_width=True)

                fig15 = self.plot_ridge_plot(df)
                st.plotly_chart(fig15, use_container_width=True)

            with col2:
                fig16 = self.plot_insights_chart(df)
                st.plotly_chart(fig16, use_container_width=True)

                fig17 = self.plot_percentile_analysis(df)
                st.plotly_chart(fig17, use_container_width=True)

            # Row 6: Additional Analysis
            st.markdown('<div class="section-header">📊 Additional Insights</div>', unsafe_allow_html=True)

            fig18 = self.plot_2d_density(df)
            st.plotly_chart(fig18, use_container_width=True)

            # Data Export Section
            st.markdown('<div class="section-header">📋 Data Export & Summary</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])
            with col1:
                display_cols = ['CountyName', 'StateAbbr', 'demand_score_normalized',
                                'demand_tier', 'positive_contributions', 'negative_contributions',
                                'num_measures_used']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols].head(100), use_container_width=True)

            with col2:
                csv = df[available_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Export Full Dataset (CSV)",
                    data=csv,
                    file_name=f"demand_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.markdown(f"""
                <div class="insight-box">
                    <strong>📈 Quick Statistics</strong><br>
                    • Total Locations: {len(df):,}<br>
                    • Avg Demand Score: {df['demand_score_normalized'].mean():.1f}<br>
                    • Median Score: {df['demand_score_normalized'].median():.1f}<br>
                    • Critical Locations: {len(df[df['demand_tier'] == 'Critical'])}<br>
                    • High Priority: {len(df[df['demand_tier'].isin(['Critical', 'High'])])}<br>
                    • Moderate Locations: {len(df[df['demand_tier'] == 'Moderate'])}<br>
                    • Low Locations: {len(df[df['demand_tier'] == 'Low'])}
                </div>
                """, unsafe_allow_html=True)

            # Ready badge
            st.markdown(f"""
            <div class="ready-badge">
                ✅ Dashboard Ready • {datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)

        else:
            # Welcome screen
            st.info("👈 Click 'Initialize Dashboard' in the sidebar to start")

            st.markdown("""
            ### 🚀 Dashboard Features:

            #### 📊 18+ Professional Visualizations:

            **Distribution & Correlation**
            1. Demand Score Distribution by Tier (Violin Plot)
            2. Top States Heatmap (Treemap)
            3. Correlation Matrix - Key Health Metrics
            4. Scatter Plot - Demand vs Measures

            **Advanced Analytics**
            5. Radar Chart - Health Categories
            6. Sunburst Chart - Demand Hierarchy
            7. Box Plot - State Comparison
            8. Cumulative Distribution Analysis

            **3D & Pipeline Analysis**
            9. 3D Scatter Plot - Multi-dimensional
            10. Funnel Analysis - Demand Pipeline
            11. Waterfall Chart - Score Components
            12. Gauge Dashboard - Key Metrics

            **Geographic Intelligence**
            13. Geographic Demand Map
            14. Bubble Chart - Market Opportunity
            15. Ridge Plot - Regional Distribution
            16. Insights Bar Chart

            **Additional Analysis**
            17. Percentile Trend Analysis
            18. 2D Density Contour Plot

            #### Key Metrics Tracked:
            - Obesity rates, hypertension, cholesterol levels
            - Cardiovascular health indicators
            - Mental health and lifestyle factors
            - Healthcare access patterns
            - Geographic demand concentration
            """)


def start():
    dashboard = DemandDashboard()
    dashboard.run()


if __name__ == "__main__":
    start()