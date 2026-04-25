import pandas as pd
import numpy as np


# Weight distribution based on clinical impact for weight loss drug
# Higher weight = stronger indicator that patient needs the drug
# Negative weight = already diagnosed/treated (less likely to need new prescription)
def calculate_weighted_demand_score_corrected(df):
    weights = {

        # Direct indicators
        'Obesity among adults': 1.0,
        'High blood pressure among adults': 0.9,
        'High cholesterol among adults who have ever been screened': 0.85,

        # Secondary metabolic conditions
        'Coronary heart disease among adults': 0.8,
        'Stroke among adults': 0.75,
        'Chronic obstructive pulmonary disease among adults': 0.6,
        'Current asthma among adults': 0.55,

        # Lifestyle factors (need intervention)
        'No leisure-time physical activity among adults': 0.7,
        'Short sleep duration among adults': 0.5,
        'Binge drinking among adults': 0.4,

        # Mental health (obesity often comorbid)
        'Depression among adults': 0.65,
        'Frequent mental distress among adults': 0.6,
        'Frequent physical distress among adults': 0.55,
        'Loneliness among adults': 0.45,

        # Disability indicators (obesity-related disability)
        'Any disability among adults': 0.6,
        'Mobility disability among adults': 0.7,
        'Self-care disability among adults': 0.55,
        'Independent living disability among adults': 0.5,

        # Social determinants (barriers to care)
        'Food insecurity in the past 12 months among adults': 0.5,
        'Housing insecurity in the past 12 months among adults': 0.45,
        'Lack of reliable transportation in the past 12 months among adults': 0.4,
        'Utility services shut-off threat in the past 12 months among adults': 0.4,

        # Cancer risk (obesity-related cancers)
        'Cancer (non-skin) or melanoma among adults': 0.55,

        # Other conditions
        'Arthritis among adults': 0.5,
        'All teeth lost among adults aged >=65 years': 0.35,
        'Lack of social and emotional support among adults': 0.4,

        # NEGATIVE WEIGHTS - Already diagnosed

        # Already diagnosed with conditions that might already be managed
        'Diagnosed diabetes among adults': -0.3,
        'Taking medicine to control high blood pressure among adults with high blood pressure': -0.25,

        # Already engaged in healthcare (less need for intervention)
        'Visits to doctor for routine checkup within the past year among adults': -0.2,
        'Cholesterol screening among adults': -0.15,
        'Mammography use among women aged 50-74 years': -0.1,
        'Colorectal cancer screening among adults aged 45–75 years': -0.1,
        'Visited dentist or dental clinic in the past year among adults': -0.1,

        # Insurance/access (already have access to care)
        'Current lack of health insurance among adults aged 18-64 years': 0.3,  # Positive - needs access
        'Received food stamps in the past 12 months among adults': 0.35,  # Positive - needs support
    }

    print("\n" + "=" * 70)
    print("WEIGHTED DEMAND SCORE CALCULATION")
    print("=" * 70)

    # Check if required columns exist
    if 'Measure' not in df.columns or 'Data_Value' not in df.columns:
        print("Error: 'Measure' or 'Data_Value' column not found!")
        return df

    #measures are available and their weights
    print("\n1. Available measures and their weights:")
    available_measures = df['Measure'].unique()
    weight_summary = []

    for measure in available_measures:
        if measure in weights:
            weight = weights[measure]
            impact = "POSITIVE" if weight > 0 else "NEGATIVE" if weight < 0 else "NEUTRAL"
            weight_summary.append({
                'Measure': measure,
                'Weight': weight,
                'Impact': impact
            })
            print(f"   ✓ {measure[:50]:50s} weight: {weight:5.2f} ({impact})")
        else:
            print(f"   ? {measure[:50]:50s} (no weight assigned - will be ignored)")

    # Pivot the data
    print("\n2. Pivoting data to location level...")

    location_cols = ['CountyName', 'StateAbbr', 'StateDesc', 'CountyFIPS',
                     'latitude', 'longitude', 'Geolocation']
    existing_location_cols = [col for col in location_cols if col in df.columns]

    if not existing_location_cols:
        print("Warning: No location columns found. Using index as location.")
        df['temp_location_id'] = df.index
        existing_location_cols = ['temp_location_id']

    # Pivot
    pivot_df = df.pivot_table(
        index=existing_location_cols,
        columns='Measure',
        values='Data_Value',
        aggfunc='first'
    ).reset_index()

    print(f"   Created pivot table with {len(pivot_df):,} locations")

    # Calculate weighted demand score
    print("\n3. Calculating weighted demand score...")

    # Initialize score columns
    pivot_df['demand_score_raw'] = 0.0
    pivot_df['positive_contributions'] = 0.0
    pivot_df['negative_contributions'] = 0.0
    pivot_df['score_components'] = None

    # Calculate score for each row
    for idx, row in pivot_df.iterrows():
        total_score = 0
        positive_sum = 0
        negative_sum = 0
        components = {}

        for measure, weight in weights.items():
            if measure in pivot_df.columns and pd.notna(row[measure]):
                # Normalize value to 0-1 scale (assuming percentages 0-100)
                normalized_value = row[measure] / 100
                contribution = normalized_value * weight
                total_score += contribution

                if weight > 0:
                    positive_sum += contribution
                elif weight < 0:
                    negative_sum += contribution

                components[measure] = {
                    'value': row[measure],
                    'weight': weight,
                    'contribution': contribution
                }

        pivot_df.at[idx, 'demand_score_raw'] = total_score
        pivot_df.at[idx, 'positive_contributions'] = positive_sum
        pivot_df.at[idx, 'negative_contributions'] = negative_sum
        pivot_df.at[idx, 'score_components'] = components

    print(f"   Raw score range: {pivot_df['demand_score_raw'].min():.3f} - {pivot_df['demand_score_raw'].max():.3f}")
    print(
        f"   Positive contributions range: {pivot_df['positive_contributions'].min():.3f} - {pivot_df['positive_contributions'].max():.3f}")
    print(
        f"   Negative contributions range: {pivot_df['negative_contributions'].min():.3f} - {pivot_df['negative_contributions'].max():.3f}")

    # Normalize to 0-100 scale
    print("\n4. Normalizing scores to 0-100 scale...")

    min_score = pivot_df['demand_score_raw'].min()
    max_score = pivot_df['demand_score_raw'].max()

    pivot_df['demand_score_normalized'] = (
            (pivot_df['demand_score_raw'] - min_score) /
            (max_score - min_score) * 100
    )

    #DEMAND_TIER COLUMN
    print("\n4.5. Creating demand tier categories...")

    def assign_demand_tier(score):
        if score >= 75:
            return 'Critical'
        elif score >= 50:
            return 'High'
        elif score >= 25:
            return 'Moderate'
        else:
            return 'Low'

    pivot_df['demand_tier'] = pivot_df['demand_score_normalized'].apply(assign_demand_tier)

    print(f"   Tier distribution:")
    tier_dist = pivot_df['demand_tier'].value_counts()
    for tier in ['Critical', 'High', 'Moderate', 'Low']:
        if tier in tier_dist:
            print(f"      {tier}: {tier_dist[tier]:,} locations")

    # Add interpretation column - CORRECTED VERSION
    def interpret_score(row):
        # First check if it's already managed (high negative contributions)
        if row['negative_contributions'] < -0.2:
            return "Already managed - lower priority"

        # Then check demand tier (now it exists!)
        if row['demand_tier'] == 'Critical':
            return "CRITICAL - Deploy immediately (1-2 weeks)"
        elif row['demand_tier'] == 'High':
            return "HIGH - Deploy within 1 month"
        elif row['demand_tier'] == 'Moderate':
            return "MODERATE - Consider deployment (2-3 months)"
        elif row['demand_tier'] == 'Low':
            return "LOW - Monitor only (reassess in 6 months)"
        else:
            # Fallback based on score
            score = row['demand_score_normalized']
            if score >= 75:
                return "CRITICAL - Deploy immediately"
            elif score >= 50:
                return "HIGH - Deploy within 1 month"
            elif score >= 25:
                return "MODERATE - Consider deployment"
            else:
                return "LOW - Monitor only"

    pivot_df['deployment_priority'] = pivot_df.apply(interpret_score, axis=1)

    print("\n5. Score distribution:")
    print(f"   Mean score: {pivot_df['demand_score_normalized'].mean():.1f}")
    print(f"   Median score: {pivot_df['demand_score_normalized'].median():.1f}")
    print(f"   Std deviation: {pivot_df['demand_score_normalized'].std():.1f}")

    print("\n6. Demand tier distribution:")
    tier_counts = pivot_df['demand_tier'].value_counts()
    for tier in ['Low', 'Moderate', 'High', 'Critical']:
        if tier in tier_counts:
            count = tier_counts[tier]
            pct = (count / len(pivot_df)) * 100
            print(f"   {tier}: {count:,} locations ({pct:.1f}%)")

    print("\n7. Deployment priority breakdown:")
    priority_counts = pivot_df['deployment_priority'].value_counts()
    for priority, count in priority_counts.items():
        pct = (count / len(pivot_df)) * 100
        print(f"   {priority}: {count:,} locations ({pct:.1f}%)")

    # Count measures used
    pivot_df['num_measures_used'] = pivot_df['score_components'].apply(
        lambda x: len(x) if isinstance(x, dict) else 0
    )

    # Clean up
    if 'temp_location_id' in pivot_df.columns:
        pivot_df = pivot_df.drop(columns=['temp_location_id'])

    # Sort by demand score
    pivot_df = pivot_df.sort_values('demand_score_normalized', ascending=False)

    print("\n" + "=" * 70)
    print("CALCULATION COMPLETE")
    print("=" * 70)

    return pivot_df


def get_deployment_priorities(df_scored, min_score=70):

    deployment_df = df_scored[
        (df_scored['demand_score_normalized'] >= min_score) &
        (df_scored['deployment_priority'].str.contains('deployment', na=False))
        ].copy()

    # Select relevant columns
    output_cols = ['CountyName', 'StateAbbr', 'demand_score_normalized',
                   'demand_tier', 'deployment_priority', 'positive_contributions',
                   'negative_contributions', 'num_measures_used']

    existing_cols = [col for col in output_cols if col in deployment_df.columns]

    return deployment_df[existing_cols].sort_values('demand_score_normalized', ascending=False)


def explain_demand_score_detailed(df_scored, location_name=None):
    # If no specific location, show summary for ALL counties
    if location_name is None:
        print(f"\n{'=' * 80}")
        print(f"DEMAND SCORE SUMMARY - ALL COUNTIES")
        print(f"{'=' * 80}")

        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total counties: {len(df_scored):,}")
        print(f"   Average score: {df_scored['demand_score_normalized'].mean():.1f}/100")
        print(f"   Median score: {df_scored['demand_score_normalized'].median():.1f}/100")
        print(
            f"   Score range: {df_scored['demand_score_normalized'].min():.1f} - {df_scored['demand_score_normalized'].max():.1f}")

        # Tier distribution
        print(f"\n📈 DEMAND TIER DISTRIBUTION:")
        tier_counts = df_scored['demand_tier'].value_counts()
        for tier in ['Critical', 'High', 'Moderate', 'Low']:
            if tier in tier_counts:
                count = tier_counts[tier]
                pct = (count / len(df_scored)) * 100
                print(f"   {tier}: {count:,} counties ({pct:.1f}%)")

        # Deployment priority breakdown
        print(f"\n🎯 DEPLOYMENT PRIORITY BREAKDOWN:")
        priority_counts = df_scored['deployment_priority'].value_counts()
        for priority, count in priority_counts.items():
            pct = (count / len(df_scored)) * 100
            print(f"   {priority}: {count:,} counties ({pct:.1f}%)")

        # Top 10 counties
        print(f"\n🏆 TOP 10 COUNTIES BY DEMAND SCORE:")
        top_10 = df_scored.nlargest(10, 'demand_score_normalized')
        print(f"\n{'Rank':<5} {'County':<25} {'State':<5} {'Score':<8} {'Tier':<10} {'Priority'}")
        print(f"{'-' * 80}")
        for idx, (i, row) in enumerate(top_10.iterrows(), 1):
            print(
                f"{idx:<5} {row['CountyName']:<25} {row['StateAbbr']:<5} {row['demand_score_normalized']:<8.1f} {row['demand_tier']:<10} {row['deployment_priority'][:20]}")

        # Bottom 10 counties
        print(f"\n📉 BOTTOM 10 COUNTIES (LOWEST NEED):")
        bottom_10 = df_scored.nsmallest(10, 'demand_score_normalized')
        print(f"\n{'Rank':<5} {'County':<25} {'State':<5} {'Score':<8} {'Tier':<10}")
        print(f"{'-' * 60}")
        for idx, (i, row) in enumerate(bottom_10.iterrows(), 1):
            print(
                f"{idx:<5} {row['CountyName']:<25} {row['StateAbbr']:<5} {row['demand_score_normalized']:<8.1f} {row['demand_tier']:<10}")

        # State-wise summary
        print(f"\n🗺️ STATE-WISE SUMMARY:")
        state_summary = df_scored.groupby('StateAbbr').agg({
            'demand_score_normalized': ['mean', 'count'],
            'demand_tier': lambda x: (x == 'Critical').sum()
        }).reset_index()
        state_summary.columns = ['State', 'Avg Score', 'Counties', 'Critical Counties']
        state_summary = state_summary.sort_values('Avg Score', ascending=False)

        print(f"\n{'State':<10} {'Counties':<10} {'Avg Score':<12} {'Critical Counties':<18}")
        print(f"{'-' * 50}")
        for _, row in state_summary.head(10).iterrows():
            print(f"{row['State']:<10} {row['Counties']:<10} {row['Avg Score']:<12.1f} {row['Critical Counties']:<18}")

        return df_scored[['CountyName', 'StateAbbr', 'demand_score_normalized',
                          'demand_tier', 'deployment_priority',
                          'positive_contributions', 'negative_contributions']].copy()

    # If specific location provided, show detailed analysis for that county
    location_data = df_scored[df_scored['CountyName'] == location_name]

    if len(location_data) == 0:
        print(f"❌ Location '{location_name}' not found")
        print(f"   Available counties: {df_scored['CountyName'].head(10).tolist()}...")
        return None

    location_data = location_data.iloc[0]

    print(f"\n{'=' * 80}")
    print(f"📍 DETAILED DEMAND SCORE ANALYSIS: {location_name}")
    print(f"{'=' * 80}")
    print(f"\n📊 SCORE OVERVIEW:")
    print(f"   Overall Score: {location_data['demand_score_normalized']:.1f}/100")
    print(f"   Tier: {location_data['demand_tier']}")
    print(f"   Priority: {location_data['deployment_priority']}")
    print(f"\n📈 SCORE COMPONENTS:")
    print(f"   Positive contributions (need indicators): {location_data['positive_contributions']:.3f}")
    print(f"   Negative contributions (already treated): {location_data['negative_contributions']:.3f}")
    print(f"   Net score: {location_data['demand_score_raw']:.3f}")
    print(f"   Measures used: {location_data['num_measures_used']}")

    # Get components
    components = location_data['score_components']

    if isinstance(components, dict):
        print(f"\n✅ TOP POSITIVE DRIVERS (Why they need the drug):")
        positive_drivers = [(m, d) for m, d in components.items() if d['weight'] > 0]
        positive_drivers.sort(key=lambda x: x[1]['contribution'], reverse=True)

        for measure, data in positive_drivers[:8]:
            print(f"   + {measure[:50]}: {data['value']:.1f}% × {data['weight']:.2f} = +{data['contribution']:.3f}")

        print(f"\n❌ TOP NEGATIVE DRIVERS (Why they might NOT need the drug):")
        negative_drivers = [(m, d) for m, d in components.items() if d['weight'] < 0]
        negative_drivers.sort(key=lambda x: x[1]['contribution'])

        if negative_drivers:
            for measure, data in negative_drivers[:5]:
                print(f"   - {measure[:50]}: {data['value']:.1f}% × {data['weight']:.2f} = {data['contribution']:.3f}")
        else:
            print(f"   No negative drivers found - strong candidate for intervention!")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if location_data['demand_tier'] in ['Critical', 'High']:
            print(f"   IMMEDIATE DEPLOYMENT - High clinical need detected")
            print(f"   Priority: Schedule sales visit within 1-2 weeks")
        elif location_data['demand_tier'] == 'Moderate':
            print(f"   CONSIDER DEPLOYMENT - Moderate clinical need")
            print(f"   Priority: Schedule sales visit within 1 month")
        else:
            print(f"   MONITOR ONLY - Low clinical need at this time")
            print(f"   Priority: Reassess in 6 months")

    return location_data



def export_all_counties_data(df_scored, output_file='all_counties_analysis.csv'):

    # Create summary DataFrame for all counties
    export_df = df_scored[[
        'CountyName', 'StateAbbr', 'demand_score_normalized',
        'demand_tier', 'deployment_priority',
        'positive_contributions', 'negative_contributions',
        'num_measures_used'
    ]].copy()

    export_df['rank'] = export_df['demand_score_normalized'].rank(ascending=False).astype(int)

    export_df = export_df.sort_values('demand_score_normalized', ascending=False)

    # Save to CSV
    export_df.to_csv(output_file, index=False)
    print(f"✅ Exported data for all {len(export_df)} counties to {output_file}")

    return export_df

