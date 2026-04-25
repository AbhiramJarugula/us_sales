# preprocess_data.py
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import warnings

from analyzer import calculate_weighted_demand_score_corrected, get_deployment_priorities, \
    explain_demand_score_detailed, export_all_counties_data


def remove_unnecessary_columns(dataset):
    # Columns to remove
    columns_to_remove = [
        'DataValueTypeID',
        'Data_Value_Footnote',
        'Data_Value_Footnote_Symbol',
        'Data_Value_Type',
        'Data_Value_Unit',
        'DataSource'
    ]

    existing_cols = [col for col in columns_to_remove if col in dataset.columns]
    if existing_cols:
        dataset = dataset.drop(columns=existing_cols)
        print(f"Removed columns: {existing_cols}")
    else:
        print("None of the specified columns found in the dataset")
    return dataset


def remove_unwanted_measures(df):
    measures_to_remove = [
        'Cancer (non-skin) or melanoma among adults',
        'Coronary heart disease among adults',
        'Frequent mental distress among adults',
        'Lack of reliable transportation in the past 12 months among adults',
        'Chronic obstructive pulmonary disease among adults',
        'Mammography use among women aged 50-74 years',
        'Self-care disability among adults',
        'Independent living disability among adults',
        'Cognitive disability among adults',
        'Food insecurity in the past 12 months among adults',
        'Short sleep duration among adults',
        'Housing insecurity in the past 12 months among adults',
        'Colorectal cancer screening among adults aged 45–75 years',
        'Frequent physical distress among adults',
        'Lack of social and emotional support among adults',
        'Visited dentist or dental clinic in the past year among adults',
        'Loneliness among adults',
        'All teeth lost among adults aged >=65 years',
        'Utility services shut-off threat in the past 12 months among adults',
        'Current cigarette smoking among adults',
        'Cholesterol screening among adults',
        'Current asthma among adults',
        'Hearing disability among adults',
        'Vision disability among adults',
        'Binge drinking among adults',
        'Mobility disability among adults',
        'Fair or poor self-rated health status among adults'
    ]

    if 'Measure' not in df.columns:
        print("Warning: 'Measure' column not found in the dataset!")
        print(f"Available columns: {list(df.columns)}")
        return df

    original_count = len(df)

    df_filtered = df[~df['Measure'].isin(measures_to_remove)]

    removed_count = original_count - len(df_filtered)
    print(f"Removed {removed_count:,} rows with unwanted measures")
    print(f"Remaining rows: {len(df_filtered):,}")

    return df_filtered


def extract_coordinates(df):
    if 'Geolocation' not in df.columns:
        print("Warning: 'Geolocation' column not found in the dataset!")
        return df

    coordinates = df['Geolocation'].str.extract(r'POINT \(([^ ]+) ([^)]+)\)')

    df['longitude'] = pd.to_numeric(coordinates[0], errors='coerce')
    df['latitude'] = pd.to_numeric(coordinates[1], errors='coerce')

    valid_coords = df['longitude'].notna().sum()
    print(f"Extracted coordinates for {valid_coords:,} out of {len(df):,} rows")

    return df


def remove_low_confidence_rows(df):
    if 'High_Confidence_Limit' not in df.columns or 'Low_Confidence_Limit' not in df.columns:
        print("Warning: Confidence limit columns not found in dataset!")
        print("Required columns: 'High_Confidence_Limit', 'Low_Confidence_Limit'")
        return df

    max_high = df['High_Confidence_Limit'].max()
    max_low = df['Low_Confidence_Limit'].max()


    high_threshold = max_high * 0.10
    low_threshold = max_low * 0.10

    print(f"Max High_Confidence_Limit: {max_high:.2f}")
    print(f"10% threshold for High: {high_threshold:.2f}")
    print(f"Max Low_Confidence_Limit: {max_low:.2f}")
    print(f"10% threshold for Low: {low_threshold:.2f}")


    original_count = len(df)

    mask = ~((df['High_Confidence_Limit'] < high_threshold) &
             (df['Low_Confidence_Limit'] < low_threshold))

    df_filtered = df[mask].copy()


    removed_count = original_count - len(df_filtered)
    print(f"\nRows removed: {removed_count:,}")
    print(f"Rows remaining: {len(df_filtered):,}")
    print(f"Removed {removed_count / original_count * 100:.1f}% of rows")

    return df_filtered


def add_scores_simple(original_df, scored_df):
    print("\n📊 Adding deployment priority scores to dataset...")

    # Create a dictionary mapping county to deployment priority
    if 'CountyFIPS' in scored_df.columns:
        # Method 1: Using CountyFIPS
        priority_dict = scored_df.set_index('CountyFIPS')['deployment_priority'].to_dict()
        tier_dict = scored_df.set_index('CountyFIPS')['demand_tier'].to_dict()
        score_dict = scored_df.set_index('CountyFIPS')['demand_score_normalized'].to_dict()

        # Map to original dataframe
        original_df['deployment_priority'] = original_df['CountyFIPS'].map(priority_dict).fillna('LOW - Monitor only')
        original_df['demand_tier'] = original_df['CountyFIPS'].map(tier_dict).fillna('Low')
        original_df['demand_score_normalized'] = original_df['CountyFIPS'].map(score_dict).fillna(0)

        print(f"✅ Mapped deployment priorities for {len(original_df):,} rows using CountyFIPS")

    elif 'CountyName' in scored_df.columns and 'StateAbbr' in scored_df.columns:
        # Method 2: Using CountyName + StateAbbr
        scored_df['location_key'] = scored_df['CountyName'] + '_' + scored_df['StateAbbr']
        original_df['location_key'] = original_df['CountyName'] + '_' + original_df['StateAbbr']

        priority_dict = scored_df.set_index('location_key')['deployment_priority'].to_dict()
        tier_dict = scored_df.set_index('location_key')['demand_tier'].to_dict()
        score_dict = scored_df.set_index('location_key')['demand_score_normalized'].to_dict()

        original_df['deployment_priority'] = original_df['location_key'].map(priority_dict).fillna('LOW - Monitor only')
        original_df['demand_tier'] = original_df['location_key'].map(tier_dict).fillna('Low')
        original_df['demand_score_normalized'] = original_df['location_key'].map(score_dict).fillna(0)

        # Clean up temporary column
        original_df = original_df.drop(columns=['location_key'])

        print(f"✅ Mapped deployment priorities for {len(original_df):,} rows using CountyName+State")

    # Print distribution of deployment priorities
    print("\n📊 Deployment Priority Distribution:")
    priority_counts = original_df['deployment_priority'].value_counts()
    for priority, count in priority_counts.items():
        print(f"   {priority}: {count:,} rows ({count / len(original_df) * 100:.1f}%)")

    return original_df


def preprocess_data(dataset):
    print("Data Preprocessing Started🟢")
    print("removing unnecessary columns")
    dataset = remove_unnecessary_columns(dataset)
    print("removing unwanted measures")
    dataset = remove_unwanted_measures(dataset)
    print("Extracting coordinates")
    dataset = extract_coordinates(dataset)
    print("Removing low confidence rows")
    dataset = remove_low_confidence_rows(dataset)
    print("Scoring based on measure column")
    scored_df = calculate_weighted_demand_score_corrected(dataset)
    print("top locations needing sales reps")
    deployment_targets = get_deployment_priorities(scored_df, min_score=70)
    print(deployment_targets)
    print("Explain score for all counties")
    all_counties = explain_demand_score_detailed(scored_df)  # No location name specified
    print("Explain score for specific county")
    specific_county = explain_demand_score_detailed(scored_df, "Jefferson")
    print("Saving all counties data to csv")
    export_df = export_all_counties_data(scored_df, 'all_counties_demand_scores.csv')

    top_20 = scored_df.nlargest(20, 'demand_score_normalized')[
        ['CountyName', 'StateAbbr', 'demand_score_normalized', 'demand_tier', 'deployment_priority']
    ]
    print("\nTop 20 Counties:")
    print(top_20)

    dataset = add_scores_simple(dataset, scored_df)

    print("Data Preprocessing Completed🟢")

    # Return both the processed dataset and the scored dataframe
    return dataset, scored_df