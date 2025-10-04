import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


# Step 1: Load and Explore the Data
def load_and_explore_data(file_path):
    """Load the dataset and perform initial exploration"""
    # Try different encodings if utf-8 fails
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Failed to read file with error: {e}")
            return None

    print("=" * 80)
    print("Step 1: Data Loading and Initial Exploration")
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types and Missing Values:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all').transpose())
    print("\nUnique Values per Column:")
    print(df.nunique())

    return df


# Step 2: Data Cleaning and Preprocessing
def clean_and_preprocess(df):
    """Clean and preprocess the data"""
    print("\n" + "=" * 80)
    print("Step 2: Data Cleaning and Preprocessing")
    print("=" * 80)

    # Check for missing values
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # Handle empty rows
    df = df.dropna(how='all')

    # Clean 'deaths' column - handle various formats
    def clean_deaths(value):
        if pd.isna(value):
            return np.nan
        # Handle non-string values
        if not isinstance(value, str):
            return float(value)
        # Remove commas, plus signs, and extract numbers
        cleaned = re.sub(r'[,\+\s]', '', str(value))
        nums = re.findall(r'\d+', cleaned)
        return float(nums[0]) if nums else np.nan

    df['deaths'] = df['deaths'].apply(clean_deaths)

    # Clean property damages - extract numeric values
    def clean_damage(value):
        if pd.isna(value):
            return np.nan
        # Handle non-string values
        if not isinstance(value, str):
            return float(value)
        # Find first number in the string (with optional decimal)
        match = re.search(r'(\d+\.?\d*)', value.replace(',', ''))
        return float(match.group(1)) if match else np.nan

    df['property_damages_numeric'] = df['property damages'].apply(clean_damage)

    # Handle date columns
    df['full_date'] = df.apply(
        lambda x: f"{x['year']}-{x['month']}-{int(x['date'])}"
        if pd.notna(x['date']) and str(x['date']).isdigit()
        else f"{x['year']}-{x['month']}",
        axis=1
    )

    # Clean cause column
    df['cause'] = df['cause'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

    # Create decade column for analysis
    df['decade'] = (df['year'] // 10) * 10

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """Perform exploratory data analysis and visualization"""
    print("\n" + "=" * 80)
    print("Step 3: Exploratory Data Analysis (EDA)")
    print("=" * 80)

    # Create directory for saving plots
    if not os.path.exists('flood_analysis_plots'):
        os.makedirs('flood_analysis_plots')

    # 1. Temporal Analysis
    print("\nTemporal Analysis:")

    # Floods by year
    plt.figure()
    df['year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Number of Flood Events by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/floods_by_year.png')
    plt.close()
    print("- Saved floods by year plot")

    # Floods by month
    plt.figure()
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
    df['month'].value_counts().sort_index().plot(kind='bar')
    plt.title('Number of Flood Events by Month')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/floods_by_month.png')
    plt.close()
    print("- Saved floods by month plot")

    # 2. Impact Analysis
    print("\nImpact Analysis:")

    # Deaths by year
    plt.figure()
    df.groupby('year')['deaths'].sum().plot(kind='bar')
    plt.title('Total Deaths by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/deaths_by_year.png')
    plt.close()
    print("- Saved deaths by year plot")

    # Property damage by year
    plt.figure()
    df.groupby('year')['property_damages_numeric'].sum().plot(kind='bar')
    plt.title('Property Damage by Year (Numeric Estimates)')
    plt.xlabel('Year')
    plt.ylabel('Damage Estimate')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/damage_by_year.png')
    plt.close()
    print("- Saved property damage by year plot")

    # 3. Location Analysis
    print("\nLocation Analysis:")

    # Top affected locations
    location_counts = df['location'].value_counts().head(10)
    plt.figure()
    location_counts.plot(kind='barh')
    plt.title('Top 10 Most Affected Locations')
    plt.xlabel('Number of Flood Events')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/top_locations.png')
    plt.close()
    print("- Saved top locations plot")

    # 4. Cause Analysis
    print("\nCause Analysis:")

    # Split causes and count occurrences
    all_causes = df['cause'].str.split(',').explode().str.strip()
    cause_counts = all_causes.value_counts().head(10)

    plt.figure()
    cause_counts.plot(kind='barh')
    plt.title('Top 10 Causes of Floods')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/top_causes.png')
    plt.close()
    print("- Saved top causes plot")


# Step 4: Hypothesis Testing
def perform_hypothesis_testing(df):
    """Perform hypothesis testing on key questions"""
    print("\n" + "=" * 80)
    print("Step 4: Hypothesis Testing")
    print("=" * 80)

    # Hypothesis 1: Monsoon months have significantly more deaths than non-monsoon months
    print("\nHypothesis 1: Monsoon months have significantly more deaths than non-monsoon months")
    monsoon_months = ['june', 'july', 'aug', 'sep']
    df['is_monsoon'] = df['month'].isin(monsoon_months)

    monsoon_deaths = df[df['is_monsoon']]['deaths'].dropna()
    non_monsoon_deaths = df[~df['is_monsoon']]['deaths'].dropna()

    # Perform t-test
    t_stat, p_val = stats.ttest_ind(monsoon_deaths, non_monsoon_deaths, equal_var=False)
    print(f"\nT-test between monsoon and non-monsoon months:")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant difference in deaths between monsoon and non-monsoon months")
        if monsoon_deaths.mean() > non_monsoon_deaths.mean():
            print("Monsoon months have significantly higher deaths")
        else:
            print("Non-monsoon months have significantly higher deaths")
    else:
        print("No significant difference in deaths between monsoon and non-monsoon months")

    # Visualize the comparison
    plt.figure()
    sns.boxplot(data=df, x='is_monsoon', y='deaths')
    plt.title('Deaths in Monsoon vs Non-Monsoon Months')
    plt.xticks([0, 1], ['Non-Monsoon', 'Monsoon'])
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/monsoon_deaths_comparison.png')
    plt.close()
    print("- Saved monsoon deaths comparison plot")

    # Hypothesis 2: More recent floods cause more property damage
    print("\nHypothesis 2: More recent floods cause more property damage")
    recent_years = df[df['year'] >= 2010]
    older_years = df[df['year'] < 2010]

    recent_damage = recent_years['property_damages_numeric'].dropna()
    older_damage = older_years['property_damages_numeric'].dropna()

    # Perform t-test
    t_stat, p_val = stats.ttest_ind(recent_damage, older_damage, equal_var=False)
    print(f"\nT-test between recent (>=2010) and older years:")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant difference in property damage between recent and older years")
        if recent_damage.mean() > older_damage.mean():
            print("Recent years have significantly higher property damage")
        else:
            print("Older years have significantly higher property damage")
    else:
        print("No significant difference in property damage between recent and older years")

    # Visualize the comparison
    plt.figure()
    sns.boxplot(data=df, x=df['year'] >= 2010, y='property_damages_numeric')
    plt.title('Property Damage in Recent vs Older Years')
    plt.xticks([0, 1], ['Before 2010', '2010 and After'])
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/damage_time_comparison.png')
    plt.close()
    print("- Saved property damage time comparison plot")


# Step 5: Statistical Analysis and Modeling
def perform_statistical_analysis(df):
    """Perform advanced statistical analysis"""
    print("\n" + "=" * 80)
    print("Step 5: Statistical Analysis and Modeling")
    print("=" * 80)

    # 1. Correlation Analysis
    print("\nCorrelation Analysis:")
    numeric_cols = ['year', 'deaths', 'property_damages_numeric']
    corr_matrix = df[numeric_cols].corr()

    plt.figure()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/correlation_matrix.png')
    plt.close()
    print("- Saved correlation matrix plot")

    # 2. Time Series Analysis
    print("\nTime Series Analysis:")

    # Deaths over time
    plt.figure()
    df.groupby('year')['deaths'].sum().plot(marker='o')
    plt.title('Flood Deaths Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Deaths')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/deaths_time_series.png')
    plt.close()
    print("- Saved deaths time series plot")

    # Property damage over time
    plt.figure()
    df.groupby('year')['property_damages_numeric'].sum().plot(marker='o')
    plt.title('Property Damage Over Time')
    plt.xlabel('Year')
    plt.ylabel('Damage Estimate')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/damage_time_series.png')
    plt.close()
    print("- Saved damage time series plot")

    # 3. Location Impact Analysis
    print("\nLocation Impact Analysis:")

    # Top locations by deaths
    top_locations = df.groupby('location')['deaths'].sum().nlargest(5)
    plt.figure()
    top_locations.plot(kind='barh')
    plt.title('Top 5 Locations by Total Deaths')
    plt.xlabel('Total Deaths')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/top_locations_deaths.png')
    plt.close()
    print("- Saved top locations by deaths plot")


# Step 6: Advanced Analytics and Insights
def perform_advanced_analytics(df):
    """Perform advanced analytics and generate insights"""
    print("\n" + "=" * 80)
    print("Step 6: Advanced Analytics and Insights")
    print("=" * 80)

    # 1. Decade Analysis
    print("\nDecade Analysis:")
    decade_stats = df.groupby('decade').agg({
        'deaths': 'sum',
        'property_damages_numeric': 'sum',
        'year': 'count'
    }).rename(columns={'year': 'flood_count'})

    print("\nFlood Statistics by Decade:")
    print(decade_stats)

    # Visualize decade analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].bar(decade_stats.index, decade_stats['flood_count'])
    axes[0].set_title('Flood Events by Decade')
    axes[0].set_xlabel('Decade')
    axes[0].set_ylabel('Count')

    axes[1].bar(decade_stats.index, decade_stats['deaths'])
    axes[1].set_title('Deaths by Decade')
    axes[1].set_xlabel('Decade')
    axes[1].set_ylabel('Total Deaths')

    axes[2].bar(decade_stats.index, decade_stats['property_damages_numeric'])
    axes[2].set_title('Property Damage by Decade')
    axes[2].set_xlabel('Decade')
    axes[2].set_ylabel('Damage Estimate')

    plt.tight_layout()
    plt.savefig('flood_analysis_plots/decade_analysis.png')
    plt.close()
    print("- Saved decade analysis plot")

    # 2. Cause Impact Analysis
    print("\nCause Impact Analysis:")

    # Create a cause-impact dataframe
    cause_impact = pd.DataFrame({
        'cause': df['cause'].str.split(',').explode().str.strip(),
        'deaths': df['deaths'].repeat(df['cause'].str.split(',').str.len()),
        'damage': df['property_damages_numeric'].repeat(df['cause'].str.split(',').str.len())
    })

    top_causes = cause_impact.groupby('cause').agg({
        'deaths': 'sum',
        'damage': 'sum'
    }).sort_values('deaths', ascending=False).head(5)

    print("\nTop 5 Causes by Impact:")
    print(top_causes)

    # Visualize cause impact
    plt.figure(figsize=(12, 6))
    top_causes.plot(kind='bar', secondary_y='damage')
    plt.title('Top 5 Causes by Deaths and Property Damage')
    plt.ylabel('Deaths')
    plt.tight_layout()
    plt.savefig('flood_analysis_plots/cause_impact.png')
    plt.close()
    print("- Saved cause impact plot")


# Step 7: Generate Final Report
def generate_final_report(df):
    """Generate final report with key insights"""
    print("\n" + "=" * 80)
    print("Step 7: Final Report and Key Insights")
    print("=" * 80)

    # Create a text file with key insights
    with open('flood_analysis_plots/flood_analysis_report.txt', 'w') as f:
        f.write("Pakistan Flood Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        # 1. Dataset Overview
        f.write("1. Dataset Overview:\n")
        f.write(f"- Total flood events recorded: {len(df)}\n")
        f.write(f"- Time period covered: {df['year'].min()} to {df['year'].max()}\n")
        f.write(f"- Locations affected: {df['location'].nunique()} distinct regions\n\n")

        # 2. Key Findings
        f.write("2. Key Findings:\n")

        # Temporal Patterns
        monsoon_months = ['june', 'july', 'aug', 'sep']
        monsoon_count = df[df['month'].isin(monsoon_months)].shape[0]
        f.write(f"\nTemporal Patterns:\n")
        f.write(
            f"- {monsoon_count} of {len(df)} flood events ({monsoon_count / len(df) * 100:.1f}%) occurred during monsoon months (June-September)\n")

        # Most Impactful Events
        top_deaths = df.nlargest(3, 'deaths')[['year', 'location', 'deaths']]
        f.write("\nMost Deadly Flood Events:\n")
        for _, row in top_deaths.iterrows():
            f.write(f"- {int(row['year'])} in {row['location']}: {int(row['deaths'])} deaths\n")

        # Location Analysis
        top_locations = df['location'].value_counts().head(3)
        f.write("\nMost Frequently Affected Regions:\n")
        for loc, count in top_locations.items():
            f.write(f"- {loc}: {count} flood events\n")

        # Cause Analysis
        all_causes = df['cause'].str.split(',').explode().str.strip()
        top_causes = all_causes.value_counts().head(3)
        f.write("\nMost Common Causes:\n")
        for cause, count in top_causes.items():
            f.write(f"- {cause}: {count} occurrences\n")

        # 3. Recommendations
        f.write("\n3. Recommendations:\n")
        f.write("- Strengthen flood preparedness and early warning systems before monsoon season\n")
        f.write("- Focus infrastructure improvements in frequently affected regions\n")
        f.write("- Address root causes like heavy rainfall and poor drainage systems\n")
        f.write("- Improve data collection on property damages for better impact assessment\n")

    print("- Final report saved as 'flood_analysis_report.txt'")


def main():
    # Step 1: Load and explore data
    df = load_and_explore_data('floods.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Clean and preprocess data
    df = clean_and_preprocess(df)

    # Step 3: Perform EDA
    perform_eda(df)

    # Step 4: Hypothesis testing
    perform_hypothesis_testing(df)

    # Step 5: Statistical analysis
    perform_statistical_analysis(df)

    # Step 6: Advanced analytics
    perform_advanced_analytics(df)

    # Step 7: Generate final report
    generate_final_report(df)

    print("\n" + "=" * 80)
    print("EHDSAF Analysis Completed Successfully!")
    print("=" * 80)
    print("\nAll output files and plots saved in the 'flood_analysis_plots' directory.")


if __name__ == "__main__":
    main()