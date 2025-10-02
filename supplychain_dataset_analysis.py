import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


# Step 1: Load and Explore the Data
def load_and_explore_data(file_path):
    """Load the dataset and perform initial exploration"""
    df = pd.read_csv(file_path)

    print("=" * 80)
    print("Step 1: Data Loading and Initial Exploration")
    print("=" * 80)
    print(f"\nDataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types and Missing Values:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe().transpose())
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
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Check for duplicates
    print(f"\nNumber of duplicates: {df.duplicated().sum()}")

    # Convert categorical columns to appropriate types
    categorical_cols = ['Product type', 'Customer demographics', 'Shipping carriers',
                        'Supplier name', 'Location', 'Inspection results',
                        'Transportation modes', 'Routes']

    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Create new features if needed
    df['Profit Margin'] = (df['Price'] - df['Manufacturing costs']) / df['Price']
    df['Inventory Turnover'] = df['Number of products sold'] / df['Stock levels']

    print("\nNew Features Created:")
    print("- Profit Margin: (Price - Manufacturing costs) / Price")
    print("- Inventory Turnover: Number of products sold / Stock levels")

    return df


# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """Perform exploratory data analysis and visualization"""
    print("\n" + "=" * 80)
    print("Step 3: Exploratory Data Analysis (EDA)")
    print("=" * 80)

    # Create directory for saving plots
    import os
    if not os.path.exists('ehdsaf_plots'):
        os.makedirs('ehdsaf_plots')

    # 1. Univariate Analysis
    print("\nUnivariate Analysis:")

    # Numeric features distribution
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols[:5]:  # Limit to first 5 for brevity
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'ehdsaf_plots/dist_{col}.png')
        plt.close()
        print(f"- Saved distribution plot for {col}")

    # Categorical features distribution
    categorical_cols = df.select_dtypes(include=['category']).columns
    for col in categorical_cols[:3]:  # Limit to first 3 for brevity
        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.savefig(f'ehdsaf_plots/count_{col}.png')
        plt.close()
        print(f"- Saved count plot for {col}")

    # 2. Bivariate Analysis
    print("\nBivariate Analysis:")

    # Product type vs Price
    plt.figure()
    sns.boxplot(data=df, x='Product type', y='Price')
    plt.title('Price Distribution by Product Type')
    plt.savefig('ehdsaf_plots/price_by_product_type.png')
    plt.close()
    print("- Saved price by product type plot")

    # Correlation matrix
    plt.figure()
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('ehdsaf_plots/correlation_matrix.png')
    plt.close()
    print("- Saved correlation matrix")

    # 3. Multivariate Analysis
    print("\nMultivariate Analysis:")

    # Pairplot for key numeric features (subset for performance)
    key_numeric = ['Price', 'Number of products sold', 'Revenue generated', 'Manufacturing costs', 'Defect rates']
    plt.figure()
    sns.pairplot(df[key_numeric])
    plt.suptitle('Pairplot of Key Numeric Features', y=1.02)
    plt.savefig('ehdsaf_plots/pairplot_key_features.png')
    plt.close()
    print("- Saved pairplot of key features")

    # Product type vs Revenue vs Defect rates
    plt.figure()
    sns.scatterplot(data=df, x='Revenue generated', y='Defect rates', hue='Product type')
    plt.title('Revenue vs Defect Rates by Product Type')
    plt.savefig('ehdsaf_plots/revenue_defect_product.png')
    plt.close()
    print("- Saved revenue vs defect rates by product type plot")


# Step 4: Hypothesis Testing
def perform_hypothesis_testing(df):
    """Perform hypothesis testing on key questions"""
    print("\n" + "=" * 80)
    print("Step 4: Hypothesis Testing")
    print("=" * 80)

    # Hypothesis 1: Different product types have different defect rates
    print("\nHypothesis 1: Different product types have different defect rates")
    product_types = df['Product type'].unique()
    for i in range(len(product_types)):
        for j in range(i + 1, len(product_types)):
            type1 = product_types[i]
            type2 = product_types[j]
            data1 = df[df['Product type'] == type1]['Defect rates']
            data2 = df[df['Product type'] == type2]['Defect rates']

            # Perform t-test
            t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
            print(f"\nT-test between {type1} and {type2}:")
            print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
            if p_val < 0.05:
                print(f"Significant difference in defect rates between {type1} and {type2}")
            else:
                print(f"No significant difference in defect rates between {type1} and {type2}")

    # Hypothesis 2: Higher manufacturing costs lead to higher prices
    print("\nHypothesis 2: Higher manufacturing costs lead to higher prices")
    corr_coef, p_val = stats.pearsonr(df['Manufacturing costs'], df['Price'])
    print(f"\nPearson correlation between Manufacturing costs and Price:")
    print(f"Correlation coefficient: {corr_coef:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        if corr_coef > 0:
            print("Significant positive correlation - higher manufacturing costs lead to higher prices")
        else:
            print("Significant negative correlation - higher manufacturing costs lead to lower prices")
    else:
        print("No significant correlation between manufacturing costs and prices")

    # Hypothesis 3: Different shipping carriers have different shipping times
    print("\nHypothesis 3: Different shipping carriers have different shipping times")
    carriers = df['Shipping carriers'].unique()
    carrier_data = [df[df['Shipping carriers'] == carrier]['Shipping times'] for carrier in carriers]

    # Perform ANOVA
    f_stat, p_val = stats.f_oneway(*carrier_data)
    print(f"\nANOVA results for shipping times across carriers:")
    print(f"F-statistic: {f_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Significant difference in shipping times across carriers")
    else:
        print("No significant difference in shipping times across carriers")


# Step 5: Statistical Analysis and Modeling
def perform_statistical_analysis(df):
    """Perform advanced statistical analysis and modeling"""
    print("\n" + "=" * 80)
    print("Step 5: Statistical Analysis and Modeling")
    print("=" * 80)

    # 1. Cluster Analysis
    print("\nCluster Analysis:")

    # Select features for clustering
    cluster_features = ['Price', 'Number of products sold', 'Revenue generated', 'Defect rates']
    cluster_data = df[cluster_features]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    # Determine optimal number of clusters using elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Cluster Number')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('ehdsaf_plots/elbow_method.png')
    plt.close()
    print("- Saved elbow method plot for cluster analysis")

    # Apply K-means with optimal clusters (assuming 3 based on elbow)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    df['PC1'] = principal_components[:, 0]
    df['PC2'] = principal_components[:, 1]

    plt.figure()
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis')
    plt.title('Cluster Visualization using PCA')
    plt.savefig('ehdsaf_plots/cluster_visualization.png')
    plt.close()
    print("- Saved cluster visualization plot")

    # Analyze cluster characteristics
    cluster_summary = df.groupby('Cluster')[cluster_features].mean()
    print("\nCluster Characteristics:")
    print(cluster_summary)

    # 2. Regression Analysis (example)
    print("\nRegression Analysis:")
    print("Predicting Revenue based on Price and Number of products sold")

    import statsmodels.api as sm
    X = df[['Price', 'Number of products sold']]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    y = df['Revenue generated']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Visualize regression results
    plt.figure()
    sns.regplot(x='Number of products sold', y='Revenue generated', data=df)
    plt.title('Regression: Revenue vs Products Sold')
    plt.savefig('ehdsaf_plots/revenue_regression.png')
    plt.close()
    print("- Saved regression plot")


# Step 6: Advanced Analytics and Insights
def perform_advanced_analytics(df):
    """Perform advanced analytics and generate insights"""
    print("\n" + "=" * 80)
    print("Step 6: Advanced Analytics and Insights")
    print("=" * 80)

    # 1. Supplier Performance Analysis
    print("\nSupplier Performance Analysis:")

    supplier_metrics = df.groupby('Supplier name').agg({
        'Defect rates': 'mean',
        'Lead time': 'mean',
        'Manufacturing costs': 'mean',
        'Revenue generated': 'sum'
    }).sort_values('Defect rates')

    print("\nSupplier Performance Metrics:")
    print(supplier_metrics)

    # Visualize supplier performance
    plt.figure(figsize=(12, 6))
    supplier_metrics[['Defect rates', 'Lead time']].plot(kind='bar', secondary_y='Lead time')
    plt.title('Supplier Performance: Defect Rates and Lead Times')
    plt.xticks(rotation=45)
    plt.savefig('ehdsaf_plots/supplier_performance.png')
    plt.close()
    print("- Saved supplier performance plot")

    # 2. Transportation Mode Analysis
    print("\nTransportation Mode Analysis:")

    transport_metrics = df.groupby('Transportation modes').agg({
        'Shipping times': 'mean',
        'Shipping costs': 'mean',
        'Defect rates': 'mean'
    }).sort_values('Shipping times')

    print("\nTransportation Mode Metrics:")
    print(transport_metrics)

    # Visualize transportation metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Transportation modes', y='Shipping costs', estimator=np.mean)
    plt.title('Average Shipping Costs by Transportation Mode')
    plt.savefig('ehdsaf_plots/transportation_costs.png')
    plt.close()
    print("- Saved transportation costs plot")

    # 3. Product Type Profitability
    print("\nProduct Type Profitability Analysis:")

    product_profit = df.groupby('Product type').agg({
        'Profit Margin': 'mean',
        'Revenue generated': 'sum',
        'Number of products sold': 'sum'
    }).sort_values('Profit Margin', ascending=False)

    print("\nProduct Type Profitability:")
    print(product_profit)

    # Visualize product profitability
    plt.figure(figsize=(10, 6))
    product_profit[['Profit Margin', 'Revenue generated']].plot(kind='bar', secondary_y='Revenue generated')
    plt.title('Product Type Profitability')
    plt.xticks(rotation=45)
    plt.savefig('ehdsaf_plots/product_profitability.png')
    plt.close()
    print("- Saved product profitability plot")


# Step 7: Generate Final Report
def generate_final_report(df):
    """Generate final report with key insights"""
    print("\n" + "=" * 80)
    print("Step 7: Final Report and Key Insights")
    print("=" * 80)

    # Create a text file with key insights
    with open('ehdsaf_plots/supply_chain_analysis_report.txt', 'w') as f:
        f.write("Supply Chain Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        # 1. Dataset Overview
        f.write("1. Dataset Overview:\n")
        f.write(f"- Total records: {len(df)}\n")
        f.write(f"- Product types: {df['Product type'].nunique()} ({', '.join(df['Product type'].unique())})\n")
        f.write(f"- Suppliers: {df['Supplier name'].nunique()}\n")
        f.write(f"- Locations: {df['Location'].nunique()} ({', '.join(df['Location'].unique())})\n\n")

        # 2. Key Findings
        f.write("2. Key Findings:\n")

        # Product Performance
        product_revenue = df.groupby('Product type')['Revenue generated'].sum().sort_values(ascending=False)
        f.write("\nProduct Performance by Revenue:\n")
        for product, revenue in product_revenue.items():
            f.write(f"- {product}: ${revenue:,.2f}\n")

        # Supplier Performance
        supplier_defects = df.groupby('Supplier name')['Defect rates'].mean().sort_values()
        f.write("\nSupplier Performance by Defect Rates (lower is better):\n")
        for supplier, defect_rate in supplier_defects.items():
            f.write(f"- {supplier}: {defect_rate:.2f}% defect rate\n")

        # Transportation Analysis
        transport_times = df.groupby('Transportation modes')['Shipping times'].mean().sort_values()
        f.write("\nTransportation Modes by Average Shipping Time (days):\n")
        for mode, time in transport_times.items():
            f.write(f"- {mode}: {time:.1f} days\n")

        # 3. Recommendations
        f.write("\n3. Recommendations:\n")
        f.write("- Focus on high-revenue products (skincare and haircare) for inventory optimization\n")
        f.write("- Work with suppliers showing lower defect rates to improve product quality\n")
        f.write("- Consider faster transportation modes for time-sensitive products despite higher costs\n")
        f.write("- Analyze clusters of products to identify patterns in pricing, sales, and defects\n")

    print("- Final report saved as 'supply_chain_analysis_report.txt'")


# Main function to execute all steps
def main():
    # Step 1: Load and explore data
    df = load_and_explore_data('supply_chain_data.csv')

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
    print("\nAll output files and plots saved in the 'ehdsaf_plots' directory.")


if __name__ == "__main__":
    main()