import pandas as pd

# Data preparation for Excel file
# Table 1: Average Delivery Time and Logistics Costs by Warehouse
delivery_data = {
    "Warehouse Location": ["Karachi", "Lahore", "Islamabad", "Peshawar"],
    "Average Delivery Time (Days)": [5.8, 3.4, 4.1, 6.2],
    "Total Logistics Cost (USD)": [120000, 80000, 60000, 70000]
}

# Table 2: Sentiment Scores by Region
sentiment_data = {
    "Region": ["Karachi", "Lahore", "Islamabad", "Peshawar"],
    "Positive Sentiment (%)": [45, 60, 50, 40],
    "Neutral Sentiment (%)": [30, 25, 35, 40],
    "Negative Sentiment (%)": [25, 15, 15, 20]
}

# Table 3: Total Logistics Costs Across Regions
logistics_costs = {
    "Region": ["Karachi", "Lahore", "Islamabad", "Peshawar"],
    "Total Logistics Cost (USD)": [120000, 80000, 60000, 70000]
}

# Create DataFrames
df_delivery = pd.DataFrame(delivery_data)
df_sentiment = pd.DataFrame(sentiment_data)
df_logistics = pd.DataFrame(logistics_costs)

# Write to Excel
file_path = '/mnt/data/Supply_Chain_Analytics_Tables_and_Graphs.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_delivery.to_excel(writer, index=False, sheet_name="Delivery_Time_Logistics")
    df_sentiment.to_excel(writer, index=False, sheet_name="Sentiment_Scores")
    df_logistics.to_excel(writer, index=False, sheet_name="Logistics_Costs")
