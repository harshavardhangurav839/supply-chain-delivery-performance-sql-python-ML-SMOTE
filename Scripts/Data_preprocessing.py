"""
Supply Chain Analysis — Part 1: Data Preprocessing

"""

import pandas as pd
import numpy as np
from warnings import filterwarnings

filterwarnings('ignore')

# 1. Load Data

DATA_PATH = r'C:\Supply chain analysis\Data\DataCoSupplyChainDataset.csv'
OUTPUT_PATH = r'C:\Supply chain analysis\Data\clean_supply_chain.csv'

df = pd.read_csv(DATA_PATH, encoding='latin-1')
print(f"Raw shape: {df.shape}")
print(f"\nMissing values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# 2. Drop Irrelevant Columns

COLUMNS_TO_DROP = [
    'Product Description', 'Product Image',
    'Customer Email', 'Customer Password',
    'Customer Fname', 'Customer Lname',
    'Customer Street', 'Customer Zipcode', 'Order Zipcode',
    'Longitude', 'Latitude',
    'Order Item Cardprod Id', 'Order Item Id',
    'Order Item Discount', 'Order Item Discount Rate',
    'Order Item Product Price', 'Order Item Quantity', 'Order Item Total',
    'Category Id', 'Department Id', 'Order Id',
    'Order Customer Id', 'Customer Id',
    'Product Card Id', 'Product Category Id',
    'Benefit per order', 'Product Status',
    'Customer City', 'Order City', 'Order Country', 'Order State',
    'Customer State', 'Market',
]

df.drop(columns=COLUMNS_TO_DROP, inplace=True)

# 3. Filter Out Cancelled Shipments

df = df[df['Delivery Status'] != 'Shipping cancelled'].copy()
print(f"\nShape after removing cancelled shipments: {df.shape}")

# 4. Parse Dates 

df['order date (DateOrders)']    = pd.to_datetime(df['order date (DateOrders)'],    errors='coerce')
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')

# 5. Feature Engineering 

# Delivery timing features
df['order_processing_time'] = (
    df['shipping date (DateOrders)'] - df['order date (DateOrders)']
).dt.days

df['Delay']      = df['order_processing_time'] - df['Days for shipment (scheduled)']
df['Is_Delayed'] = df['Delay'] > 0

# Calendar features
df['order_month'] = df['order date (DateOrders)'].dt.month
df['order_day']   = df['order date (DateOrders)'].dt.day_name()
df['order_hour']  = df['order date (DateOrders)'].dt.hour

# Profitability flag
df['Profitability_flag'] = np.where(
    df['Order Profit Per Order'] > 0, 'Profitable',
    np.where(df['Order Profit Per Order'] < 0, 'Not Profitable', 'Break Even')
)

# 6. Final Check

print(f"\nFinal shape : {df.shape}")
print(f"Columns     : {df.columns.tolist()}")
print(f"\nData types  :\n{df.dtypes}")
print(f"\nMissing values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
print(f"\nDelay distribution:\n{df['Is_Delayed'].value_counts()}")
print(f"\nProfitability distribution:\n{df['Profitability_flag'].value_counts(normalize=True).round(3)}")

# 7. Save Clean Data

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nClean dataset saved to: {OUTPUT_PATH}")
