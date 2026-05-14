"""
Supply Chain Analysis — Part 2: Exploratory Data Analysis (EDA)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
from warnings import filterwarnings

filterwarnings('ignore')

# Styling 

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

viridis_colors   = cm.viridis(np.linspace(0, 1, 5))
primary_color    = viridis_colors[0]
secondary_color  = viridis_colors[1]
accent_color     = viridis_colors[2]
neutral_color    = viridis_colors[4]

# Load Data 

CLEAN_PATH = r'C:\Supply chain analysis\Data\clean_supply_chain.csv'

df = pd.read_csv(CLEAN_PATH)
df['order date (DateOrders)']    = pd.to_datetime(df['order date (DateOrders)'],    errors='coerce')
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# A.  KEY METRICS SUMMARY


def format_value(v):
    if v >= 1e6:  return f'{v/1e6:.1f}M'
    if v >= 1e3:  return f'{v/1e3:.1f}K'
    return f'{int(v):,}'

delayed_df = df[df['Delay'] > 0]

metrics = {
    'Total Orders'         : len(df),
    'Delayed Orders'       : len(delayed_df),
    '90th-Pct Delay (days)': round(delayed_df['Delay'].quantile(0.9), 2),
    'On-Time Delivery Rate': f"{(1 - len(delayed_df)/len(df))*100:.2f}%",
    'Late Delivery Rate'   : f"{len(delayed_df)/len(df)*100:.2f}%",
    'Total Profit'         : format_value(df.loc[df['Order Profit Per Order'] > 0, 'Order Profit Per Order'].sum()),
    'Loss from Delayed Ord': format_value(delayed_df['Order Profit Per Order'].sum()),
}

print("\n" + "="*50)
print("  KEY METRICS")
print("="*50)
for k, v in metrics.items():
    print(f"  {k:<28}: {v}")

# B.  PROFITABILITY & DELAY DISTRIBUTION


# --- Profitability pie chart ---
profit_pct = df['Profitability_flag'].value_counts(normalize=True) * 100
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(profit_pct, labels=profit_pct.index, autopct='%1.1f%%',
       colors=['lightgreen', 'lightcoral', 'lightblue'], startangle=140)
ax.set_title('Profitability Distribution')
plt.tight_layout()
plt.show()

# --- Delay distribution + profit by delay days ---
profit_metrics = (
    df.groupby('Delay')['Order Profit Per Order']
    .agg(mean_profit='mean', total_profit='sum', orders_count='count')
    .reset_index()
)

delay_dist = (
    df['Delay'].value_counts(normalize=True).sort_index() * 100
).reset_index()
delay_dist.columns = ['Delay_Days', 'Percentage']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x='Delay_Days', y='Percentage', data=delay_dist, color=accent_color, ax=ax1)
ax1.set_title('Delay Distribution')
ax1.set_xlabel('Delay (days)')
ax1.set_ylabel('Percentage of Orders (%)')
for bar in ax1.patches:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

ax2.bar(profit_metrics['Delay'], profit_metrics['total_profit'], color=primary_color, label='Total Profit')
ax2.set_ylabel('Total Profit', color=primary_color)
ax2.tick_params(axis='y', labelcolor=primary_color)

ax3 = ax2.twinx()
ax3.plot(profit_metrics['Delay'], profit_metrics['mean_profit'], marker='o', color=accent_color, label='Mean Profit')
ax3.set_ylabel('Mean Profit', color=accent_color)
ax3.tick_params(axis='y', labelcolor=accent_color)

def fmt_currency(val, _):
    if val >= 1e6:  return f'{val/1e6:.1f}M $'
    if val >= 1e3:  return f'{val/1e3:.1f}K $'
    return f'{val:.0f} $'

ax2.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_currency))
ax2.set_xlabel('Delay Days')
ax2.set_title('Profit vs Delay Days')

plt.tight_layout()
plt.show()


# C.  BOTTLENECK DETECTION


def bottleneck_detection(category):
    grp = df.groupby(category).agg(
        total_orders   = ('Delay',      'count'),
        delayed_orders = ('Is_Delayed', 'sum')
    ).reset_index()
    grp.rename(columns={category: 'category_name'}, inplace=True)
    grp['delay_rate'] = grp['delayed_orders'] / grp['total_orders'] * 100
    return grp.sort_values('delay_rate', ascending=False).head(10)

categories = ['Order Region', 'Customer Segment', 'Shipping Mode',
              'Order Status', 'Type', 'Department Name']

fig, axes = plt.subplots(3, 2, figsize=(18, 12), constrained_layout=True)
axes = axes.flatten()

for ax, cat in zip(axes, categories):
    data = bottleneck_detection(cat)
    sns.barplot(data=data, x='delay_rate', y='category_name', ax=ax, color=secondary_color)
    ax.set_title(f'Delay Rate by {cat}')
    ax.set_xlabel('Delay Rate (%)')
    ax.set_ylabel(cat)
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', ha='left', va='center', fontsize=8)

plt.suptitle('Bottleneck Detection by Operational Category', fontsize=14, y=1.02)
plt.show()


# D.  REGIONAL ROOT-CAUSE ANALYSIS


def analyze_region(region):
    df_region = df[df['Order Region'] == region].copy()
    drivers = ['Customer Segment', 'Shipping Mode', 'Order Status', 'Type', 'Department Name']
    all_metrics = []

    for factor in drivers:
        temp = df_region.groupby(factor).agg(
            total_orders   = ('Delay',      'count'),
            delayed_orders = ('Is_Delayed', 'sum'),
            avg_delay      = ('Delay',      'mean')
        ).reset_index()
        temp['delay_percentage'] = temp['delayed_orders'] / temp['total_orders'] * 100
        temp['Factor_level']     = factor + ': ' + temp[factor].astype(str)
        temp['driver']           = factor
        all_metrics.append(temp[['Factor_level', 'delay_percentage', 'avg_delay', 'driver', 'total_orders']])

    final_df    = pd.concat(all_metrics, ignore_index=True)
    top_factors = final_df.sort_values('delay_percentage', ascending=False).head(10)

    plt.figure(figsize=(12, 8))
    plt.barh(top_factors['Factor_level'], top_factors['delay_percentage'], color=primary_color)
    plt.title(f'Top Delay Drivers — {region} Region')
    plt.xlabel('Delay Percentage (%)')
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)

    for i, (_, row) in enumerate(top_factors.iterrows()):
        plt.text(row['delay_percentage'] - 10, i, f"{row['delay_percentage']:.1f}%",
                 fontsize=9, va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Run for all unique regions (or pick specific ones)
for region in df['Order Region'].unique():
    analyze_region(region)


# E.  TIME-BASED DELAY PATTERNS


def delay_pct(group_col):
    grp = df.groupby(group_col)['Is_Delayed'].mean().reset_index()
    grp['delay_percentage'] = grp['Is_Delayed'] * 100
    return grp

delay_by_month = delay_pct('order_month')
delay_by_day   = delay_pct('order_day')
delay_by_hour  = delay_pct('order_hour')

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

configs = [
    (delay_by_month, 'order_month', 'Month',        primary_color),
    (delay_by_day,   'order_day',   'Day of Week',   secondary_color),
    (delay_by_hour,  'order_hour',  'Hour of Day',   accent_color),
]

for ax, (data, x_col, label, color) in zip(axes, configs):
    sns.barplot(x=x_col, y='delay_percentage', data=data, ax=ax, color=color)
    ax.set_title(f'Delay % by {label}')
    ax.set_xlabel(label)
    ax.set_ylabel('Delay (%)')
    threshold = data['delay_percentage'].nlargest(3).iloc[-1]
    for bar in ax.patches:
        h = bar.get_height()
        if h >= threshold:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%',
                    ha='center', va='bottom', color='red', fontweight='bold', fontsize=8)

plt.suptitle('Time-Based Delay Patterns', fontsize=14)
plt.show()
