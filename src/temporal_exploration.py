#!/usr/bin/env python
"""
Temporal Analysis of Flickr Data - Explore photos through time dimension
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")

print("="*80)
print("TEMPORAL EXPLORATION OF FLICKR DATA")
print("="*80)

# ---- LOAD DATA ----
print("\n[1/5] Loading and reconstructing datetime...")
df = pd.read_csv("../flickr_data2.csv")
print(f"  • Total photos: {len(df):,}")

# Reconstruct datetime from separate year/month/day columns
year_col = ' date_taken_year'
month_col = ' date_taken_month'
day_col = ' date_taken_day'

if all(col in df.columns for col in [year_col, month_col, day_col]):
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Reconstruct datetime
    df['datetime_taken'] = pd.to_datetime(
        pd.DataFrame({
            'year': df['date_taken_year'],
            'month': df['date_taken_month'],
            'day': df['date_taken_day'],
        }),
        errors='coerce'
    )
    
    # Filter invalid dates (likely data quality issues in year field)
    df = df[(df['datetime_taken'] >= pd.Timestamp('1990-01-01')) & 
            (df['datetime_taken'] <= pd.Timestamp('2024-12-31'))].copy()
    
    print(f"  • Valid dates: {len(df):,} photos")
    print(f"  • Date range: {df['datetime_taken'].min()} to {df['datetime_taken'].max()}")
    
    date_col = 'datetime_taken'
else:
    print("  ⚠️  Date columns not found")
    exit(1)

# ---- TEMPORAL STATISTICS ----
print("\n[2/5] Computing temporal statistics...")

df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month
df['season'] = df['month'].apply(
    lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Autumn'
)

# Time span
date_range = df[date_col].max() - df[date_col].min()
print(f"  • Time span: {date_range.days} days ({date_range.days/365:.1f} years)")

# Photos per year
photos_per_year = df['year'].value_counts().sort_index()
print(f"\n  Photos per year:")
for year, count in photos_per_year.items():
    pct = 100 * count / len(df)
    bar = "█" * int(count / photos_per_year.max() * 40)
    print(f"    {int(year)}: {count:7,} photos ({pct:5.1f}%) {bar}")

# Photos per month
photos_per_month = df['month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print(f"\n  Photos per month (aggregated):")
for month, count in photos_per_month.items():
    pct = 100 * count / len(df)
    bar = "█" * int(count / photos_per_month.max() * 30)
    print(f"    {month_names[int(month)-1]:3s}: {count:7,} photos ({pct:5.1f}%) {bar}")

# Seasonal
photos_per_season = df['season'].value_counts()
print(f"\n  Seasonal distribution:")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    count = photos_per_season.get(season, 0)
    if count > 0:
        pct = 100 * count / len(df)
        bar = "█" * int(count / photos_per_season.max() * 40)
        print(f"    {season:10s}: {count:7,} photos ({pct:5.1f}%) {bar}")

# Peak days
print(f"\n  Peak activity days (top 10):")
df['date_only'] = df[date_col].dt.date
photos_per_day = df['date_only'].value_counts().head(10)
for idx, (date, count) in enumerate(photos_per_day.items(), 1):
    print(f"    {idx:2d}. {date}: {count:5,} photos")

# ---- TREND ANALYSIS ----
print("\n[3/5] Trend analysis...")

monthly_trend = df.groupby(df[date_col].dt.to_period('M')).size()
print(f"\n  Monthly trend:")
print(f"    • Total months: {len(monthly_trend)}")
print(f"    • Avg photos/month: {monthly_trend.mean():.0f}")
print(f"    • Max photos/month: {monthly_trend.max():.0f}")
print(f"    • Min photos/month: {monthly_trend.min():.0f}")

# Trend direction
if len(monthly_trend) > 1:
    first_third = monthly_trend.iloc[:len(monthly_trend)//3].mean()
    last_third = monthly_trend.iloc[-len(monthly_trend)//3:].mean()
    trend_direction = "↗ Increasing" if last_third > first_third else "↘ Decreasing"
    trend_pct = 100 * (last_third - first_third) / first_third
    print(f"    • Trend: {trend_direction} ({trend_pct:+.1f}%)")

# ---- VISUALIZATION ----
print("\n[4/5] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Temporal Analysis of Flickr Data (Lyon)', fontsize=16, fontweight='bold')

# 1. Monthly timeline
monthly_data = monthly_trend.astype(int)
axes[0, 0].plot(range(len(monthly_data)), monthly_data.values, color='steelblue', linewidth=2, marker='o', markersize=3)
axes[0, 0].set_title('Photos per Month Over Time')
axes[0, 0].set_xlabel('Month Index')
axes[0, 0].set_ylabel('Number of Photos')
axes[0, 0].grid(True, alpha=0.3)

# 2. Photos per month (histogram)
bars = axes[0, 1].bar(range(1, 13), photos_per_month.values, color='coral')
axes[0, 1].set_title('Photos by Month (Aggregated Across Years)')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Number of Photos')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Year distribution
photos_per_year.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Photos by Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Number of Photos')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Seasonal distribution
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
seasonal_data = [photos_per_season.get(s, 0) for s in season_order]
colors_season = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
axes[1, 1].pie(seasonal_data, labels=season_order, autopct='%1.1f%%', colors=colors_season, startangle=90)
axes[1, 1].set_title('Seasonal Distribution')

plt.tight_layout()
out_plot = 'outputs/temporal_analysis.png'
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {out_plot}")
plt.close()

# Heatmap of month-year
df_hm = df.groupby(['year', 'month']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(df_hm, cmap='YlOrRd', annot=False, fmt='d', ax=ax, cbar_kws={'label': 'Photos'})
ax.set_title('Photo Activity Heatmap (Year × Month)', fontsize=14, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_xticklabels(month_names)
plt.tight_layout()
out_heatmap = 'outputs/temporal_heatmap.png'
plt.savefig(out_heatmap, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {out_heatmap}")
plt.close()

# ---- SUMMARY ----
print("\n[5/5] Generating summary report...")

print("\n" + "="*80)
print("TEMPORAL EXPLORATION SUMMARY")
print("="*80)

print(f"\n📊 Key Findings:")
print(f"  • Dataset spans {date_range.days} days ({date_range.days/365.25:.1f} years)")
print(f"  • Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
print(f"  • Total photos: {len(df):,}")
print(f"  • Peak year: {int(photos_per_year.idxmax())} ({photos_per_year.max():,} photos)")
print(f"  • Peak month: {month_names[int(photos_per_month.idxmax())-1]} ({photos_per_month.max():,} photos)")
print(f"  • Most active season: {photos_per_season.idxmax()} ({photos_per_season.max():,} photos, {100*photos_per_season.max()/len(df):.1f}%)")
print(f"  • Daily average: {len(df)/date_range.days:.0f} photos/day")
print(f"  • Monthly average: {monthly_trend.mean():.0f} photos/month")

print(f"\n📈 Temporal Insights:")
print(f"  • Best month for photography: {month_names[int(photos_per_month.idxmax())-1]} (summer season)")
print(f"  • Least busy month: {month_names[int(photos_per_month.idxmin())-1]} (off-season)")
print(f"  • Peak year indicates high tourism/photography activity")

print(f"\n📁 Outputs generated:")
print(f"  ✅ outputs/temporal_analysis.png - 4-panel temporal visualization")
print(f"  ✅ outputs/temporal_heatmap.png - Year×Month activity heatmap")

print("\n" + "="*80)
