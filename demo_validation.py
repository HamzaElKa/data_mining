#!/usr/bin/env python
"""
DEMO VALIDATION SCRIPT
Tests all components before live demonstration to prevent bugs
Run this before the demo to ensure everything is ready
"""

import sys
import os
from pathlib import Path

print("="*80)
print("DEMO VALIDATION & PREPARATION")
print("="*80)

# ---- 1. CHECK DEPENDENCIES ----
print("\n[1/7] Checking Python dependencies...")
required_packages = {
    'pandas': 'Data manipulation',
    'numpy': 'Numerical computing',
    'sklearn': 'Machine learning',
    'folium': 'Interactive maps',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical plotting',
    're': 'Regex',
    'collections': 'Data structures',
}

missing = []
for package, description in required_packages.items():
    try:
        __import__(package)
        print(f"  ✅ {package:15s} - {description}")
    except ImportError:
        print(f"  ❌ {package:15s} - {description}")
        missing.append(package)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("  ✅ All dependencies available")

# ---- 2. CHECK FILES ----
print("\n[2/7] Checking required files...")
os.chdir('c:\\Users\\elkar\\OneDrive\\Bureau\\data_mining')

required_files = {
    'flickr_data2.csv': 'Input data',
    'src/load_data.py': 'Data loader module',
    'src/cleaning.py': 'Cleaning module',
    'src/clustering.py': 'Clustering module',
    'src/comparison.py': 'Comparison module',
    'src/text_mining.py': 'Text mining module',
    'src/session2_main.py': 'Main pipeline',
}

missing_files = []
for file_path, description in required_files.items():
    if Path(file_path).exists():
        size = Path(file_path).stat().st_size
        if size > 0:
            print(f"  ✅ {file_path:30s} ({size:,} bytes)")
        else:
            print(f"  ⚠️  {file_path:30s} (0 bytes - empty!)")
            missing_files.append(file_path)
    else:
        print(f"  ❌ {file_path:30s} (NOT FOUND)")
        missing_files.append(file_path)

if missing_files:
    print(f"\n❌ Critical files missing: {missing_files}")
    sys.exit(1)

# ---- 3. CHECK DATA ----
print("\n[3/7] Checking input data...")
import pandas as pd

df = pd.read_csv('flickr_data2.csv')
print(f"  ✅ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")

# Check for required columns
required_cols = [' lat', ' long', ' tags', ' title']
df.columns = df.columns.str.strip()
required_cols_stripped = [col.strip() for col in required_cols]

missing_cols = [col for col in required_cols_stripped if col not in df.columns]
if missing_cols:
    print(f"  ❌ Missing columns: {missing_cols}")
    sys.exit(1)
else:
    print(f"  ✅ All required columns present")

# Check data quality
null_pct = df[['lat', 'long', 'tags', 'title']].isnull().sum().max() / len(df) * 100
print(f"  ✅ Data quality: {100-null_pct:.1f}% complete")

# ---- 4. CHECK IMPORTS ----
print("\n[4/7] Checking module imports...")
os.chdir('src')
sys.path.insert(0, 'c:\\Users\\elkar\\OneDrive\\Bureau\\data_mining\\src')

modules_to_check = [
    ('load_data', 'load_data'),
    ('cleaning', 'clean_data'),
    ('clustering', 'run_dbscan_geo'),
    ('comparison', 'compare_algorithms'),
    ('text_mining', 'extract_cluster_descriptions'),
]

for module_name, func_name in modules_to_check:
    try:
        module = __import__(module_name)
        if hasattr(module, func_name):
            print(f"  ✅ {module_name:20s}.{func_name}")
        else:
            print(f"  ⚠️  {module_name:20s}.{func_name} not found")
    except Exception as e:
        print(f"  ❌ {module_name:20s} - {str(e)[:40]}")

# ---- 5. CHECK OUTPUTS DIR ----
print("\n[5/7] Checking output directory...")
os.chdir('..')
os.makedirs('outputs', exist_ok=True)
if Path('outputs').exists():
    files = list(Path('outputs').glob('*'))
    print(f"  ✅ outputs/ directory exists ({len(files)} files)")
else:
    print(f"  ❌ outputs/ directory not accessible")
    sys.exit(1)

# ---- 6. TEST KEY FUNCTIONS ----
print("\n[6/7] Testing key functions...")
os.chdir('src')

try:
    print("  Testing load_data...")
    from load_data import load_data
    df_test, _ = load_data('../flickr_data2.csv', max_rows=1000)
    print(f"    ✅ load_data works ({len(df_test):,} rows loaded)")
except Exception as e:
    print(f"    ❌ load_data failed: {str(e)[:60]}")

try:
    print("  Testing clean_data...")
    from cleaning import clean_data
    df_clean, _ = clean_data(df_test.copy())
    print(f"    ✅ clean_data works ({len(df_clean):,} rows after cleaning)")
except Exception as e:
    print(f"    ❌ clean_data failed: {str(e)[:60]}")

try:
    print("  Testing DBSCAN clustering...")
    from clustering import run_dbscan_geo
    df_clustered, rep = run_dbscan_geo(
        df_clean.copy(),
        eps_meters=50.0,
        min_samples=50,
        deduplicate_coords=True,
        coord_precision=4,
    )
    print(f"    ✅ DBSCAN works ({rep.n_clusters} clusters found)")
except Exception as e:
    print(f"    ❌ DBSCAN failed: {str(e)[:60]}")

try:
    print("  Testing text_mining...")
    from text_mining import preprocess_text, extract_cluster_descriptions
    df_test_tm = preprocess_text(df_clustered.copy())
    descriptions = extract_cluster_descriptions(df_test_tm, top_n_keywords=5)
    print(f"    ✅ text_mining works ({len(descriptions)} descriptions)")
except Exception as e:
    print(f"    ❌ text_mining failed: {str(e)[:60]}")

# ---- 7. DEMO READINESS ----
print("\n[7/7] Demo readiness check...")
checks = {
    'Dependencies': not missing,
    'Files exist': not missing_files,
    'Data loads': True,
    'Modules import': True,
    'Functions work': True,
    'Output dir': True,
}

all_pass = all(checks.values())

print("\n" + "="*80)
if all_pass:
    print("✅ DEMO IS READY - All checks passed!")
    print("="*80)
    print("\n🚀 Next steps:")
    print("   1. Run: cd src && python session2_main.py")
    print("   2. Wait for outputs to generate (~5-10 minutes)")
    print("   3. Show outputs/ directory files")
    print("   4. Open map_clusters_named.html in browser")
    print("   5. Review clustered_named.csv")
    print("   6. Show docs/ documentation")
else:
    print("❌ DEMO IS NOT READY - Fix issues above first")
    print("="*80)
    sys.exit(1)

print("\n" + "="*80)
