#!/usr/bin/env python3
# src/validate_demo.py
# Pre-demo validation script - checks all components are working

import os
import sys
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{END}")
    print(f"{BLUE}{text:^80}{END}")
    print(f"{BLUE}{'='*80}{END}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{END}")

def print_error(text):
    print(f"{RED}❌ {text}{END}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{END}")

def check_file_exists(path, name):
    """Check if file exists"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size > 0:
            print_success(f"{name}: {path} ({size:,} bytes)")
            return True
        else:
            print_error(f"{name}: {path} (empty file)")
            return False
    else:
        print_error(f"{name}: {path} (not found)")
        return False

def check_module(module_name, import_name=None):
    """Check if Python module can be imported"""
    if import_name is None:
        import_name = module_name
    try:
        __import__(import_name)
        print_success(f"Module '{module_name}' available")
        return True
    except ImportError:
        print_error(f"Module '{module_name}' not found")
        return False

def main():
    print_header("🎬 PRE-DEMO VALIDATION SCRIPT")
    
    # Change to src directory
    src_path = Path(__file__).parent
    os.chdir(src_path)
    sys.path.insert(0, str(src_path))
    
    all_passed = True
    
    # ========== 1. Check Python Version ==========
    print_header("1️⃣ Python Version Check")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 8:
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_error(f"Python {py_version.major}.{py_version.minor} (need 3.8+)")
        all_passed = False
    
    # ========== 2. Check Dependencies ==========
    print_header("2️⃣ Dependencies Check")
    required_modules = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('folium', 'folium'),
        ('wordcloud', 'wordcloud'),
        ('nltk', 'nltk'),
    ]
    
    for module_name, import_name in required_modules:
        if not check_module(module_name, import_name):
            all_passed = False
    
    # ========== 3. Check Data Files ==========
    print_header("3️⃣ Data Files Check")
    if check_file_exists("../flickr_data2.csv", "Flickr dataset"):
        # Quick validation: load data
        try:
            import pandas as pd
            df = pd.read_csv("../flickr_data2.csv", nrows=10)
            required_cols = ['lat', 'long', 'id', 'taken_dt', 'tags', 'title']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if not missing_cols:
                print_success(f"CSV has all required columns: {', '.join(required_cols)}")
            else:
                print_error(f"CSV missing columns: {', '.join(missing_cols)}")
                all_passed = False
        except Exception as e:
            print_error(f"Could not read CSV: {e}")
            all_passed = False
    else:
        all_passed = False
    
    # ========== 4. Check Source Modules ==========
    print_header("4️⃣ Source Modules Check")
    required_modules_src = [
        'load_data.py',
        'cleaning.py',
        'clustering.py',
        'text_mining.py',
        'comparison.py',
        'visualization.py',
        'session2_main.py',
    ]
    
    for module in required_modules_src:
        if check_file_exists(module, f"Source module: {module}"):
            # Try to import it
            try:
                module_name = module.replace('.py', '')
                __import__(module_name)
                print_success(f"  → {module_name} imports successfully")
            except Exception as e:
                print_warning(f"  → {module_name} import warning: {str(e)[:50]}")
        else:
            all_passed = False
    
    # ========== 5. Check Output Directory ==========
    print_header("5️⃣ Output Directory Check")
    output_path = Path("../outputs")
    if output_path.exists():
        print_success(f"Output directory exists: {output_path.absolute()}")
        
        # List existing files
        output_files = list(output_path.glob("*"))
        if output_files:
            print_success(f"Found {len(output_files)} existing output files:")
            for f in sorted(output_files)[:10]:
                size = f.stat().st_size if f.is_file() else "dir"
                print(f"  • {f.name}")
        else:
            print_warning("Output directory is empty (will be populated during demo)")
    else:
        print_warning(f"Output directory doesn't exist (will be created)")
    
    # ========== 6. Test Data Loading ==========
    print_header("6️⃣ Data Loading Test")
    try:
        from load_data import load_data, print_report
        print_success("Importing load_data module...")
        
        # Try to load data
        print("Loading data (this may take a moment)...")
        df_raw, rep_raw = load_data("../flickr_data2.csv")
        
        if len(df_raw) > 0:
            print_success(f"Data loaded: {len(df_raw):,} photos")
            print_success(f"Columns: {', '.join(df_raw.columns.tolist()[:5])}...")
        else:
            print_error("Data loaded but empty")
            all_passed = False
    except Exception as e:
        print_error(f"Data loading failed: {e}")
        all_passed = False
    
    # ========== 7. Test Data Cleaning ==========
    print_header("7️⃣ Data Cleaning Test")
    try:
        from cleaning import clean_data
        print_success("Importing cleaning module...")
        
        df_clean, rep_clean = clean_data(df_raw)
        print_success(f"Data cleaned: {len(df_clean):,} photos ({len(df_clean)/len(df_raw)*100:.1f}% retained)")
    except Exception as e:
        print_error(f"Data cleaning failed: {e}")
        all_passed = False
    
    # ========== 8. Test Clustering ==========
    print_header("8️⃣ Clustering Test")
    try:
        from clustering import run_dbscan_geo
        print_success("Importing clustering module...")
        
        print("Running DBSCAN (this may take a moment)...")
        df_clustered, rep_cluster = run_dbscan_geo(
            df_clean.head(5000),  # Test on smaller subset
            eps_meters=50.0,
            min_samples=50,
        )
        print_success(f"DBSCAN executed: {rep_cluster.n_clusters} clusters found (on sample)")
    except Exception as e:
        print_error(f"Clustering failed: {e}")
        all_passed = False
    
    # ========== 9. Test Text Mining ==========
    print_header("9️⃣ Text Mining Test")
    try:
        from text_mining import preprocess_text, extract_cluster_descriptions
        print_success("Importing text mining module...")
        
        if 'df_clustered' in locals():
            df_test = preprocess_text(df_clustered)
            print_success(f"Text preprocessing: {df_test['text'].notna().sum():,} rows with text")
    except Exception as e:
        print_error(f"Text mining failed: {e}")
        all_passed = False
    
    # ========== 10. Test Visualization ==========
    print_header("🔟 Visualization Test")
    try:
        from visualization import create_map, MapConfig
        print_success("Importing visualization module...")
        print_success("Folium mapping available")
    except Exception as e:
        print_error(f"Visualization import failed: {e}")
        all_passed = False
    
    # ========== 11. Check Notebooks ==========
    print_header("1️⃣1️⃣ Jupyter Notebooks Check")
    notebooks = [
        "../notebooks/02_session2_complete_analysis.ipynb",
        "../notebooks/03_final_demo.ipynb",
    ]
    
    for nb_path in notebooks:
        if check_file_exists(nb_path, f"Notebook: {Path(nb_path).name}"):
            pass
        else:
            all_passed = False
    
    # ========== Final Summary ==========
    print_header("🎯 VALIDATION SUMMARY")
    
    if all_passed:
        print_success("✅ ALL CHECKS PASSED - READY FOR DEMO!")
        print("\nNext steps:")
        print("  1. Run: python session2_main.py")
        print("  2. Open: jupyter notebook ../notebooks/03_final_demo.ipynb")
        print("  3. View: ../outputs/map_clusters_named.html")
        return 0
    else:
        print_error("❌ SOME CHECKS FAILED - SEE ERRORS ABOVE")
        print("\nTroubleshooting:")
        print("  • Install missing dependencies: pip install -r ../requirements.txt")
        print("  • Verify data file path: ../flickr_data2.csv")
        print("  • Check Python version: python --version (need 3.8+)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
