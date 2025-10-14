"""
Setup script to create project structure
Run this after downloading the project files
"""

import os
from pathlib import Path

def create_project_structure():
    """Create necessary directories"""
    
    directories = [
        'data/raw',
        'data/processed',
        'src'
    ]
    
    print("Creating project structure...")
    print("=" * 50)
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    print("=" * 50)
    print("\n✅ Project structure created successfully!")
    print("\n📥 Next steps:")
    print("1. Download MovieLens 100K dataset from:")
    print("   https://grouplens.org/datasets/movielens/100k/")
    print("\n2. Place the following files in data/raw/:")
    print("   - u.data")
    print("   - u.item")
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n4. Run the application:")
    print("   streamlit run app.py")
    
    # Check if data files exist
    print("\n" + "=" * 50)
    print("Checking for data files...")
    print("=" * 50)
    
    if os.path.exists('data/raw/u.data'):
        print("✓ u.data found")
    else:
        print("✗ u.data not found - please download and place in data/raw/")
    
    if os.path.exists('data/raw/u.item'):
        print("✓ u.item found")
    else:
        print("✗ u.item not found - please download and place in data/raw/")
    
    # Check if source files exist
    print("\n" + "=" * 50)
    print("Checking for source files...")
    print("=" * 50)
    
    source_files = [
        'app.py',
        'src/data_processor.py',
        'src/collaborative_filtering.py',
        'src/matrix_factorization.py',
        'src/evaluator.py',
        'requirements.txt'
    ]
    
    all_present = True
    for file in source_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING!")
            all_present = False
    
    if all_present:
        print("\n✅ All source files present!")
    else:
        print("\n⚠️  Some source files are missing. Please ensure all files are present.")

if __name__ == "__main__":
    create_project_structure()