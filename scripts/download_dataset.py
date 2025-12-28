import os
import subprocess
import requests
from urllib.parse import urljoin
import zipfile
import shutil

def download_wlasl_dataset(subset='dev', data_dir='data/wlasl'):
    """
    Download WLASL dataset (World Level American Sign Language)
    
    Args:
        subset: 'dev', 'train', or 'test' (dev is smallest for initial testing)
        data_dir: Directory to store the dataset
    """
    base_url = "https://www.dropbox.com/s/"
    # WLASL dataset links (actual links from official source)
    # Note: These are example links; replace with actual WLASL download links
    # WLASL videos are available via their GitHub repo and separate download links
    
    # First, clone the metadata repository
    repo_url = "https://github.com/dxli94/WLASL.git"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Cloning WLASL metadata repository to {data_dir}...")
    try:
        subprocess.run(['git', 'clone', repo_url, data_dir], check=True)
        print("✅ Metadata repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone repository: {e}")
        return False
    
    # Download video files (this is a large download ~21GB for full dataset)
    # For development, download a subset
    videos_dir = os.path.join(data_dir, 'videos')
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    
    # Example: Download development set (smaller subset)
    if subset == 'dev':
        # WLASL dev set has ~1000 videos; actual download links need to be fetched from metadata
        print(f"Downloading {subset} subset videos...")
        # Parse XML metadata to get video URLs
        metadata_path = os.path.join(data_dir, 'WLASL_v0.3.xml')
        if os.path.exists(metadata_path):
            # This is a placeholder; in practice, parse XML and download videos
            print("📄 Metadata found. To download videos:")
            print("1. Parse WLASL_v0.3.xml for video URLs")
            print("2. Download videos using wget or requests")
            print("Example command for full download (requires actual links):")
            print("wget -r -np -nH --cut-dirs=3 -R 'index.html*' https://path/to/wlasl/videos/")
        else:
            print("⚠️ Metadata XML not found. Run 'git pull' in data/wlasl to update.")
    
    print("✅ WLASL dataset setup complete. Videos need to be downloaded separately.")
    print("Full dataset is ~21GB. Consider using Google Drive/Dropbox links from WLASL repo.")
    return True

def download_ms_asl_dataset(data_dir='data/ms_asl'):
    """
    Download MS-ASL dataset (Microsoft American Sign Language)
    
    Args:
        data_dir: Directory to store the dataset
    """
    # MS-ASL is available via Kaggle or direct links
    # Requires Kaggle API or manual download
    print("MS-ASL dataset requires Kaggle account or manual download.")
    print("1. Install Kaggle: pip install kaggle")
    print("2. Download: kaggle datasets download -d microsoft/ms-asl")
    print("3. Extract to data/ms_asl/")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Placeholder for automated download
    print("✅ MS-ASL setup directory created. Manual download required.")
    return True

if __name__ == "__main__":
    # Choose dataset
    dataset_choice = input("Choose dataset (1: WLASL, 2: MS-ASL): ")
    if dataset_choice == "1":
        download_wlasl_dataset(subset='dev')
    elif dataset_choice == "2":
        download_ms_asl_dataset()
    else:
        print("Invalid choice. Exiting.")
