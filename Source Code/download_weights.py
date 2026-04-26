import gdown
import os

def download_file(url, output_path):
    print(f"Downloading from {url} to {output_path}...")
    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print(f"Successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    
    os.makedirs("./output/WWE-UIE/UIEB", exist_ok=True)
    os.makedirs("./utils/uranker", exist_ok=True)
    
    
    ckpt_url = 'https://drive.google.com/file/d/1c80kkDbgpLD1MtqVLql1Y1NQVD7IzxiO/view?usp=sharing'
    ckpt_path = './output/WWE-UIE/UIEB/best_model.pth'
    if not os.path.exists(ckpt_path):
        download_file(ckpt_url, ckpt_path)
    else:
        print(f"{ckpt_path} already exists. Skipping.")

    
    uranker_url = 'https://drive.google.com/file/d/1vBmD3ZvgVtz8xBTh3UmHGO72cAwlLG7G/view?usp=sharing'
    uranker_path = './utils/uranker/uranker.pth'
    if not os.path.exists(uranker_path):
        download_file(uranker_url, uranker_path)
    else:
        print(f"{uranker_path} already exists. Skipping.")
