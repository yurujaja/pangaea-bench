import os
import tqdm
import urllib.request
import urllib.error
import gdown


class DownloadProgressBar:
    def __init__(self, text="Downloading..."):
        self.pbar = None
        self.text = text
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(desc=self.text, total=total_size, unit="b", unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def download_model(model_config):
    if "download_url" in model_config:
        if not os.path.isfile(model_config["encoder_weights"]):
            os.makedirs("pretrained_models", exist_ok=True)

            pbar = DownloadProgressBar(f"Downloading {model_config['encoder_weights']}")

            if model_config["download_url"].startswith("https://drive.google.com/"):
                # Google drive needs some extra stuff compared to a simple file download
                gdown.download(model_config["download_url"], model_config["encoder_weights"])
            else:
                try:
                    urllib.request.urlretrieve(model_config["download_url"], model_config["encoder_weights"], pbar)
                except urllib.error.HTTPError as e:
                    print('Error while downloading model: The server couldn\'t fulfill the request.')
                    print('Error code: ', e.code)
                    return
                except urllib.error.URLError as e:
                    print('Error while downloading model: Failed to reach a server.')
                    print('Reason: ', e.reason)
                    return
