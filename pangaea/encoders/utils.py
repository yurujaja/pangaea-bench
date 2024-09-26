import os
import urllib.error
import urllib.request

import gdown
import tqdm

# from . import prithvi_vit_base, spectral_gpt_vit_base, scale_mae_large, croma, remote_clip, ssl4eo_mae, ssl4eo_dino_small, ssl4eo_moco, ssl4eo_data2vec_small, dofa_vit, gfm_swin_base, satlasnet
# from . import SATLASNetWeights
# from . import adapt_gfm_pretrained
# from .pos_embed import interpolate_pos_embed


class DownloadProgressBar:
    def __init__(self, text="Downloading..."):
        self.pbar = None
        self.text = text

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(
                desc=self.text,
                total=total_size,
                unit="b",
                unit_scale=True,
                unit_divisor=1024,
            )

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def download_model(download_url=False, encoder_weights=False):
    if download_url:
        if not os.path.isfile(encoder_weights):
            # TODO: change this path
            os.makedirs("pretrained_models", exist_ok=True)

            pbar = DownloadProgressBar(f"Downloading {encoder_weights}")

            if download_url.startswith("https://drive.google.com/"):
                # Google drive needs some extra stuff compared to a simple file download
                gdown.download(download_url, encoder_weights)
            else:
                try:
                    urllib.request.urlretrieve(
                        download_url,
                        encoder_weights,
                        pbar,
                    )
                except urllib.error.HTTPError as e:
                    print(
                        "Error while downloading model: The server couldn't fulfill the request."
                    )
                    print("Error code: ", e.code)
                    return False
                except urllib.error.URLError as e:
                    print("Error while downloading model: Failed to reach a server.")
                    print("Reason: ", e.reason)
                    return False
        return True
    else:
        return False
