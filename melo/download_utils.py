import os
import torch
from . import utils
from huggingface_hub import hf_hub_download

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}


def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")
        else:
            raise ValueError("Direct URL downloads are deprecated; use_hf must be True.")
    return utils.get_hparams_from_file(config_path)


def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        if use_hf:
            assert language in LANG_TO_HF_REPO_ID
            ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth")
        else:
            raise ValueError("Direct URL downloads are deprecated; use_hf must be True.")
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def load_pretrain_model(language='EN'):
    """
    Downloads the base Generator weights from Hugging Face.
    Discriminators (D and DUR) train from scratch during fine-tuning.
    """
    lang_key = language.upper()
    repo_id = LANG_TO_HF_REPO_ID.get(lang_key, 'myshell-ai/MeloTTS-English')
    
    print(f"Downloading base generator weights from Hugging Face repo: {repo_id}...")
    pretrain_G = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth")
    
    return pretrain_G, None, None