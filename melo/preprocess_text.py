import os
import json
from collections import defaultdict
from random import shuffle
from typing import Optional

import click
import torch
from tqdm import tqdm

try:
    from melo.text.cleaner import clean_text_bert
    from melo.text.symbols import symbols, num_languages, num_tones
except ModuleNotFoundError:
    from text.cleaner import clean_text_bert
    from text.symbols import symbols, num_languages, num_tones


@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4, type=int)
@click.option("--max-val-total", default=8, type=int)
@click.option("--clean/--no-clean", default=True)
@click.option("--device", default=None, help="Device for BERT feature extraction")
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: Optional[str],
    val_path: Optional[str],
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    device: Optional[str],
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadata_dir = os.path.dirname(metadata)
    if train_path is None:
        train_path = os.path.join(metadata_dir, 'train.list')
    if val_path is None:
        val_path = os.path.join(metadata_dir, 'val.list')
    out_config_path = os.path.join(metadata_dir, 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    if clean:
        new_symbols = []
        with open(metadata, "r", encoding="utf-8") as in_f, open(cleaned_path, "w", encoding="utf-8") as out_file:
            for line in tqdm(in_f):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    utt, spk, language, text = line_str.split("|")
                    norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device=device)

                    for ph in phones:
                        if ph not in symbols and ph not in new_symbols:
                            new_symbols.append(ph)
                            with open(f'{language}_symbol.txt', 'w', encoding='utf-8') as sf:
                                sf.write(str(new_symbols))

                    assert len(phones) == len(tones), f"Tone length mismatch in: {utt}"
                    assert len(phones) == sum(word2ph), f"Word2ph sum mismatch in: {utt}"

                    out_file.write(
                        "{}|{}|{}|{}|{}|{}|{}\n".format(
                            utt,
                            spk,
                            language,
                            norm_text,
                            " ".join(phones),
                            " ".join(str(i) for i in tones),
                            " ".join(str(i) for i in word2ph),
                        )
                    )

                    bert_path = utt.replace(".wav", ".bert.pt")
                    os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                    torch.save(bert.cpu(), bert_path)
                except Exception as error:
                    print(f"Error processing line: {line_str} -> {error}")

        metadata = cleaned_path

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("|")
            spk = parts[1]
            spk_utt_map[spk].append(line_str + "\n")

            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list.extend(utts[:val_per_spk])
        train_list.extend(utts[val_per_spk:])

    if len(val_list) > max_val_total:
        train_list.extend(val_list[max_val_total:])
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_list)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_list)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Preprocessing completed: {len(train_list)} train, {len(val_list)} val items.")


if __name__ == "__main__":
    main()