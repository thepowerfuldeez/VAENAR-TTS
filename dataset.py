import json
import math
import os

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
            self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.scaler = self.setup_scaler(preprocess_config['path']['stats_path'])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        # expects [80, mel_T]
        mel = self.normalize(mel.T, self.scaler).T

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
        }

        return sample

    def process_meta(self, filename):
        with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    "{}-mel-{}.npy".format(s, n),
                )
                mel = np.load(mel_path)
                if mel.shape[0] < 1030:
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
            return name, speaker, text, raw_text

    def setup_scaler(self, stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()
        # mel_mean, mel_std = torch.tensor(stats['mel_mean']), torch.tensor(stats['mel_std'])
        mel_mean, mel_std = np.array(stats['mel_mean']), np.array(stats['mel_std'])
        return StandardScaler(mel_mean, mel_std)

    def normalize(self, S, scaler=None):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        mel = scaler.transform(S.T).T
        return mel

    def denormalize(self, S, scaler=None, speaker=0):
        """denormalize values"""
        # pylint: disable=no-else-return
        S_denorm = S.clone()
        # mean-var scaling
        return scaler.inverse_transform(S_denorm.T).T

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size):]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.basename, self.speaker, self.text, self.raw_text = self.process_test_meta(
            filepath
        )

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def process_test_meta(self, filename):
        name = []
        speaker = []
        text = []
        raw_text = []
        lines = Path(filename).read_text().strip("\n").split("\n\n")
        for i, line in enumerate(lines, 1):
            t = line.strip("\n")
            name.append(str(i))
            speaker.append(list(self.speaker_map)[0])
            text.append(t)
            raw_text.append(t)
        return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


# @torch.jit.script
class StandardScaler:
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        self.mean_ = mean
        self.scale_ = scale

    def transform(self, x: torch.Tensor):
        x -= self.mean_#.to(x.device)
        x /= self.scale_#.to(x.device)
        return x

    def inverse_transform(self, x: torch.Tensor):
        x *= self.scale_#.to(x.device)
        x += self.mean_#.to(x.device)
        return x
