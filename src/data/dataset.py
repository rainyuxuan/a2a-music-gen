import csv
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pandas import DataFrame
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import os
import soundfile as sf
from custom_types import Waveform, SpectroData, Spectrogram


class WaveDataset(Dataset):
    """
    Dataset class for loading WAV files
    """

    __files: List[str] = []
    __metadata_files: List[str] = []
    __data_dir: str = ""
    # __split_ratio: float = 0.8
    __sr: int = 44100
    __max_sec: int = 60

    def __init__(
            self,
            data_dir: str,
            # split_ratio: float = 0.8,
            sr: int = 44100,
            max_sec: int = 60,
            transform=None,
    ):
        self.__data_dir = data_dir
        self.__files = os.listdir(data_dir)
        # self.__split_ratio = split_ratio
        self.__transform = transform
        self.__sr = sr
        self.__max_sec = max_sec

    def __len__(self) -> int:
        return len(self.__files)

    def __getitem__(self, idx: int) -> (Waveform, str):
        """
        Load wav file from wav file, split the file into x and y by split ratio
        :param idx: index of file to be loaded
        :return: (WaveformX, WaveformY, sampling rate)
        """
        data_path = os.path.join(self.__data_dir, self.__files[idx])
        wav_data, orig_sr = torchaudio.load(
            data_path, normalize=True
        )  # wav_data: (n_channels, n_samples)
        # Resample the data to the desired sampling rate
        wav_data = F.resample(wav_data, orig_sr, self.__sr)
        # Trim the data to the desired length
        wav_data = wav_data[:, : self.__max_sec * self.__sr]
        # Apply transform if available
        if self.__transform:
            wav_data = self.__transform(wav_data)
        # Split the file into x and y by split ratio
        wav_data = wav_data[0]  # Only use the first channel
        # x = wav_data[: int(len(wav_data) * self.__split_ratio)]
        # y = wav_data[int(len(wav_data) * self.__split_ratio):]
        fname = self.__files[idx].replace(".wav", "")
        return wav_data, fname

    def get_file_name(self, idx: int) -> str:
        return self.__files[idx]

    def get_sr(self) -> int:
        return self.__sr

    def get_max_sec(self) -> int:
        return self.__max_sec

    def get_split_ratio(self) -> float:
        return self.__split_ratio

    @staticmethod
    def save(wav_data: Waveform, sr: int, save_path: str, file_name: str) -> None:
        """
        Save wav to destination path with file name
        :param wav_data: waveform data in numpy form
        :param sr: sampling rate of wav file
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        wav_data = wav_data.unsqueeze(0)
        destination_path = os.path.join(save_path, f"{file_name}.wav")
        torchaudio.save(destination_path, wav_data, sr)

    @staticmethod
    def collate_fn(batch: List[Waveform]) -> Waveform:
        tensors, names = [], []
        for x, name in batch:
            tensors.append(x)
            names.append(name)
        tensors = pad_sequence(tensors, batch_first=True)
        return tensors, names


class SpectrogramDataset(Dataset):
    """
    Dataset class for loading Spectrogram files (np.array)
    """

    __files: List[str] = []
    __metadata_files: List[str] = []
    __data_dir: str = ""
    __label_dir: str = ""
    __metadata_dir: str = ""
    __sr: int = 0
    __split_ratio: float = 0.8

    def __init__(
            self,
            data_dir: str,
            transform=None,
            metadata_dir: str = None,
            split_ratio: float = 0.8,
    ):
        self.__data_dir = data_dir
        self.__metadata_dir = metadata_dir
        self.__split_ratio = split_ratio

        self.__files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        if metadata_dir:
            self.__metadata_files = [
                f for f in os.listdir(metadata_dir) if f.endswith(".csv")
            ]
        self.__transform = transform

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, idx: int) -> (Spectrogram, Spectrogram, str):
        """
        Load wav file from wav file store the wav_data and sampling rate
        :param idx: index of file to be loaded
        :return: (Spectrogram, Label, Metadata)
        """
        data_path = os.path.join(self.__data_dir, self.__files[idx])
        spectro: Spectrogram = torch.tensor(np.load(data_path, mmap_mode="r"))
        if self.__transform:
            spectro = self.__transform(spectro)
        # Split the file into x and y by split ratio
        if self.__split_ratio < 1.0:
            spectro, remain = spectro[:, : int(spectro.shape[1] * self.__split_ratio)], spectro[:, int(spectro.shape[1] * self.__split_ratio):]
        else:
            remain = []

        return spectro, remain, self.__files[idx].replace(".npy", "")

    def get_metadata(self, idx: int) -> Dict:
        metadata_path = os.path.join(self.__metadata_dir, self.__metadata_files[idx])
        with open(metadata_path) as csv_file:
            reader = csv.reader(csv_file)
            metadata: Dict = dict(reader)
        return metadata

    @staticmethod
    def save(
            spectro_data: Spectrogram,
            save_path: str,
            file_name: str
    ) -> None:
        """
        Save spectrogram to destination path with file name
        :param spectro_data: wav data in numpy form
        :param save_path: destination path
        :param file_name: name of file to be saved
        :param is_label: whether the spectrogram is a label
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.npy")
        np.save(destination_path, spectro_data)

    @staticmethod
    def save_metadata(metadata: Dict, save_path: str, file_name: str) -> None:
        """
        Save metadata to destination path with file name
        :param metadata: metadata in DataFrame form
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.csv")
        with open(destination_path, "w") as csv_file:
            writer = csv.writer(csv_file)
            for key, value in metadata.items():
                writer.writerow([key, value])

    @staticmethod
    def collate_fn(batch: List[SpectroData]) -> SpectroData:
        tensors, labels, names = [], [], []
        for x, y, name in batch:
            tensors.append(x)
            labels.append(y)
            names.append(name)
        tensors = pad_sequence(tensors, batch_first=True)
        labels = torch.stack(labels)
        names = torch.stack(names)
        return tensors, labels, names


def __test_wave_dataset():
    dataset = WaveDataset("data/raw/musicnet/train_data")
    dl = DataLoader(dataset, batch_size=2, shuffle=True)
    sr = dataset.get_sr()
    for x, y, fname in dl:
        print(fname, x.shape, y.shape, sr)
        break


def __test_spectrogram_dataset():
    dataset = SpectrogramDataset(
        "data/processed/musicnet/train_data"
    )
    dl = DataLoader(dataset, batch_size=2, shuffle=False)
    for xs, ys, fnames in dl:
        print(fnames, xs.shape, ys.shape)
        break


if __name__ == "__main__":
    __test_wave_dataset()
    __test_spectrogram_dataset()
