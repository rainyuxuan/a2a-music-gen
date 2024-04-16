import csv
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pandas import DataFrame
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
    __input_size: int = 256
    __output_size: int = 128
    __sr: int = 44100

    def __init__(
            self,
            data_dir: str,
            input_size: int = 256,
            output_size: int = 128,
            sr: int = 44100,
            transform=None,
    ):
        self.__data_dir = data_dir
        self.__files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.__input_size = input_size
        self.__output_size = output_size
        self.__transform = transform
        self.__sr = sr
        # self.__max_sec = max_sec

    def __len__(self) -> int:
        return len(self.__files)

    def __getitem__(self, idx: int) -> (Waveform, Waveform, str):
        """
        Load wav file from wav file, split the file into x and y by split ratio
        :param idx: index of file to be loaded
        :return: (WaveformX, WaveformY, sampling rate)
        """
        data_path = os.path.join(self.__data_dir, self.__files[idx])
        fname = self.__files[idx]
        wav_data, orig_sr = torchaudio.load(
            data_path, normalize=True
        )  # wav_data: (n_channels, n_samples)
        # Resample the data to the desired sampling rate
        wav_data = F.resample(wav_data, orig_sr, self.__sr)
        # Trim the data to the desired length
        # wav_data = wav_data[:, : self.__max_sec * self.__sr]
        # Apply transform if available
        if self.__transform:
            wav_data = self.__transform(wav_data)
        # Split the file into x and y by input and output size
        wav_data = wav_data[0]  # Only use the first channel
        wav_idx = random.randint(0, int(len(wav_data)-(self.__input_size+self.__output_size)))
        x = wav_data[wav_idx: wav_idx + self.__input_size]
        y = wav_data[wav_idx + self.__input_size: wav_idx + self.__input_size+self.__output_size]

        return x, y

    def get_file_name(self, idx: int) -> str:
        return self.__files[idx]

    def get_sr(self) -> int:
        return self.__sr

    # def get_max_sec(self) -> int:
    #     return self.__max_sec
    #
    # # def get_split_ratio(self) -> float:
    # #     return self.__split_ratio


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
        tensors, targets, names = [], [], []
        for x, y, name in batch:
            tensors.append(x)
            targets.append(y)
            names.append(name)
        tensors = pad_sequence(tensors, batch_first=True)
        targets = pad_sequence(targets, batch_first=True)
        return tensors, targets, names


class SpectrogramDataset(Dataset):
    """
    Dataset class for loading Spectrogram files (np.array)
    """

    __files: List[str] = []
    __label_files: List[str] = []
    __metadata_files: List[str] = []
    __data_dir: str = ""
    __input_length: int = 256   # along time axis of spectro
    __output_length: int = 128  # along time axis of spectro
    # __label_dir: str = ""
    __metadata_dir: str = ""
    __sr: int = 0

    def __init__(
            self,
            data_dir: str,
            transform=None,
            # label_dir: str = None,
            metadata_dir: str = None,
    ):
        self.__data_dir = data_dir

        self.__metadata_dir = metadata_dir

        self.__files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        # if label_dir:
        #     self.__label_files = [
        #         f for f in os.listdir(label_dir) if f.endswith(".npy")
        #     ]
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

        time_length = spectro.shape[1]
        spe_idx = random.randint(0, time_length-(self.__input_length + self.__output_length))
        input_spectro = spectro[:, spe_idx: spe_idx + self.__input_length]
        out_spectro = spectro[:, spe_idx + self.__input_length: spe_idx + self.__input_length+self.__output_length]

        return input_spectro, out_spectro, self.__files[idx]

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
            file_name: str,
            is_label: bool = False,
    ) -> None:
        """
        Save spectrogram to destination path with file name
        :param spectro_data: wav data in numpy form
        :param save_path: destination path
        :param file_name: name of file to be saved
        :param is_label: whether the spectrogram is a label
        :return: None
        """
        if is_label:
            destination_path = os.path.join(save_path, f"{file_name}.y.npy")
        else:
            destination_path = os.path.join(save_path, f"{file_name}.x.npy")
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
    dataset = WaveDataset("/Users/xunuo/Desktop/a2a-music-gen/data/raw/train_data")
    dl = DataLoader(dataset, batch_size=2, shuffle=True)
    print(dataset.__len__())
    sr = dataset.get_sr()
    i = 0
    for x, y, fname in dl:
        print(fname, x.shape, y.shape, sr)
        if i == 4:
            break


def __test_spectrogram_dataset():
    dataset = SpectrogramDataset(
        "/Users/xunuo/Desktop/a2a-music-gen/data/raw/train_data"
    )
    dl = DataLoader(dataset, batch_size=4, shuffle=False)
    print(dataset.__len__())
    for xs, ys, fnames in dl:
        print(fnames, xs.shape, ys.shape)


if __name__ == "__main__":
    __test_wave_dataset()
    # __test_spectrogram_dataset()
