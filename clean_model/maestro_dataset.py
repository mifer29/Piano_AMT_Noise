import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
import torch.nn.functional as F
import os 
import math
from scipy.ndimage import convolve1d

class MAESTROSeq2SeqDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str,
        split: str,
        sample_rate: int = 16000,
        hop_length: int = 128,
        n_mels: int = 512,
        max_input_frames: int = 512,
        max_output_tokens: int = 1024,
        time_resolution: float = 0.01,
        logger=None,
        use_noisy: bool = False,
        p_clean: float = 0.5,

    ):
        super().__init__()

        self.hdf5_path = hdf5_path
        self.file = None  # lazy open per worker
        self.logger = logger
        self.split = split

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_input_frames = max_input_frames
        self.max_output_tokens = max_output_tokens
        self.time_resolution = time_resolution
        self.use_noisy = use_noisy
        self.p_clean = p_clean

        # Safely reading lengths without no persistent handle
        # The hdf5 files are opened temporarily, reading the two small arrays and closing immediately.
        # if dataset[1] is requested: we get mel_start = mel_offsets[1], mel_len   = mel_lengths[1], and reads it from the complete array: mel = f["mel"][mel_start : mel_start + mel_len]
        with h5py.File(self.hdf5_path, "r") as f:
            self.mel_lengths = f["mel_lengths"][:]
            self.event_lengths = f["event_lengths"][:]

        # Compute offsets
        self.mel_offsets = np.cumsum(
            np.insert(self.mel_lengths[:-1], 0, 0)
        )

        self.event_offsets = np.cumsum(
            np.insert(self.event_lengths[:-1], 0, 0)
        )

        self._build_vocab()
   
    # Lazy HDF5 opener (multiprocessing safe)

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")
        return self.file

    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except:
                pass

   
    # Vocabulary

    def _build_vocab(self):

        self.special_tokens = ["PAD", "BOS", "EOS"]

        # Absolute time tokens
        self.max_time_bins = 415
        self.time_tokens = [f"time_{i}" for i in range(self.max_time_bins)]

        self.velocity_tokens = [f"velocity_{i}" for i in range(128)]

        self.note_on_tokens  = [f"note_on_{i}" for i in range(128)]
        self.note_off_tokens = [f"note_off_{i}" for i in range(128)]

        self.vocab = (
            self.special_tokens +
            self.time_tokens +
            self.velocity_tokens +
            self.note_on_tokens +
            self.note_off_tokens
        )

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]

    
    def spec_augment(
        self,
        mel,
        num_time_masks=2,
        num_freq_masks=2,
        time_mask_size=30,
        freq_mask_size=30,
    ):
        """
        SpecAugment:
        - time masking (removes temporal regions)
        - frequency masking (removes frequency bands)

        mel: (T, n_mels)
        """
        
        T, F = mel.shape

        # Time masking
        for _ in range(num_time_masks):
            max_t = max(1, min(time_mask_size, T // 4))
            t = random.randint(1, max_t)
            t0 = random.randint(0, max(0, T - t))
            mel[t0:t0 + t, :] = 0.0

        # Frequency masking
        for _ in range(num_freq_masks):
            max_f = max(1, min(freq_mask_size, F // 4))
            f = random.randint(1, max_f)
            f0 = random.randint(0, max(0, F - f))
            mel[:, f0:f0 + f] = 0.0

        return mel


    def add_noise(self, mel):
        """Gaussian noise to simulate background noise in phone recordings."""
        noise_std = random.uniform(0.005, 0.02)
        return mel + np.random.randn(*mel.shape).astype(np.float32) * noise_std


    def add_reverb(self, mel):
        T, n_freq = mel.shape
        kernel_size = random.randint(3, 7)
        kernel = np.exp(-np.linspace(0, 2, kernel_size)).astype(np.float32)
        kernel = kernel / kernel.sum()

        # Pure numpy convolution along time axis — no torch overhead
        
        mel = convolve1d(mel, kernel, axis=0, mode='reflect')
        return mel[:T, :n_freq]


    def apply_augmentation(self, mel):
        """
        Augmentation pipeline for training only.
        Applied after mel loading, before returning the sample.

        Order matters:
        1. SpecAugment first: operates on clean mel
        2. Reverb: smears energy (room simulation)
        3. Noise: adds on top of reverbed signal

        Pitch shift removed: rolling mel bins doesn't shift pitch correctly
        on a log-scale mel spectrogram and doesn't shift tokens, so it was
        teaching the model wrong pitch mappings.
        """
        if random.random() < 0.5:
            mel = self.spec_augment(mel)

        if random.random() < 0.1:
            mel = self.add_reverb(mel)

        if random.random() < 0.2:
            mel = self.add_noise(mel)

        return mel

    # Dataset API

    def __len__(self):
        return len(self.mel_lengths)

    def __getitem__(self, idx):

        f = self._get_file()

        # Random crop on mel. 4.1 second segments are used before: hop_length/sample_rate = 0.008 s, max_input_frames * 0.008 ≈ 4.1 s
        # Now I stick to the paper, I randomly select a segment length each time.
        # Therefore step-based training will work well.

        mel_start = self.mel_offsets[idx] 
        mel_len = self.mel_lengths[idx]

        if self.split == "train":
            segment_frames = random.randint(64, self.max_input_frames)
        else:
            segment_frames = self.max_input_frames

        max_start = max(0, mel_len - segment_frames)
        start_frame = random.randint(0, max_start)

        if self.use_noisy and self.split == "train":
            if random.random() < self.p_clean:
                mel = f["mel_clean"][
                    mel_start + start_frame :
                    mel_start + start_frame + segment_frames
                ]
            else:
                mel = f["mel_noisy"][
                    mel_start + start_frame :
                    mel_start + start_frame + segment_frames
                ]
        else:
            # validation / test OR clean-only training
            mel = f["mel_clean"][
                mel_start + start_frame :
                mel_start + start_frame + segment_frames
            ]

        # if the piece is shorter than the proposed seconds then pad with zeroes. Every sample must be (512, 512)
        if mel.shape[0] < self.max_input_frames:
            pad = np.zeros(
                (self.max_input_frames - mel.shape[0], self.n_mels),
                dtype=np.float32
            )
            mel = np.vstack([mel, pad])

        # Expand the float16 saved spectograms into float32
        if mel.dtype != np.float32:
            mel = mel.astype(np.float32)

        mel = mel.copy()

        if self.split == "train":
            mel = self.apply_augmentation(mel)

        # Load events

        event_start = self.event_offsets[idx]
        event_len = self.event_lengths[idx]

        notes_array = f["events"][
            event_start : event_start + event_len
        ]

        # Each row looks like: [onset, offset, pitch, velocity]

        # Frame position to time
        start_time = (
            start_frame * self.hop_length / self.sample_rate
        )

        target_ids = self._events_to_tokens(
            notes_array,
            start_time,
            segment_frames
        )

        return {
            "audio_features": torch.from_numpy(mel),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }

    # Events to tokens

    def _events_to_tokens(self, notes_array, start_time, segment_frames):

        segment_duration = (
            segment_frames *
            self.hop_length /
            self.sample_rate
        )

        seg_end = start_time + segment_duration

        mask = np.logical_and(
            notes_array[:, 1] >= start_time,
            notes_array[:, 0] < seg_end
        )

        segment_notes = notes_array[mask]

        events = []

        for onset, offset, pitch, velocity in segment_notes:

            pitch = int(round(float(pitch)))
            velocity = int(round(float(velocity)))

            # Case 1: note fully inside the segment
            if onset >= start_time and offset <= seg_end:

                events.append(("on", onset - start_time, pitch, velocity))
                events.append(("off", offset - start_time, pitch, 0))

            # Case 2: note started before segment and ends inside segment
            elif onset < start_time and offset <= seg_end:

                events.append(("off", offset - start_time, pitch, 0))

            # Case 3: note started inside segment but ends after the segment
            elif onset >= start_time and offset > seg_end:

                events.append(("on", onset - start_time, pitch, velocity))

                # IMPORTANT: force note-off at segment boundary
                events.append(("off", segment_duration, pitch, 0))

            # Case 4: note spans the entire segment
            elif onset < start_time and offset > seg_end:

                # IMPORTANT: only forced note-off
                events.append(("off", segment_duration, pitch, 0))

        # Sort events properly
        events.sort(
            key=lambda x: (x[1], 0 if x[0] == "off" else 1)
        )

        tokens = []

        for event_type, event_time, pitch, velocity in events:

            time_bin = int(round(event_time / self.time_resolution))
            time_bin = min(time_bin, self.max_time_bins - 1)

            tokens.append(
                self.token_to_id[f"time_{time_bin}"]
            )

            if event_type == "on":
                tokens.append(self.token_to_id[f"note_on_{pitch}"])
                tokens.append(self.token_to_id[f"velocity_{velocity}"])
                
            else:
                tokens.append(self.token_to_id[f"note_off_{pitch}"])

        tokens.append(self.eos_id)

        if len(tokens) > self.max_output_tokens:
            tokens = tokens[:self.max_output_tokens - 1]
            tokens.append(self.eos_id)

        return tokens


# Collate function: it is used to combine multiple samples into a single batch.

def collate_fn(batch):

    audio = torch.stack(
        [item["audio_features"] for item in batch]
    )

    targets = [
        item["target_ids"]
        for item in batch
    ]

    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=0
    )

    return {
        "audio_features": audio,
        "target_ids": padded_targets
    }