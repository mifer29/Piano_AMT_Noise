# maestro_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random


class MAESTROSeq2SeqDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str,
        split: str,
        sample_rate: int = 16000,
        hop_length: int = 128,
        n_mels: int = 512,
        max_input_frames: int = 512,
        max_output_tokens: int = 512,
        time_resolution: float = 0.01,
        logger=None
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

        # Safely reading lengths without no persistent handle
        # The hdf5 files are opened temporarily, reading the two small arrays and closing immediately.
        # if dataset[1] is requested: we get mel_start = mel_offsets[1], mel_len   = mel_lengths[1], and reads it from the complete array: mel = f["mel"][mel_start : mel_start + mel_len]
        with h5py.File(self.hdf5_path, "r") as f:
            self.mel_lengths = f["mel_lengths"][:]
            self.event_lengths = f["event_lengths"][:]

        # Compute offsets
        self.mel_offsets = np.cumsum(
            np.insert(self.mel_lengths[:-1], 0, 0)  # if mel_lengths = [1200, 980, 1500], then offsets (start indeces) become: Piece 0 -> 0, Piece 1 -> 1200, Piece 2 -> 2180
        )

        self.event_offsets = np.cumsum(
            np.insert(self.event_lengths[:-1], 0, 0) # same methodology as in mel_offsets but using event lenghts
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

        self.max_time_bins = 6000
        self.time_tokens = [f"time_{i}" for i in range(self.max_time_bins)]
        self.velocity_tokens = [f"velocity_{i}" for i in range(128)]
        self.note_tokens = [f"note_{i}" for i in range(128)]

        self.vocab = (
            self.special_tokens +
            self.time_tokens +
            self.velocity_tokens +
            self.note_tokens
        )

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}

        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]

    # Dataset API

    def __len__(self):
        return len(self.mel_lengths)

    def __getitem__(self, idx):

        f = self._get_file()

        # Random crop on mel. 4.1 second segments are used: hop_length/sample_rate = 0.008 s, max_input_frames * 0.008 ≈ 4.1 s
        # Therefore step-based training will work well.

        mel_start = self.mel_offsets[idx] 
        mel_len = self.mel_lengths[idx]

        max_start = max(0, mel_len - self.max_input_frames) # this is the maximum safe start for segment. start_frame <= mel_len - 512 (512 is the max input frames)
        start_frame = random.randint(0, max_start)

        mel = f["mel"][
            mel_start + start_frame :
            mel_start + start_frame + self.max_input_frames
        ]

        # if the piece is shorter than 4 seconds then pad with zeroes. Every sample must be (512, 512)
        if mel.shape[0] < self.max_input_frames:
            pad = np.zeros(
                (self.max_input_frames - mel.shape[0], self.n_mels),
                dtype=np.float32
            )
            mel = np.vstack([mel, pad])

        # Expand the float16 saved spectograms into float32
        if mel.dtype != np.float32:
            mel = mel.astype(np.float32)

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
            start_time
        )

        return {
            "audio_features": torch.from_numpy(mel),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }

    # Events to tokens

    def _events_to_tokens(self, notes_array, start_time):

        segment_duration = (
            self.max_input_frames *
            self.hop_length /
            self.sample_rate
        )                       # around 4.1 s

        seg_end = start_time + segment_duration

        # Select notes that overlap the segment. offset >= segment start AND onset < segment end
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

                onset_bin = int(round(
                    (onset - start_time) / self.time_resolution
                ))

                offset_bin = int(round(
                    (offset - start_time) / self.time_resolution
                ))

                events.append((onset_bin, velocity, pitch))
                events.append((offset_bin, 0, pitch))

            # Case 2: note started before segment and ends inside segment
            elif onset < start_time and offset <= seg_end:

                offset_bin = int(round(
                    (offset - start_time) / self.time_resolution
                ))

                events.append((offset_bin, 0, pitch))

            # Case 3: note started inside segment but ends after the segment
            elif onset >= start_time and offset > seg_end:

                onset_bin = int(round(
                    (onset - start_time) / self.time_resolution
                ))

                events.append((onset_bin, velocity, pitch))

        # Sort events properly: time first, and note-off before note-on
        events.sort(
            key=lambda x: (x[0], 0 if x[1] == 0 else 1)
        )

        # Now converting the events into tokens from the vocabulary:
        tokens = []
        current_time = -1

        for time_bin, velocity, pitch in events:

            time_bin = min(
                time_bin,
                self.max_time_bins - 1
            )

            if time_bin != current_time:
                tokens.append(
                    self.token_to_id[f"time_{time_bin}"]
                )
                current_time = time_bin

            tokens.append(
                self.token_to_id[f"velocity_{velocity}"]
            )

            tokens.append(
                self.token_to_id[f"note_{pitch}"]
            )
        
        # Finally end of sentence token
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

    # Target arrays may have different length across samples, therefore we need to pad having as reference the longest target array.
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=0
    )

    return {
        "audio_features": audio,
        "target_ids": padded_targets
    }