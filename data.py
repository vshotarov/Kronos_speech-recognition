import torch
from torch import nn
import torchaudio
import string
import csv
import random
from ctcdecode import CTCBeamDecoder

LABELS = {letter:i+2 for i, letter in enumerate(string.ascii_lowercase)}
LABELS[' '] = 1
LABELS['_'] = 0
LABEL_INDICES = {v:k for k,v in LABELS.items()}

class Preprocessor(torchaudio.transforms.MelSpectrogram):
    def __init__(self):
        super(Preprocessor, self).__init__(n_mels=81, win_length=160, hop_length=80)

class LanaguageModelDecoder(CTCBeamDecoder):
    def __init__(self, language_model_path):
        super(LanaguageModelDecoder, self).__init__(
            [LABEL_INDICES[i] for i in range(len(LABEL_INDICES))],
            model_path=language_model_path,
            alpha=0.52272,
            beta=0.96505,
            log_probs_input=True
            )

    def decode(self, x, num_top_results=5):
        beam_results, _, _, out_lens = super(LanaguageModelDecoder, self).decode(x)
        top_results = []
        for j in range(5):
            top_results.append(
                "".join(LABEL_INDICES[n.item()] for n in beam_results[0][j][:out_lens[0][j]]))
        return top_results

class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()

        self.augment = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=70))

    def forward(self, x):
        if torch.rand(1, 1).item() > .5:
            return self.augment(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, validation=False):
        super(Dataset, self).__init__()

        self.validation = validation
        self.data_table = []
        with open(csv_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader):
                if i > 0:
                    self.data_table.append(row)

        self.preprocessor = Preprocessor()

        if validation:
            self.process = nn.Sequential(self.preprocessor)
        else:
            self.process = nn.Sequential(
                    self.preprocessor,
                    Augment())

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        audio, _bytes, transcript = self.data_table[idx]
        waveform, _ = torchaudio.load("resampled_audio/"+audio)

        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                [["gain", "-n"]])

        if not self.validation:
            if torch.rand(1,1) < .4:
                time_shift_percentage = [-1,1][random.randint(0,1)] * random.randint(200,600)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                        [["pitch", str(time_shift_percentage)]])
            elif torch.rand(1,1) < .4:
                pitch_shift_percentage = [-1,1][random.randint(0,1)] * random.randint(25,175)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                        [["pitch", str(pitch_shift_percentage)], ["rate", "8000"]])

        spectrogram = self.process(waveform)
        transcript_vector = [LABELS[x] for x in transcript]

        return spectrogram, transcript_vector, spectrogram.shape[-1] // 2, len(transcript)

def pad(device, data):
    spectrograms = []
    transcripts = []
    spectrogram_lengths = []
    transcript_lengths = []
    for (spectrogram, transcript, spectrogram_len, transcript_len) in data:
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1).to(device))
        transcripts.append(torch.Tensor(transcript).to(device))
        spectrogram_lengths.append(spectrogram_len)
        transcript_lengths.append(transcript_len)

    spectrograms = nn.utils.rnn.pad_sequence(
            spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    transcripts = nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

    return spectrograms, transcripts, spectrogram_lengths, transcript_lengths
