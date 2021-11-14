# Kronos - Speech recognition
The speech recognition module for [Kronos - my personal virtual assistant proof of concept](https://github.com/vshotarov/Kronos).

## Table of Contents
<ol>
<li><a href="#overview">Overview</a></li>
<li><a href="#training">Training</a></li>
<li><a href="#data-augmentation">Data Augmentation</a></li>
</ol>

## Overview
The speech recognition module is responsible for turning voice recordings into text.

The way it is implemented is as a much smaller version of the [DeepSpeech](https://arxiv.org/pdf/1512.02595.pdf) architecture. Where DeepSpeech has ~39m parameters, this model has 899k.

The inputs to the network are **NOT** the raw audio waveform, but rather a processed version of it and that preprocessor lives in `data.py`. All it does is it converts the audio from an amplitude over time graph to a heatmap of frequencies over time using a [Mel Spectrogram](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.MelSpectrogram). 

The output of the network is a sequence of 28 (number of characters in the alphabet + the space character + a padding character) dimensional vectors of unnormalized log probabilities, but you can use the `STTModel.recognize()` method to return a transposed and normalized version of it, which can then be fed into the [`CTCBeamDecoder.decode()`](https://github.com/parlance/ctcdecode/blob/c90ad94a0b19554f80804fb7812f2a1447a34a70/ctcdecode/__init__.py#L53) method to perform beam search using a language model, returning the most probable interpretation of those characters.

During training we also randomly augment a percentage of the samples, in order to provide a bit more variety to the small dataset and help the network build resilience to unimportant variations. Have a look at the [data augmentation](#data-augmentation) section fore more info.

For an overview of how speech recognition fits in the full application have a look at [the main repo](https://github.com/vshotarov/Kronos#overview).

## Training
To train, run `train.py` with the relevant arguments.

```
usage: train.py [-h] [-ps PATH_TO_SAVE_MODEL] [-pw PATH_TO_WAV] [-ne NUM_EPOCHS]
                [-se SAVE_MODEL_EVERY] [-ve VALIDATE_EVERY] [-lrde LEARNING_RATE_DECAY_EVERY]
                [-lrdr LEARNING_RATE_DECAY_RATE]
                train_dataset validation_dataset test_dataset

Kronos virtual assistant - Speech recognition trainer

The dataset .csv files need to have the following columns:
   wav_filename wav_filesize transcript

positional arguments:
  train_dataset         path to the dataset .csv file for training
  validation_dataset    path to the dataset .csv file for validation during training
  test_dataset          path to the dataset .csv file for testing after training

optional arguments:
  -h, --help            show this help message and exit
  -ps PATH_TO_SAVE_MODEL, --path_to_save_model PATH_TO_SAVE_MODEL
                        path to save the trained model at. By default it's a file
                        calledsaved_model.torch in the current directory. NOTE: Intermediate versions
                        are also saved, at the same path but with the epoch appended to the name. To
                        control how often those are saved, look at `save_model_every`.
  -pw PATH_TO_WAV, --path_to_wav PATH_TO_WAV
                        path to the directory storing the .wav files specified in the datasets. By
                        default it's 'wav' directory in the current directory.
  -ne NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        how many epochs of training to run. By default it's 1250.
  -se SAVE_MODEL_EVERY, --save_model_every SAVE_MODEL_EVERY
                        how often save an intermediate version of the model. By default it's every
                        100.
  -ve VALIDATE_EVERY, --validate_every VALIDATE_EVERY
                        how often to validate in epochs. By default it's every 10.
  -lrde LEARNING_RATE_DECAY_EVERY, --learning_rate_decay_every LEARNING_RATE_DECAY_EVERY
                        how often to decay learning rate. By default it's every 15.
  -lrdr LEARNING_RATE_DECAY_RATE, --learning_rate_decay_rate LEARNING_RATE_DECAY_RATE
                        how much to decay learning rate. By default it's .99
```

Here's an example of what the dataset .csv files look like:

```
wav_filename,wav_filesize,transcript
sample_0.wav,32044,whats the weather like
sample_1.wav,32044,is it hot outside
sample_2.wav,32216,whats the time in london
sample_3.wav,32174,set a timer for five minutes
```

where the `wav_filename` columns contains names relative to the `-pw, --path_to_wav` argument.

## Data Augmentation
During training we perform some data augmentation in order to try and avoid overfitting and force the network to build resilience against unimportant variations.

The single most important process we do is normalize the gain, which massively improves the results by ignoring variations in how loud the utterance was spoken.

After that 40% of the time we perform some time stretching or pitch shifting.

Lastly after converting the waveform to a spectrogram, half of the time we perform some [time](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.TimeMasking) and [frequency masking](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.FrequencyMasking) to essentially damage the sample a bit, in order to force the network to build some resilience to recording issues.

