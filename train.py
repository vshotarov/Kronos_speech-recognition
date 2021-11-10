import torch
from torch import nn
from model import STTModel
from data import Dataset, LABEL_INDICES, LABELS, pad
from functools import partial
import argparse
import textwrap


def train(train_dataset, validation_dataset, test_dataset,
        path_to_save_model, path_to_wav, num_epochs, save_model_every,
        validate_every, learning_rate_decay_every, learning_rate_decay_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = STTModel().to(device)
    model.train()
    loader = torch.utils.data.DataLoader(Dataset(
        train_dataset, wav_files_path=path_to_wav),
        batch_size=8, collate_fn=partial(pad, device), shuffle=True)
    val_loader = torch.utils.data.DataLoader(Dataset(
        validation_dataset, validation=True, wav_files_path=path_to_wav),
        batch_size=16, collate_fn=partial(pad, device), shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(
        test_dataset, validation=True, wav_files_path=path_to_wav),
        batch_size=1, collate_fn=partial(pad, device))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

    best_val_loss = float("inf")
    best_val_loss_epoch = -1

    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                spectrograms, transcripts, spectrogram_lens, transcript_lens = data
                spectrogram_lens = torch.tensor(spectrogram_lens).to(device)
                transcript_lens = torch.tensor(transcript_lens).to(device)

                optimizer.zero_grad()

                hn, c0 = [x.to(device) for x in model.get_initial_hidden(len(transcripts))]
                outputs, _ = model(spectrograms, (hn, c0))
                #outputs = torch.nn.functional.log_softmax(outputs, dim=2)

                loss = criterion(outputs, transcripts, spectrogram_lens, transcript_lens)
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()

                print("epoch %i [ %i / %i ]: running_loss %f\r" % (epoch+1, i,
                    len(loader.dataset) / loader.batch_size, running_loss / (i+1)), end="")
            print()

            if epoch % learning_rate_decay_every == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= learning_rate_decay_rate
                    lr = g["lr"]
                print("lr is now", lr)

            if epoch % validate_every == 0:
                with torch.no_grad():
                    val_running_loss = 0.0
                    for i, data in enumerate(val_loader, 0):
                        if i >= 0: # always
                            spectrograms, transcripts, spectrogram_lens, transcript_lens = data
                            spectrogram_lens = torch.tensor(spectrogram_lens).to(device)
                            transcript_lens = torch.tensor(transcript_lens).to(device)

                            optimizer.zero_grad()

                            hn, c0 = [x.to(device) for x in model.get_initial_hidden(len(transcripts))]
                            outputs, _ = model(spectrograms, (hn, c0))
                            outputs = torch.nn.functional.log_softmax(outputs, dim=2)

                            val_loss = torch.nn.functional.ctc_loss(
                                    outputs, transcripts, spectrogram_lens, transcript_lens)

                            val_running_loss += val_loss.item()

                            outputs = outputs.transpose(0,1)
                            arg_maxes = torch.argmax(outputs[0], dim=1)

                            print(''.join([LABEL_INDICES[int(x)] for x in transcripts[0]]))
                            print(''.join([LABEL_INDICES[int(x)] for x in arg_maxes if x != 0]))
                            print('val_loss: ', val_running_loss / (i+1))

                    if val_running_loss / (i+1) < best_val_loss:
                        best_val_loss = val_running_loss / (i+1)
                        best_val_loss_epoch = epoch
                        torch.save(model, path_to_save_model)
                        print("saved model to", path_to_save_model)

            if epoch % save_model_every == 0:
                torch.save(model, path_to_save_model + "." + str(epoch))
                print("saved model to", path_to_save_model + "." + str(epoch))

    except KeyboardInterrupt:
        # We assume that a keyboard interupt means stop training, but still carry on to test
        pass

    with torch.no_grad():
        test_running_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            if i >= 0: # always
                print("#" * 15)
                print("test", i)
                spectrograms, transcripts, spectrogram_lens, transcript_lens = data
                spectrogram_lens = torch.tensor(spectrogram_lens).to(device)
                transcript_lens = torch.tensor(transcript_lens).to(device)

                optimizer.zero_grad()

                hn, c0 = [x.to(device) for x in model.get_initial_hidden(len(transcripts))]
                outputs, _ = model(spectrograms, (hn, c0))
                outputs = torch.nn.functional.log_softmax(outputs, dim=2)

                test_loss = torch.nn.functional.ctc_loss(
                        outputs, transcripts, spectrogram_lens, transcript_lens)

                test_running_loss += test_loss.item()

                outputs = outputs.transpose(0,1)
                arg_maxes = torch.argmax(outputs[0], dim=1)

                print(''.join([LABEL_INDICES[int(x)] for x in transcripts[0]]))
                print(''.join([LABEL_INDICES[int(x)] for x in arg_maxes if x != 0]))
                print('test_loss: ', test_running_loss / (i+1))

    print("best_val_loss %f, achieved at epoch %i" % (best_val_loss, best_val_loss_epoch))

class RawHelpFormatter(argparse.HelpFormatter):
    # https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    def _fill_text(self, text, width, indent):
                return "\n".join([textwrap.fill(line, width) for line in\
                    textwrap.indent(textwrap.dedent(text), indent).splitlines()])

if __name__ == "__main__":
    help_text = """
Kronos virtual assistant - Speech recognition trainer

The dataset .csv files need to have the following columns:
   wav_filename wav_filesize transcript
"""

    parser = argparse.ArgumentParser(
        description=help_text, formatter_class=RawHelpFormatter)
    parser.add_argument("train_dataset",
        help="path to the dataset .csv file for training")
    parser.add_argument("validation_dataset",
        help="path to the dataset .csv file for validation during training")
    parser.add_argument("test_dataset",
        help="path to the dataset .csv file for testing after training")
    parser.add_argument("-ps", "--path_to_save_model", default="saved_model.torch",
        help="path to save the trained model at. By default it's a file called"
             "saved_model.torch in the current directory. NOTE: Intermediate "
             "versions are also saved, at the same path but with the epoch appended "
             "to the name. To control how often those are saved, look at `save_model_every`.")
    parser.add_argument("-pw", "--path_to_wav", default="wav",
        help="path to the directory storing the .wav files specified in the datasets. By default it's 'wav' directory in the current directory.")
    parser.add_argument("-ne", "--num_epochs", default=1250, type=int,
        help="how many epochs of training to run. By default it's 1250.")
    parser.add_argument("-se", "--save_model_every", default=100, type=int,
        help="how often save an intermediate version of the model. By default it's every 100.")
    parser.add_argument("-ve", "--validate_every", default=10, type=int,
        help="how often to validate in epochs. By default it's every 10.")
    parser.add_argument("-lrde", "--learning_rate_decay_every", default=15, type=int,
        help="how often to decay learning rate. By default it's every 15.")
    parser.add_argument("-lrdr", "--learning_rate_decay_rate", default=.99, type=float,
        help="how much to decay learning rate. By default it's .99.")

    parsed_args = parser.parse_args()

    train(train_dataset=parsed_args.train_dataset,
        validation_dataset=parsed_args.validation_dataset,
        test_dataset=parsed_args.test_dataset,
        path_to_save_model=parsed_args.path_to_save_model,
        path_to_wav=parsed_args.path_to_wav,
        num_epochs=parsed_args.num_epochs,
        save_model_every=parsed_args.save_model_every,
        validate_every=parsed_args.validate_every,
        learning_rate_decay_every=parsed_args.learning_rate_decay_every,
        learning_rate_decay_rate=parsed_args.learning_rate_decay_rate)
