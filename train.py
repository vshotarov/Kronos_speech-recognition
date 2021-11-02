import torch
from torch import nn
from model import STTModel
from data import Dataset, LABEL_INDICES, LABELS, pad
from functools import partial


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = STTModel().to(device)
    model.train()
    loader = torch.utils.data.DataLoader(Dataset("speech_train.csv"), batch_size=8,
            collate_fn=partial(pad, device), shuffle=True)
    val_loader = torch.utils.data.DataLoader(Dataset("speech_test.csv", validation=True),
            batch_size=16, collate_fn=partial(pad, device), shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset("speech_test.csv", validation=True),
            batch_size=1, collate_fn=partial(pad, device))
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

    best_val_loss = float("inf")
    best_val_loss_epoch = -1

    try:
        for epoch in range(1250):
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

            if epoch % 15 == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= .99
                    lr = g["lr"]
                print("lr is now", lr)

            if epoch % 10 == 0:
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
                        torch.save(model, "saved_model.torch")
                        print("saved model to saved_model.torch")

                    if epoch % 100 == 0:
                        torch.save(model, "saved_model_%s.torch" % str(epoch))
                        print("saved model to saved_model_%s.torch" % str(epoch))

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

if __name__ == "__main__":
    train()
