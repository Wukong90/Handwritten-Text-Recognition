import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torch.autograd import Variable
from configure import Preprocessing
from configure import myDataset
from utils import CER, WER

from model import HAMVisContexNN
#from evaluate import evaluate

out_f = open('./train_loss/train_CTC_HAM_Vis_Contex.txt','w')
save_model_dir = './weights/'
model_name = 'CTC_HAM_Vis_Contex_epoch'
alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet)}  # int -> character



def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    


    img = data[0]
    targets = data[1]

    images = Variable(img.data.unsqueeze(1))
    images = images.cuda()


    logits = crnn(images)
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)

    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)  #logits.size(0) denote the width of image
    input_lengths = input_lengths.cuda()
    # Process labels

    labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(targets)]))
    labels = labels.cuda()



    label_lengths = torch.LongTensor([len(t) for t in targets])
    label_lengths = label_lengths.cuda()




    loss = criterion(log_probs, labels, input_lengths, label_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def val(model, criterion, val_loader, len_val_set):
    model.eval()
    avg_cost = 0
    avg_CER = 0
    avg_WER = 0

    for val_data in val_loader:
        # Process predictions
        img = val_data[0]
        transcr = val_data[1]

        images = Variable(img.data.unsqueeze(1))
        images = images.cuda()

        preds = model(images)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * images.size(0)))

        # Process labels for CTCLoss
        labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(transcr)]))
        label_lengths = torch.LongTensor([len(t) for t in transcr])
        # Compute CTCLoss
        preds_size = preds_size.cuda()
        labels = labels.cuda()
        label_lengths = label_lengths.cuda()
        cost = criterion(preds, labels, preds_size, label_lengths)  # / batch_size
        avg_cost += cost.item()

        # Convert paths to string for metrics
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        if tdec.ndim == 1:
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            # Compute metrics
            avg_CER += CER(transcr[0], dec_transcr)
            avg_WER += WER(transcr[0], dec_transcr)
        else:
            for k in range(len(tdec)):
                tt = [v for j, v in enumerate(tdec[k]) if j == 0 or v != tdec[k][j - 1]]
                dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
                # Compute metrics
                avg_CER += CER(transcr[k], dec_transcr)
                avg_WER += WER(transcr[k], dec_transcr)

    avg_cost = avg_cost / len(val_loader)
    avg_CER = avg_CER / len_val_set
    avg_WER = avg_WER / len_val_set
    return avg_cost, avg_CER, avg_WER


def main():

    epochs = 400
    train_batch_size = 20
    lr = 0.0005
    cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = myDataset(data_type='IAM', data_size=(124, 1751),
                          set='train', centered=False, deslant=False, data_aug=True,
                          keep_ratio=True, enhance_contrast=False)

    val1_set = myDataset(data_type='IAM', data_size=(124, 1751),
                         set='val', centered=False, deslant=False, keep_ratio=True,
                         enhance_contrast=False)


    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=Preprocessing.pad_packed_collate)

    val_loader = DataLoader(
        dataset=val1_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=Preprocessing.pad_packed_collate)

    len_val_set = val1_set.__len__()

    num_class = len(alphabet)
    HVCNN = HAMVisContexNN(1, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256)   

    HVCNN.cuda()
    optimizer = optim.RMSprop(HVCNN.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')

    criterion.cuda()

    #assert save_interval % valid_interval == 0
    i = 1
    show_interval = 5
    #Train
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}',file=out_f)
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(HVCNN, train_data, optimizer, criterion, device)
            train_size = train_batch_size
            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('current_train_batch_loss[', i, ']: ', loss / train_size,file=out_f)
            out_f.flush()            
            i += 1
        save_model_path = save_model_dir + model_name + str(epoch)
        torch.save(HVCNN.state_dict(), save_model_path)      
        i = 1
        print('train_loss: ', tot_train_loss / tot_train_count,file=out_f)

        # Validation
        if epoch % 1 == 0:
            val_loss, val_CER, val_WER = val(HVCNN, criterion, val_loader, len_val_set)
            #if params.save:
            print('val loss ', val_loss, 'epoch' , epoch, file=out_f)
            print('val CER ', val_CER, 'epoch' ,epoch, file=out_f)
            print('val WER ', val_WER, 'epoch' ,epoch, file=out_f)







if __name__ == '__main__':
    main()
