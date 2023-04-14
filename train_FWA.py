import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torch.autograd import Variable
from configure import Preprocessing
from configure import myDataset
from utils import CER, WER


from model import HAMVisContexNN,WIDNN,Bridge
#from evaluate import evaluate

out_f = open('./train_loss/train_adaptation.txt','w')
save_model_dir = './weights/ADA_weights/'
model_name1 = 'ada_rec_epoch'
model_name2 = 'ada_sen_epoch'
model_name3 = 'ada_bri_epoch'
pre_trained_rec = './weights/'
pre_trained_wid = './weights/'
alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
icdict = {i: c for i, c in enumerate(alphabet)}  # int -> character


def train_batch(recognet, idnet, brinet, data, optimizer, criterion, device):
    recognet.train()
    idnet.train()
    brinet.train()
    
    img = data[0]
    targets = data[1]

    images = Variable(img.data.unsqueeze(1))
    images = images.cuda()

    global_wid = idnet(images,True)
    win1,win2,win3 = brinet(global_wid)

    logits = recognet(images, win1,win2,win3,True)
    
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


def val(recognet,idnet,brinet, criterion, val_loader, len_val_set):
    recognet.eval()
    idnet.eval()
    brinet.eval()
    avg_cost = 0
    avg_CER = 0
    avg_WER = 0

    for val_data in val_loader:
        # Process predictions
        img = val_data[0]
        transcr = val_data[1]

        images = Variable(img.data.unsqueeze(1))
        images = images.cuda()

        global_wid = idnet(images,True)
        win1,win2,win3 = brinet(global_wid)

        preds = recognet(images, win1,win2,win3,True)

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
    train_batch_size = 16
    lr = 0.0005
    cdict = {c: i for i, c in enumerate(alphabet)}  # character -> int
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = myDataset(data_type='IAM', data_size=(124, 1751),
                          set='train', centered=False, deslant=False, data_aug=True,set_wid=False,
                          keep_ratio=True, enhance_contrast=False, data_shuffle=False)

    val1_set = myDataset(data_type='IAM', data_size=(124, 1751),
                         set='val', centered=False, deslant=False, keep_ratio=True,set_wid=False,
                         enhance_contrast=False,data_shuffle=False)
    '''
    val2_set = myDataset(data_type='IAM', data_size=(124, 1751),
                         set='val2', centered=False, deslant=False, keep_ratio=True,set_wid=False,
                         enhance_contrast=False,data_shuffle=False)
    '''

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
    '''
    val_loader2 = DataLoader(
        dataset=val2_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=Preprocessing.pad_packed_collate)
    '''
    len_val_set = val1_set.__len__()
    #len_val2_set = val2_set.__len__()

    num_class = len(alphabet)
    
    recog_net = HAMVisContexNN(1, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256)  

    id_net = WIDNN(1, 283,
                map_to_seq_hidden=32,
                rnn_hidden=128)

    bri = Bridge(hidden_dim=256)
 
    recog_net.load_state_dict(torch.load(pre_trained_rec))
    id_net.load_state_dict(torch.load(pre_trained_wid))

    recog_net.cuda()
    id_net.cuda()
    bri.cuda()
    optimizer = optim.RMSprop([{'params':recog_net.parameters()},{'params':id_net.parameters()},{'params':bri.parameters()}], lr=lr)
    criterion = CTCLoss(reduction='sum')

    criterion.cuda()

    i = 1
    show_interval = 5

    #Train
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}',file=out_f)
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(recog_net, id_net, bri , train_data, optimizer, criterion, device)
            train_size = train_batch_size
            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('current_train_batch_loss[', i, ']: ', loss / train_size,file=out_f)
            out_f.flush()

            i += 1
        save_model1_path = save_model_dir + model_name1 + str(epoch)
        save_model2_path = save_model_dir + model_name2 + str(epoch)
        save_model3_path = save_model_dir + model_name3 + str(epoch)

        torch.save(recog_net.state_dict(), save_model1_path)
        torch.save(id_net.state_dict(), save_model2_path)
        torch.save(bri.state_dict(), save_model3_path)

        i = 1
        print('train_loss: ', tot_train_loss / tot_train_count,file=out_f)

        # Validation
        if epoch % 1 == 0:
            val_loss, val_CER, val_WER = val(recog_net,id_net,bri,criterion, val_loader, len_val_set)
            #val_loss2, val_CER2, val_WER2 = val(recog_net,id_net,bri,criterion, val_loader2, len_val2_set)
            print('val WER CER', val_WER,val_CER, 'epoch' ,epoch, file=out_f)
            #print('val2 WER CER', val_WER2,val_CER2, 'epoch' ,epoch, file=out_f)
            #avg_CER = ((val_CER * len_val_set) + (val_CER2 * len_val2_set)) / (len_val_set + len_val2_set)
            #avg_WER = ((val_WER * len_val_set) + (val_WER2 * len_val2_set)) / (len_val_set + len_val2_set)
            #print('avg WER CER', avg_WER,avg_CER, 'epoch' ,epoch, file=out_f)

if __name__ == '__main__':
    main()
