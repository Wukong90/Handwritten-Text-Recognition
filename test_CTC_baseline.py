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
from tqdm import tqdm


model_name = './weights/CTC_HAM_Vis_Contex_epoch_epoch'
alphabet = """_!#&\()*+,-.'"/0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz """
cdict = {c: i for i, c in enumerate(alphabet)}  
icdict = {i: c for i, c in enumerate(alphabet)}  

def test(model, criterion, test_loader, len_test_set):
    print("Testing...")
    model.eval()

    avg_cost = 0
    avg_CER = 0
    avg_WER = 0
    for iter_idx, (img, transcr, ids) in enumerate(tqdm(test_loader)):
        
        img = Variable(img.data.unsqueeze(1))
        img = img.cuda()
       
        with torch.no_grad():
            preds = model(img)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * img.size(0)))

        
        labels = Variable(torch.LongTensor([cdict[c] for c in ''.join(transcr)]))
        label_lengths = torch.LongTensor([len(t) for t in transcr])
      
        preds_size = preds_size.cuda()
        labels = labels.cuda()
        label_lengths = label_lengths.cuda()
        cost = criterion(preds, labels, preds_size, label_lengths)  
        avg_cost += cost.item()

       
        tdec = preds.argmax(2).permute(1, 0).cpu().numpy().squeeze()

        if tdec.ndim == 1:  
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            
            avg_CER += CER(transcr[0], dec_transcr)
            avg_WER += WER(transcr[0], dec_transcr)
        else:
            for k in range(len(tdec)):
                tt = [v for j, v in enumerate(tdec[k]) if j == 0 or v != tdec[k][j - 1]]
                dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
              
                avg_CER += CER(transcr[k], dec_transcr)
                avg_WER += WER(transcr[k], dec_transcr)

                if iter_idx % 50 == 0 and k % 2 == 0:
                    print('label:', transcr[k])
                    print('prediction:', dec_transcr)
                    print('CER:', CER(transcr[k], dec_transcr))
                    print('WER:', WER(transcr[k], dec_transcr))


    avg_cost = avg_cost / len(test_loader)
    avg_CER = avg_CER / len_test_set
    avg_WER = avg_WER / len_test_set
    print('Average CTCloss', avg_cost)
    print("Average CER", avg_CER)
    print("Average WER", avg_WER)

    print("Testing done.")
    return avg_cost, avg_CER, avg_WER



test_set = myDataset(data_type='IAM', data_size=(124, 1751),
                set='test', centered=False, deslant=False,  keep_ratio=False,
                enhance_contrast=False)

TEST_LOADER = DataLoader(
        dataset=test_set,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=Preprocessing.pad_packed_collate)

num_class = len(alphabet)

CRNN = CRNNOri(1, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256)   

CRNN.load_state_dict(torch.load(model_name))
CRNN.cuda()

CRITERION = CTCLoss()
CRITERION.cuda()

LEN_TEST_SET = test_set.__len__()

test(CRNN, CRITERION, TEST_LOADER, LEN_TEST_SET)



