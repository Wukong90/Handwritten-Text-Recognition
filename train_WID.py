import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from torch.autograd import Variable
from configure import Preprocessing
from configure import myDataset
from torch import nn
from model import WIDNN

out_f = open('./train_loss/train_wids.txt','w')
save_model_dir = './weights/WID_weights/'
model_name = 'wid_epoch'
img2id_f = open('./data/IAM/split/train_img2id.txt','r')

img2id = {}
ids = {}
count=0
while True:
	line=img2id_f.readline()
	line=line.strip("\n")
	if not line:
		break
	tmp_line=line.split("	")
	if tmp_line[1] not in ids:
		ids[tmp_line[1]] = count
		count = count + 1
	img2id[tmp_line[0]] = ids[tmp_line[1]]



def train_batch(crnn, data, optimizer, criterion):
    crnn.train()

    img = data[0]
    labels = data[2]

    images = Variable(img.data.unsqueeze(1))
    images = images.cuda()

    target = Variable(torch.from_numpy(np.array(labels)))


    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)

    target = target.cuda()
    loss = criterion(log_probs, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def val(model, criterion, val_loader, len_val_set):
    model.eval()
    avg_AR = 0
    total_count = 0
    for val_data in val_loader:
        # Process predictions
        count = 0
        img = val_data[0]
        labels = val_data[2]

        images = Variable(img.data.unsqueeze(1))
        images = images.cuda()

        target = Variable(torch.from_numpy(np.array(labels)))

        preds = model(images)

        preds = preds.argmax(dim=1)
        preds = preds.cpu()
        
        for num in range(0,len(preds)):
                if (target[num] == preds[num]):
                        count = count + 1
        total_count = total_count + count

    avg_AR = float(total_count) / len_val_set
    return avg_AR



def main():
    epochs = 400
    train_batch_size = 64
    lr = 0.0005

    train_set = myDataset(data_type='IAM', data_size=(124, 1751),
                          set='train', set_wid=True, centered=False, deslant=False, data_aug=True,
                          keep_ratio=True, enhance_contrast=False,data_shuffle=True)

    val1_set = myDataset(data_type='IAM', data_size=(124, 1751),
                         set='val', set_wid=True, centered=False, deslant=False, keep_ratio=True,
                         enhance_contrast=False,data_shuffle=True)

    criterion = nn.CrossEntropyLoss().cuda()

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

    num_class = len(ids)


    wnn = WIDNN(1, num_class,
                map_to_seq_hidden=32,
                rnn_hidden=128)

    wnn.cuda()
    optimizer = optim.RMSprop(wnn.parameters(), lr=lr)

    len_val_set = val1_set.__len__()
    len_val_set = float(len_val_set)

    i = 1
    show_interval = 5
    #Train
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}',file=out_f)
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(wnn, train_data, optimizer, criterion)
            train_size = train_batch_size
            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('current_train_batch_loss[', i, ']: ', loss / train_size,file=out_f)
            out_f.flush()

            i += 1


        save_model_path = save_model_dir + model_name + str(epoch)
        torch.save(wnn.state_dict(), save_model_path)      
        i = 1
        print('train_loss: ', tot_train_loss / tot_train_count,file=out_f)



        # Validation
        if epoch % 1 == 0:
            val_AR = val(wnn, criterion, val_loader, len_val_set)
            #if params.save:
            print('val AR ', val_AR, file=out_f)




if __name__ == '__main__':
    main()
