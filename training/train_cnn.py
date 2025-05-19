# training/train_cnn.py
import torch, time, os
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from data_prep import prepare_data, HASYDataset
from model_cnn import PopoCNN
from tqdm import tqdm

BATCH     = 256
EPOCHS    = 30
LR        = 1e-3
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'best_popo_cnn.pth'

def main():
    (tr_f,tr_l),(vl_f,vl_l) = prepare_data()
    train_dl = DataLoader(HASYDataset(tr_f,tr_l), batch_size=BATCH,
                          shuffle=True,num_workers=4,pin_memory=True)
    val_dl   = DataLoader(HASYDataset(vl_f,vl_l), batch_size=BATCH,
                          shuffle=False,num_workers=4,pin_memory=True)

    model = PopoCNN(len(set(tr_l))).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.CrossEntropyLoss()
    writer= SummaryWriter('runs/hasy_popo')

    best = 0
    for ep in range(1,EPOCHS+1):
        model.train(); tloss=0
        for x,y in tqdm(train_dl, desc=f'E{ep:02d} train'):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); out=model(x); loss=crit(out,y)
            loss.backward(); opt.step(); tloss+=loss.item()
        tloss/=len(train_dl)

        model.eval(); vloss=0; correct=0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(DEVICE), y.to(DEVICE)
                out=model(x); vloss+=crit(out,y).item()
                correct+=(out.argmax(1)==y).sum().item()
        vloss/=len(val_dl); acc=correct/len(val_dl.dataset)
        writer.add_scalars('loss', {'train':tloss,'val':vloss}, ep)
        writer.add_scalar('val_acc', acc, ep)
        print(f'E{ep:02d} TL {tloss:.3f} VL {vloss:.3f} ACC {acc:.3f}')

        if acc>best:
            best=acc; torch.save(model.state_dict(), SAVE_PATH)
            print(f'⇪ new best {best:.3f}')
        sched.step()
    print('Done, best acc', best)

if __name__=='__main__':
    main()
