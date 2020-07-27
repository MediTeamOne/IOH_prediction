import os
import torch
import numpy as np
import pandas as pd 
import sys
sys.path.append('../')
from model import ResidualNetHDI, HDIClassifier
from torch.utils.data import Dataset
from util import utils
import argparse
from filters import filter_eeg,filter_signal

class WaveDataset (Dataset):
    def __init__(self, file_list, wave_types):
        self.root_wav = '/Dropbox/data/npz_wave'
        self.df_file_list = file_list
        self.len = len(self.df_file_list)
        self.wave_types = wave_types
 
    def __getitem__(self, index):    
        #wave = np.empty((0,1))
        input_dict = dict()
        #abp, ecg, eeg = np.array([0]), np.array([0]), np.array([0])
        label = self.df_file_list.at[index,'label']
        if label =='n':
            y = torch.zeros(1,dtype = torch.float32) 
        else:
            y = torch.ones(1,dtype = torch.float32) 

        for wav in self.wave_types: 
            if wav == 'abp':
                wf_name = (os.path.join(self.root_wav,self.df_file_list.at[index,wav])[:-4]+'.npz')
                abp = np.load(wf_name)['arr_0']
                input_dict['abp'] = torch.from_numpy(abp.reshape(1,-1)).float()
                #wave = np.concatenate((wave,abp),axis=None)
                #abp = (tmp-tmp.min()) / (tmp.max() - tmp.min())                
            elif wav == 'ecg':
                wf_name = (os.path.join(self.root_wav,self.df_file_list.at[index,wav])[:-4]+'.npz')
                ecg = np.load(wf_name)['arr_0']
                ecg = filter_signal(ecg,ftype='FIR',band='bandpass',order=int(0.3*500),frequency=(1,40),sampling_rate=500).copy()
                ecg = (ecg-np.mean(ecg)) / ((np.std(ecg))+1e-5)
                input_dict['ecg'] = torch.from_numpy(ecg.reshape(1,-1)).float()
                #wave = np.concatenate((wave,ecg),axis=None)
            elif wav == 'eeg':
                wf_name = (os.path.join(self.root_wav,self.df_file_list.at[index,wav])[:-4]+'.npz')
                eeg = np.load(wf_name)['arr_0']
                eeg =  filter_eeg(eeg,sampling_rate=128).copy()
                input_dict['eeg'] = torch.from_numpy(eeg.reshape(1,-1)).float()  
                #wave = np.concatenate((wave,eeg),axis=None)
                #eeg = (tmp-tmp.min()) / (tmp.max() - tmp.min())
        
        #ecg = (ecg-ecg.mean()) / (ecg.std())
        #print(abp.shape,ecg.shape,eeg.shape)
        #wave = wave.reshape(1,-1) 
        return input_dict, y    
    
    def __len__(self):
        return self.len
 

class HDIDeepModel(utils.DeepModel):
    def __init__(self, model, lr, batch_size, epoch,
                        num_GPUs, num_workers, task, save_path, note):
        super(HDIDeepModel, self).__init__(model, lr, batch_size, epoch,
                        num_GPUs, num_workers, task, save_path, note)

        self.optimizer = torch.optim.Adam(self.m.parameters(), lr=lr, weight_decay=1e-5,)
        self.criterion= torch.nn.BCELoss()    

    def feed_foward (self, x, y, eval_mode = False):  
        y_hat, y = self.m.forward(x, y)        
        loss = self.criterion(y_hat, y)
        return loss, y_hat ,y   


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--channel', type = int, help='num_channels', default=5)    
    # parser.add_argument('--dropout', type = float, help='drop_out',default=0.5)
    # parser.add_argument('--hidden', type = int, help='hidden',default=32)
    parser.add_argument('--lr', type = float, help='learning rate',default=1e-3)
    parser.add_argument('--num_epoch', type = int, help='integer number', default=100)
    parser.add_argument('--batch_size', type = int, help='batch size',default=32)
    parser.add_argument('--train', type = str, help='training_case list',default='../dataset_file_info/3M_0.08_train.csv')
    parser.add_argument('--val', type = str, help='training_case list',default='../dataset_file_info/3M_0.08_val.csv')
    parser.add_argument('--test', type = str, help='test_case list',default='../dataset_file_info/3M_0.08_test.csv')
    parser.add_argument('--wform',type=str, help='wave form list', nargs = '+', default= ['abp', 'ecg', 'eeg'])
    parser.add_argument('--note',type=str, help='note',  default= 'toy')
    parser.add_argument('--ext_validation', type = bool, help = 'option for ext_validation', default = False)
    parser.add_argument('--best_model', type = str, help='for external validation', default = '')

    args = parser.parse_args()
    np.random.seed(0)

    train_list = pd.read_csv(args.train)#.sample(frac=1).reset_index(drop=True)
    train_list_e = train_list.query("label=='e'")
    train_list_n = train_list.query("label=='n'")
    unbalance_rate = int(len(train_list_n)/len(train_list_e))
    train_df = pd.concat([train_list]+[train_list_e]*unbalance_rate)
    train_df = train_df.sample(frac=1,random_state =1).reset_index(drop=True)


    val_df = pd.read_csv(args.val).sample(frac=1,random_state =1).reset_index(drop=True)
    test_df = pd.read_csv(args.test).sample(frac=1,random_state =1).reset_index(drop=True)

    print('{:} waves used'.format(len(args.wform)))
    print('{}: {:}, {}: {:}, {}: {:}'.format(args.train.split('/')[-1][:-4], len(train_df), \
                                args.val.split('/')[-1][:-4], len(val_df), \
                                args.test.split('/')[-1][:-4], len(test_df)))

    train_set = WaveDataset(train_df,args.wform)  
    test_set = WaveDataset(test_df,args.wform)
    valid_set = WaveDataset(val_df,args.wform)  
        
    if args.ext_validation: 
        model_path = './eval/eval_all_sqi_2ch_05-14/model'
        models = []
        for _,_,f in os.walk(model_path):
            models.extend(f)
        models = [x for x in models if x.find(args.train.split('/')[-1].split('.')[0])>-1]
        for model in models:
            hdi_arch = torch.load(os.path.join(model_path,model)) ## model이 존재하는 path와 파일 이름
            hdi_predictor = HDIDeepModel(hdi_arch, 
                                lr=args.lr, batch_size =args.batch_size, epoch=args.num_epoch,
                                num_GPUs = 2, num_workers =8, 
                                task='cls', save_path='eval_'+args.note,
                                note = model)

            hdi_predictor.execution(None, test_set, None, save=True)

    else:
        arch = dict()
        # ch = args.channel
        hidden_size = 32 # args.hidden
        dropout = 0.5 # args.dropout
        for wav in args.wform:
            if wav == 'abp':
                arch['abp'] = ResidualNetHDI(seq_length=30000,
                            kernel_size=15,
                            num_channel=2,
                            output = hidden_size, do = dropout)
            elif wav == 'ecg':
                arch['ecg']  = ResidualNetHDI(seq_length=30000,
                            kernel_size=15,
                            num_channel=2,
                            output = hidden_size, do = dropout)
            elif wav == 'eeg':
                arch['eeg']  = ResidualNetHDI(seq_length=7680,
                            kernel_size=7, 
                            num_channel=2,
                            output = int(hidden_size), do = dropout)
        hdi_arch = HDIClassifier(arch, args.wform, output=1, hidden=hidden_size)

        hdi_predictor = HDIDeepModel(hdi_arch, 
                                    lr=args.lr, batch_size =args.batch_size, epoch=args.num_epoch,
                                    num_GPUs = 2, num_workers =8, 
                                    task='cls', save_path='./eval/eval_'+args.note,
                                    note = args.train.split('/')[-1][:-4]+'_'+'_'.join(args.wform)+'_')
        
        hdi_predictor.execution(train_set, test_set, valid_set, save=True)
    