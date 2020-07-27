from oct2py import Oct2Py
import numpy as np
import pandas as pd
import os
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt

#root_path = '/share_folder/ncc_jyy/art_0M_event/'
# root_path = '/share_folder/data/nevent_wave/'
root_path = '/Dropbox/data/3M_nointv_event_wave/'
THREADS=12

save_file_name = '3M_nointv_event_wave.csv'
save_error_name = '3M_nointv_event_wave_err.csv'


def show_fig(file_name,err,art):
    fig = plt.figure(figsize=(6,4)) 
    ax1 = fig.add_subplot(1,1,1) 
    ax1.plot(art['second'],art['SNUADC/ART'],c='r',linewidth=1) 
    ax1.set_ylabel('value')
    ax1.set_xlabel('time')
    fig.text(0.4,0.9, err, fontsize=15)
    plt.savefig('./err/'+file_name+err+'.png',dpi=75)
    plt.close(fig)
    #plt.show()

def new_metric_jsqi(onset, beat_q):
    cum_interval =0 
    for i ,(on, sqi) in enumerate( zip (onset,beat_q[:,0]),1):
        #print(i, (on, sqi), end =' ')
        if i != len(onset):
            if sqi ==0:
                cum_interval += onset[i]-on
                #print('interval',onset[i]-on,'cum_interval', cum_interval)
#             else:
#                 #print()
    new_metric = cum_interval/7500
    #print('new metric (based on jsqi):',cum_interval/7500)
    return new_metric[0]
 
def sqi_chk(sample_waves, result_dict, error_waves,thd_id=0):
    oc = Oct2Py()
    for i, file_name in enumerate (sample_waves,1): 
        try: 
            if i% 10 ==0:
                print('THREAD_ID {}:{},{}/{}'.format(thd_id,file_name,i,len(sample_waves)))
            art = pd.read_csv(os.path.join(root_path,file_name), index_col =0)
            x = art['SNUADC/ART'].to_numpy()
            x = np.reshape(x[::4],(-1,1)) 
            onset = oc.wabp(x)         
            features = oc.abpfeature(x, onset) 
            beat_q, jsqi = oc.jSQI(features, onset, x, nout=2)
            new_sqi = new_metric_jsqi(onset, beat_q)
            print(len(features), len(onset), jsqi,new_sqi)
            result_dict[file_name] = {'jsqi':jsqi, 'new_metric':new_sqi}
        except Exception as e: 
            print(e)
            error_waves[file_name] = e

if __name__ == '__main__': 
    sample_waves = []    
    for _,_,files in os.walk(root_path): 
        for f in files: 
            try:
                if 'ART' == f.split('_')[2].split('.')[0]: 
                    sample_waves.append(f)
            except: pass 

    # #single
    # error_dict = dict()
    # result_dict = dict()
    # sqi_chk(sample_waves[bounds[0]:bounds[1]],result_dict,error_dict)

    ## multiprocess
    error_dict = Manager().dict()
    result_dict = Manager().dict()    
 
    procs = []
    total_progress = len(sample_waves)
    th_job = int(total_progress/THREADS)
 
    
    for thd_id in range (THREADS):
        job_list = []
        if thd_id == THREADS-1: 
            job_list = sample_waves[thd_id*th_job:]

        else: 
            job_list = sample_waves[thd_id*th_job:(thd_id+1)*th_job]

        proc = Process(target=sqi_chk, args=([job_list,result_dict,error_dict, thd_id]))
        procs.append(proc)
        proc.start() 
    
    for proc in procs:
        proc.join()   

  #  result_f_name = root_path.split('/'[-1])
    pd.DataFrame.from_dict(result_dict, orient='index').to_csv(save_file_name)
    pd.DataFrame.from_dict(error_dict, orient='index').to_csv(save_error_name)