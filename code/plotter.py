import atexit
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
plt.rcParams['figure.figsize'] = [10, 5]

root = './logs/'

def load_json_df (filename):

    df = pd.read_json(root+filename)
    df = df.set_index('Epoch')

    return df

def plots (filenames):

    dfs = [ load_json_df(f) for f in filenames]
    cols = load_json_df(filenames[0]).columns
    rows = np.min([ d.shape[0] for d in dfs])
    dfs = [ x.iloc[:rows,:] for x in dfs]

    for x in cols:
        plt.figure()
        for i, d in enumerate(dfs):
            plt.plot(d.index, d[x],label=filenames[i])
            plt.title(x)
            plt.legend()
            plt.savefig(f'tmp_plot_{x}.png')

    return

def plot_report(filenames,name,legends, idx=None):

    dfs = [ load_json_df(f) for f in filenames]
    cols = load_json_df(filenames[0]).columns
    rows = np.min([ d.shape[0] for d in dfs])
    if idx!=None:
        rows = min(idx,rows)
    dfs = [ x.iloc[:rows,:] for x in dfs]
    x = dfs[0].index

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    ax1.plot(x, dfs[0]['Train_loss'], 'r-',label=legends[0])
    ax1.plot(x, dfs[1]['Train_loss'], 'b-',label=legends[1])
    ax1.legend(loc='center right', shadow=True)

    ax2.plot(x, dfs[0]['Test_loss'], 'r--')
    ax2.plot(x, dfs[1]['Test_loss'], 'b--')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax2.set_ylabel('Test Loss')

    plt.savefig(f'{name}.png')
    
    return

list = [
    #compare learning rate
    # 'res50_e100_SDG_step_lr5e-05_bs16_m0.8_wd0.005.json', #*
    # 'res50_e100_SDG_step_lr0.0001_bs16_m0.8_wd0.005.json',
    # 'res50_e100_SDG_step_lr0.0005_bs16_m0.8_wd0.005.json',

    # # compare momentum
    # # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.95_wd0.0005.json',
    # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.9_wd0.0005.json',
    # # # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.85_wd0.0005.json',
    # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.8_wd0.0005.json', #*
    # # # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.7_wd0.0005.json'

    # 'B_swin_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0.json', #21.7
    # 'B_swin_e50_SDG_cosine_lr5e-05_bs32_m0.9_wd0.0.json' #23

    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.75_wd0.005.json', #* 23.2
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.005.json',
    # 'res50_e100_SDG_step_lr5e-05_bs16_m0.85_wd0.005.json' #

    # compare batch size
    # 'res50_e100_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005.json',
    # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.9_wd0.0005.json', #*

    # compare optimizer
    # 'res50_e100_Adam_cosine_lr0.001_bs16_m0.9_wd0.0005.json',
    # 'res50_e100_SDG_cosine_lr0.001_bs16_m0.9_wd0.0005.json'  #*

    # compare scheduler
    # 'res50_e100_SDG_step_lr0.001_bs32_m0.9_wd0.0005.json',  #*
    # 'res50_e100_SDG_wcosine_lr0.001_bs32_m0.9_wd0.0005.json',
    # 'res50_e100_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005.json'

    # compare weight decay
    # 'A_res50_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.005.json',
    # 'A_res50_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005.json'
    # 'res50_e100_SDG_step_lr0.001_bs16_m0.85_wd0.05.json',
    # # 'res50_e100_SDG_step_lr0.001_bs16_m0.85_wd0.01.json',
    # 'res50_e100_SDG_step_lr0.001_bs16_m0.85_wd0.005.json', #*
    # 'res50_e100_SDG_step_lr0.001_bs16_m0.85_wd0.001.json'

    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.0005.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001.json', #**
    # 'res50_e100_SDG_step_lr5e-05_bs16_m0.8_wd0.005.json', #*
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.01.json'

    # 'res50_e50_SDG_wcosine_lr0.0001_bs16_m0.9_wd0.001.json', #*
    # 'res50_e50_SDG_wcosine_lr0.0001_bs16_m0.9_wd0.005.json',
    # 'res50_e50_SDG_wcosine_lr0.0001_bs16_m0.9_wd0.01.json'

    # 'res101_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.01.json',
    # 'res101_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.005.json',
    # 'res101_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001.json'

    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001input_stats.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001.json'

    # 'swin_e50_SDG_wcosine_lr5e-05_bs16_m0.9_wd0.005.json',
    # 'swin_e50_SDG_wcosine_lr5e-05_bs16_m0.9_wd0.001.json'

    # 'res50_e100_SDG_step_lr5e-05_bs16_m0.8_wd0.001augmix.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001autoaug.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001automixaug.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001manual_aug.json',

    # 'res50_e100_SDG_step_lr5e-05_bs16_m0.8_wd0.001augmix.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001mix_aug.json'

    # 'R_res50_e50_SDG_step_lr5e-05_bs32_m0.8_wd0.001.json',
    # 'R_res50_e50_SDG_cosine_lr0.0001_bs32_m0.9_wd0.001.json',
    # 'R_res50_e50_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.001.json',

    # 'R_res50_e50_SDG_cosine_lr0.0005_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_cosine_lr0.0001_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_cosine_lr5e-05_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_cosine_lr1e-05_bs32_m0.9_wd0.0005.json'

    # 'A_res50_e50_Adam_cyclic_lr0.001_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005.json',
    # 'R_res50_e50_SDG_cosine_lr0.001_bs64_m0.9_wd0.0005.json'
    'A_res50_e50_SDG_cosine_lr5e-05_bs16_m0.8_wd0.0005l1.json',
    # 'A_res50_e50_SDG_cosine_lr5e-05_bs32_m0.85_wd0.001l10.0005.json',
    # 'A_res50_e36_SDG_cosine_lr5e-05_bs16_m0.8_wd0.0005l1l2.json'

    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.002.json',
    # 'res50_e50_SDG_step_lr5e-05_bs16_m0.8_wd0.004.json',
    'res50_e100_SDG_step_lr5e-05_bs16_m0.8_wd0.005.json',

    # weight decay
    # 'A_res50_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005l1.json',
    # 'A_res50_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.005l1.json'

    # weight decay 100
    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005.json',
    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.01.json'

    # 'A_res50_e100_SDG_cosine_lr0.0005_bs32_m0.8_wd0.005.json',
    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005.json'

    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005.json',
    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005l1.json'

    # #white
    # 'R_res50_e50_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005imagenet.json',
    # # 'R_res50_e50_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005no-norm.json',
    # 'R_res50_e50_SDG_cosine_lr0.001_bs32_m0.9_wd0.0005white.json'

    # 'A_res50_e100_SDG_cosine_lr1e-05_bs32_m0.8_wd0.0005l10.005.json',
    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005l10.005.json'
    
    'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005l1.json',
    'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005.json'

    # 'A_res50_e100_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0005noaug_nol1.json',
    # 'A_res50_e100_SDG_cosine_lr3e-05_bs32_m0.8_wd0.0005noaug_nol1.json',

    # 'A_res50_e100_SDG_cosine_lr3e-05_bs32_m0.8_wd0.0005nol1.json',
    # 'A_res50_e100_SDG_cosine_lr3e-05_bs32_m0.8_wd0.0005l10.005.json'

    # 'B_swin_e50_Adam_cosine_lr0.0001_bs32_m0.9_wd0.0.json',
    # 'B_res50_e50_Adam_cosine_lr5e-05_bs32_m0.9_wd0.0.json',
    # 'B_swin_e50_SDG_cosine_lr5e-05_bs32_m0.8_wd0.0.json'

    # 'B_swin_e30_Adam_cosine_lr7e-05_bs32_m0.9_wd0.001.json',
    # 'B_res101_e30_Adam_cosine_lr7e-05_bs32_m0.9_wd0.001.json',
    # 'B_swin2_e30_Adam_cosine_lr7e-05_bs32_m0.9_wd0.001.json'

    # 'SMP2_dp_H512W512_e50_bs8_lr5e-05_seglambda5e-05.json'
    # 'B_swin_e30_Adam_warmcosine_lr7e-05_bs32_m0.9_wd0.0005.json',
    # 'B_swin_e30_Adam_cosine_lr7e-05_bs32_m0.9_wd0.001.json'

    # 'B_swin2_e50_Adam_warmcosine_lr7e-05_bs32_wd0.001WD.json',
    # 'B_swin2_e50_Adam_warmcosine_lr7e-05_bs32_wd0.0005WD.json'

    # 'A_res50_e50_SDG_step_lr5e-05_bs32_m0.8_wd0.0005.json',
    # 'A_res50_e50_SDG_step_lr5e-05_bs32_m0.8_wd0.001.json'

    
    # 'B_swin2_e50_Adam_warmcosine_lr1e-05_bs32_wd0.001LR.json',
    # 'B_swin2_e50_Adam_warmcosine_lr7e-05_bs32_wd0.001WD.json',
    # 'B_swin2_e50_Adam_warmcosine_lr0.0001_bs32_wd0.001LR.json'
    # 'B_swin2_e50_Adam_warmcosine_lr7e-05_bs32_wd0.0WD.json',
    # 'A_res50_e100_SDG_cosine_lr3e-05_bs32_m0.8_wd0.0005l10.005.json',
    # 'A_res50_e100_SDG_cosine_lr3e-05_bs32_m0.8_wd0.0005nol1.json'
    # 'B_swin_e10_Adam_warmcosine_lr3e-05_bs32_wd0.0005FINAL.json',
    # 'B_swin_e6_Adam_warmcosine_lr3e-05_bs32_wd0.0005FINAL.json'
    
    # 'B_swin_e15_Adam_warmcosine_lr3e-05_bs32_wd0.0005.json'
    
    ]

# plots(list)

# plot_report(list, 'weightdecay',['wd=0','wd=0.001'])
# plot_report(list, 'weightdecay',['wd=0.0005 mR=22.6','wd=0.001 mR=24.2'])


# plot_report(list, 'momentum',['m=0.9','m=0.8'], 50)
# plot_report(list, 'learningrate',['lr=1e-05 mR=20.4','lr=5e-05 mR=22.3'])
# plot_report(list, 'learningrate',['lr=1e-05','lr=5e-05'],50)

# plot_report(list, 'white',['No Data Whitening mR=8.1','Data Whitening mR=17.1'])

# plot_report(list, 'l1reg',['l1=0 mR=22.1','l1=0.005 mR=22.6'])
plot_report(list, 'l1reg',['L1 Norm','No L1 Norm'],50)

# plot_report(list, 'learningrate',['lr=1e-05 mR=21.0','lr=5e-05 mR=22.6'])

# plot_report(list, 'aug',['No Augmentation','Image Augmentation'],50)

