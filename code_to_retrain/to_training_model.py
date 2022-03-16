import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
cudnn.benchmark = True
cudnn.deterministic = False


def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            print(csv_path)
            df = pd.read_csv(csv_path, sep='\t', engine='python', names=['filename', 'words'], keep_default_na=False, dtype = str)
            print(df['words'])
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        print(characters)
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

opt = get_config("config_files/en_filtered_config.yaml")
train(opt, amp=False)


#'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#sep='^([^,]+)
### Line 149 in /zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/dataset.py has been changed:
# index_col=False is added, according to helpful advice from
# @NinaM31 https://github.com/JaidedAI/EasyOCR/issues/580

