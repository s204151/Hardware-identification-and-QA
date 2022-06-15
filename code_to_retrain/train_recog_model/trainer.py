import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
import wandb

cudnn.benchmark = True
cudnn.deterministic = False
#cudaSetDevice(1)



def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='\t', engine='python', names=['filename', 'words'], index_col=False, keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        print(True)
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

opt = get_config("config_files/en_filtered_config.yaml")

num_rounds = 1

opt.num_iter = 30000
opt.valInterval = 1000
opt.lr = 0.001
opt.weight_decay = 0.0005

for round_nr in range(num_rounds):
    opt.train_data = 'all_data/clean_cropped_en_train_split_' + str(round_nr)
    opt.valid_data = 'all_data/clean_cropped_en_val_split_' + str(round_nr)
    wandb.init(
        project=str(
            opt.Transformation + "-" + opt.FeatureExtraction + "-" + opt.SequenceModeling + "-" + opt.Prediction + "_" + "1"),
        name = f"experiment_{round_nr}",
        config={
            "learning_rate": opt.lr,
            "beta1": opt.beta1,
            "rho": opt.rho,
            "weight_decay": opt.weight_decay,
            "eps": opt.eps,
            "num_iter": opt.num_iter,
            "batch_size": opt.batch_size,
            "imgH": opt.imgH,
            "imgW": opt.imgW,
            "manualSeed": int(opt.manualSeed + round_nr),
        })

    train(opt, amp=False,round_nr = round_nr)

    # api = wandb.Api()
    # run = api.run(f"otovo-dtu-qa/{wandb.config.project}/{wandb.config.name}")
    #
    # validation_loss = run.history()["val loss"]
    # validation_accuracy = run.history()["val accuracy"]
    #
    # print(validation_loss)
    # print(validation_loss)
    # print(validation_loss)
    # print(validation_loss)
    # print(validation_loss)
    # print(validation_loss)
    # print(validation_loss)

















