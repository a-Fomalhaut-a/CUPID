 
datanow = 'GLV2'
modelnow = 'res18_cnn'
insertnow = 'CUPID'

data_all = {
    'GLV2': {
        'name': 'GLV2',
        'initial_data_road': '',    # path of dataset
        'fea_dim':  [512,512],
        'lab_dim': 2,
        'trainnum': 4000*2,
        'testnum': 385*2,
        'valnum': 385*2,
    },
}

data_info = data_all[datanow]

train_opt = {
    'savedir': 'save_model_{}/'.format(datanow),    # predicted results save dir
    'i_pretrain_savedir': "",   # path of pretrained model
}

model_all = {
            'res18_cnn': {
                'name': "res18_cnn",
                'num_classes': data_info['lab_dim'],
                'position':5,
            },
        }

insert_model = {'in_channels':512,'out_channels':512,'trunknum':16, 'lastsize':256, 'output_size':data_info['lab_dim']}

model_info = model_all[modelnow]
