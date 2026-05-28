from config import datanow, modelnow, insertnow, data_all, data_info, model_all, model_info, insert_model


train_opt = {
    'seed_value': 2024,

    'studyratio': 1e-4,
    'studyratio_ep': 1,
    'studyratio_de': 0.8,
    'epoch': 5,
    'batchsize': 4,
    'savename': datanow + '_model_main',
    'savedir': 'save_model_{}/'.format(datanow),
    'saveepoch': 5,
    'pretrainflag': False,
    'pretrain_savedir': '',

    'i_studyratio': 1e-5,
    'i_studyratio_ep': 1,
    'i_studyratio_de': 0.9,
    'i_epoch': 15,
    'i_batchsize': 8,
    'i_loss_para1': 1.0,
    'i_loss_para2': 9e-3,
    'i_loss_para3': 0.01,
    'i_savename': datanow + '_model_de',
    'i_savedir': 'save_model_{}/'.format(datanow),
    'i_saveepoch': 2,
    'i_pretrainflag': True,
    'i_pretrain_savedir': '',
}
