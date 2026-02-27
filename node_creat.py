
from config import modelnow, datanow,insertnow, model_info,insert_model
from node_model.res18_cnn import nres18_cnn
from insert_model import CUPID
from data_pre import GLV2


def detemine_model(which_model = modelnow):
    insert_model = determine_insert_model()
    if which_model == 'res18_cnn':
        mainNet = nres18_cnn(model_info,insert_model)
        return mainNet
    else:
        print("no model exist")

def determine_insert_model(which_model = insertnow):
    if which_model == 'CUPID':
        insertNet = CUPID.CUPID(in_channels=insert_model['out_channels'],out_channels=insert_model['out_channels'],
                                             trunck_num=insert_model['trunknum'],linearnum=insert_model['lastsize'])
        return insertNet
    else:
        print("no insert model exist")

def determine_data(which_data = datanow):
    if which_data == 'GLV2':
        dataset = GLV2.preprocess()
        return dataset
    else:
        print("no dataset exist")

