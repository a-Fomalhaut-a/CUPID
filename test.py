
from data_read import *
from config import *
from node_creat import detemine_model, determine_data
import shutil
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from Log import Logger
from util import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#-----------root and log---------------
# CREAT DIR
ATIME = datetime.now().strftime("%Y%m%d_%H%M%S")
SDR = train_opt['savedir'] + 't' + ATIME + 'xx'
if not os.path.exists(SDR):
    os.makedirs(SDR)

# config loggers. Before it, the log will not work
log = Logger(SDR+'/train.log', level='info')

class Node_model():
    def __init__(self, savedir):
        super(Node_model, self).__init__()
        self.mainnet = detemine_model()
        self.savedir = savedir

    def e_load(self, loaddir):
        log.logger.info("---------------------------------load pretrained main model---------------------------")
        log.logger.info("load path:"+loaddir)

        if self.device != 'cpu':
            self.mainnet.cuda()
        pr_dict = torch.load(loaddir, map_location=self.device)

        model_dict = self.mainnet.state_dict()
        pretrained_dict = {k: v for k, v in pr_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.mainnet.load_state_dict(model_dict)

    def compute_var_uncertaintyff(self,sigma):
        return torch.exp(sigma)

    def in_test(self, dataset, preflag, insertflag=True, dataflag = 'test'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device != 'cpu':
            self.mainnet.cuda()

        # load pretrain model
        if preflag:
            self.e_load(train_opt['i_pretrain_savedir'])

        test_loader = datacreate(dataset, flag=dataflag)
        log.logger.info('-------------------------------------------------')
        log.logger.info('insert validation begin')
        self.mainnet.eval()

        softreslist,varusum = [], []
        picnamelist,labellist,predlist,predlistinit,rightlist = [],[],[],[],[]

        save_excel = True

        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for step, data in enumerate(test_bar):
                feature, label = data
                picname = (test_loader.dataset.imgs[step])[0].split('/')[-1]
                feature = feature.to(self.device)
                outinit, _ = self.mainnet(feature)
                outlast, edresult = self.mainnet(feature,insertflag)
                _, preds = torch.max(outlast, 1)
                _, predsinit = torch.max(outinit, 1)

                # Uncertainty compute ===================================================================
                softres = float(np.array(((torch.abs(torch.softmax(outlast,dim=1)-torch.softmax(outinit,dim=1))).cpu()).mean()))
                varu = self.compute_var_uncertaintyff(edresult[0][0])

                # List load ===================================================================
                picnamelist.append(picname)
                labellist.append(np.array(label.cpu())[0])
                predlist.append(np.array(preds.cpu())[0])
                predlistinit.append(np.array(predsinit.cpu())[0])
                rightlist.append(np.abs(np.array(preds.cpu())[0]-np.array(label.cpu())[0]))
                softreslist.append(softres)
                varusum.append(np.array(varu[0].mean().detach().cpu()))

        # save excel
        if save_excel:
            sheet = [picnamelist, labellist, predlist, rightlist, softreslist,varusum]
            colname = ["picnamelist","actual","predlist","wronglist","softoutres","var_usum"]
            df = pd.DataFrame(sheet).T
            datasave_dir = self.savedir
            df.to_csv(datasave_dir + '/cupid.csv', index=False, header=colname)

        log.logger.info("result initial ***************************")
        log.logger.info(classification_report(labellist, predlistinit, digits = 4))

        log.logger.info("result after insert ***************************")
        log.logger.info(classification_report(labellist, predlist, digits = 4))

        return 0


def test_only_in(savedirroot):
    # Data
    log.logger.info("++++++ val insert ++++++")
    dataset = determine_data()
    savedir = savedirroot+"/in_model"
    os.makedirs(savedir)
    Net = Node_model(savedir)
    # Save config
    shutil.copy('config.py', savedir)
    # Val
    Net.in_test(dataset, preflag = True)

if __name__ == '__main__':
    log.logger.info(dict2str(data_info))
    log.logger.info(dict2str(model_info))
    log.logger.info(dict2str(train_opt))

    test_only_in(SDR)
