# CUPID

Code for "CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model".

## Getting Started
The `config.py` file contains the experiment settings for CUPID. To run `test.py`, please first complete the following settings in `config.py`:

1. Download the GLV2 dataset from this link "https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2/" and fill in the `initial_data_path` in `data_all`.
```
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
```
2. Specify the path of the pretrained model in `train_opt`. The provided pretrained model is saved as `save_model_GLV2/pretrain_GLV2_model_mainlast_insert` in link "https://drive.google.com/drive/folders/1U3vQiQwp3bnBeJhQL6tTHw_tUTiSZO1n?usp=drive_link".
```
train_opt = {
    'savedir': 'save_model_{}/'.format(datanow),    # predicted results save dir
    'i_pretrain_savedir': "",   # path of pretrained model
}
```
3. Run `test.py`, and the predicted results will be saved in `save_model_GLV2/`.

## Test the CUPID Model
The notebook `CUPID.ipynb` demonstrates the testing of the AUC metric on the uncertainty estimated by CUPID.

## TODO
- [ ] Add a full training pipeline:




