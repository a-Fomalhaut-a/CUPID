# CUPID

Code for "CUPID: A Plug-in Framework for Joint Aleatoric and Epistemic Uncertainty Estimation with a Single Model".

## Getting Started
The `config.py` file contains the experiment settings for CUPID. To run `test.py`, please first complete the following settings in `config.py`:

1. Download the EyePACS dataset from this link and fill in the `initial_data_path` in `data_all`.
```
data_all = {
    'EyePACS': {
        'name': 'EyePACS',
        'initial_data_road': '',    # path of dataset
        'fea_dim':  [512,512],
        'lab_dim': 2,
        'trainnum': 4000*2,
        'testnum': 385*2,
        'valnum': 385*2,
    },
}
```
2. Specify the path of the pretrained model in `train_opt`. The provided pretrained model is saved in `save_model_EyePACS/EyePACS_model_mainlast_insert`.
```
train_opt = {
    'savedir': 'save_model_{}/'.format(datanow),    # predicted results save dir
    'i_pretrain_savedir': "",   # path of pretrained model
}
```
3. Run `test.py`, and the predicted results will be saved in `save_model_EyePACS/`.

## Test the CUPID Model
The notebook `CUPID.ipynb` demonstrates the testing of the AUC metric on the uncertainty estimated by CUPID.

