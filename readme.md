# SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds

PyTorch implementation of the paper ["SpSequenceNet: Semantic Segmentation Network on 4D Point Clouds"](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_SpSequenceNet_Semantic_Segmentation_Network_on_4D_Point_Clouds_CVPR_2020_paper.html) in CVPR 2020. Vedio Link: [here](https://drive.google.com/file/d/1HYnPqdDZn0cphLkptWlhhFe6PlWCPeNt/view?usp=sharing)


## Dependencies

Python >= 3.6

PyTorch >= 1.3

numpy >= 1.17.2

sparseconvnet >= 0.2

tqdm

## Scripts

Training:

Firstly, the dataset setting is in the data_base and val_base of config.yaml.
Modify it to the direction of your own dataset.
Secondly, run as following:

```
cd train/semanticKITTI
python unet.py
```

Evaluation:

If you are validating your own trainined model, run as following:

```
cd train/semanticKITTI
python val_unet.py
```

If you want to use our trained model, add 'val_model_dir' under 'model' in the config.yaml.
The val_model_dir is the directory of your model.

Our trained model is in [here](https://drive.google.com/file/d/1mgOg9bsozfiXxc5EhtpVAbIcg1BXAbDu/view?usp=sharing)



