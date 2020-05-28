## Skin Lesion Classification Using Ensembles of Multi-Resolution EfficientNets with Meta Data

Code for team DAISYLab's participation in the ([ISIC 2019 challenge](https://challenge2019.isic-archive.com/)).

We achieved first place in both tasks: ([Leaderboards](https://challenge2019.isic-archive.com/leaderboard.html)).

Arxiv paper: https://arxiv.org/abs/1910.03910

Please cite our MethodsX article if you make use of our work: https://doi.org/10.1016/j.mex.2020.100864

### Usage

Here, we explain the basic usage of our code. Note that we used additional datasets that need to be prepared in a similar way. Most of it is based on our [last year's approach](https://github.com/ngessert/isic2018).

### Data and Path Preparation

The images' and labels' directory strucutre should look like this: /isic2019/images/official/ISIC_0024306.jpg and /isic2019/labels/official/labels.csv. The labels in the CSV file should be structured as follows: first column contains the image ID ("ISIC_0024306"), then the one-hot encoded labels follow.

Other datasets such as the 7-point dataset need to be formatted in a similar way. I.e. there needs to be a "sevenpoint" folder (instead of "official") for the images and a "sevepoint" folder for the labels with the properly fromatted label files.

Our split for training/validation with 5-Fold CV is included in the "indices_isic2019.pkl" file. This should be placed in the same directory as /isic2019. Note that we do not use a test set.

In pc_cfgs we include an example for a machine specific cfg. Here, the base folder can be adjusted for different machines.

In the cfgs folder, there example configs. You can swap out models by using the names given in models.py. For the EfficientNets we used the recommended resolution from the paper --> https://github.com/lukemelas/EfficientNet-PyTorch.

When training a model with additional meta data, you need the prepared meta data file in the meta_data folder. The meta_data folder is structured similar to the images or labels folder (one subfolder for each dataset).

### Training a model

We included two example config files for full training and 5-Fold CV. More details on the different options, e.g. for balancing and cropping, are given in the paper. To start training, run: `python train.py example 2019.test_effb0_ss gpu0` 

gpu0 indicates the number of the GPU that should be used for training. This is helpful for machines with multiple GPUs.

### Evaluate a model 

For model evaluation, there are multiple options. First, a 5-Fold CV model can be evaluated on each held out split. For evaluation of same-sized cropping model (see paper for explanation), run: `python eval.py example 2019.test_effb0_ss multiorder36 average NONE bestgpu0` 

`multiorder36` indicates that ordered, multi-crop evaluation with 36 crops should be performed. Always use 9, 16, 25, 36, 49... etc. number of crops. `average` indicates the predictions should be averaged over the crops (can also be `vote`). `best` indicates that the best model obtained during training should be used. Can be `last` to use the last model saved. 

When evaluating a model with random-resize option (see paper for explanation), run this instead: `python eval.py example 2019.test_effb0_rr multideterm1sc4f4 average NONE bestgpu0`

If final predictions on new, unlabeled images should be performed, add the path to said images at the end of the evaluation call: `python eval.py example 2019.test_effb0_ss multiorder36 average NONE bestgpu0 NONE /home/Gessert/data/isic/isic2019/images/Test` 

Each evaluation run generates a pkl file that can be used for further ensemble aggregation.

### Construct an Ensemble

Testing ensembles is also split into two parts. First, an ensemble can be constructed based on 5-Fold CV error and the corresponding best models are saved. Then, the final predictions on a new dataset can be made using the generated files from the evaluation section.

For 5-Fold CV performance assessment, run: `python ensemble.py /path/to/evaluation/files evalexhaust15 /path/to/file/best_models.pkl`
The first path indicates the location where all evaluation pkl files are located. `evalexhaust15`: `eval` indicates that 5-Fold CV evaluation is desired. `exhaust15` indicates that the top 15 performing models should be tested for their optimal combination. I.e., every possible combination (average predictions) of those models is tested for the best performance. Without the exhaust option, only the top N combinations are considered, i.e., the tested combinations are: top1 model, top1+top2 model, top1+top2+top3 model, etc. The last argument indicates the path where the best performing combination is saved.

For generation of new predictions for unlabeled data, run: `python ensemble.py /path/to/evaluation/files best /path/to/file/best_models.pkl /path/to/predictions.csv /path/to/image/files`
`best` indicates that only the models with best in the name should be considered. This relates to the evaluation where either the best performing model or the last checkpoint can be used for generation. This can be `last` or `bestlast` for both. The next argument is the path to the file that was generated in the first ensemble run. This can just be `NONE` if all models should be included. The next argument is the path to the CSV file that should contain the predictions. The last argument is the path to the image files which is used to match the predictions to image file names.
