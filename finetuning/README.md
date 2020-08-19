# Finetuning Residual Networks

This tutorial describes the process of finetuning a pretrained residual network for AF classification on the PhysioNet Challenge 2017 dataset. If you don't have a pretrained network yet, go to [pretraining](../pretraining) for instructions on how get one. 

### Steps

1. *Download the PhysioNet Challenge 2017 dataset.* Follow instructions at <https://physionet.org/content/challenge-2017/1.0.0/>. Once downloaded, extract the archive with all recordings into the `data/physionet` directory.

2. *Split the dataset into train and test.* We will compare the pretrained models based on their performance on the test set. Therefore, we split data in 80/20 ratio. Running the code snippet below will produce train (`data/physionet_train.pkl`) and test (`data/physionet_test.pkl`) sets containing numpy arrays stored in a pickle file.
    
    ```python
   from finetuning import datasets
   from finetuning.utils import train_test_split
   from transplant.utils import save_pkl
   data = datasets.get_challenge17_data(
       db_dir='data/physionet',
       fs=250,  # keep sampling frequency the same as Icentia11k
       pad=16384,  # zero-pad recordings to keep the same length at about 65 seconds
       normalize=True)  # normalize each recording with mean and std computed over the entire dataset
   # maintain class ratio across both train and test sets by using the `stratify` argument
   train_set, test_set = train_test_split(
       data, test_size=0.2, stratify=data['y'])
   save_pkl('data/physionet_train.pkl', **train_set)
   save_pkl('data/physionet_test.pkl', **test_set)
    ```

3. *Finetune a pretrained network for AF classification.* Make sure that you've followed the pretraining steps from `pretraining/README.md` and have a pretrained ResNet-18 saved in the `jobs/beat_classification/resnet18.weights` file. Running the shell script bellow will produce the output files such as training history and predictions in the `jobs/af_classification` directory. The script additionally reserves a portion of the train set for validation (5% of the entire dataset). The validation set is used for early stopping. For more options, see `finetuning/trainer.py`.
    
    ```shell script
    python -m finetuning.trainer \
    --job-dir "jobs/af_classification" \
    --train "data/physionet_train.pkl" \
    --test "data/physionet_test.pkl" \
    --weights-file "jobs/beat_classification/resnet18.weights" \
    --val-size 0.0625 \
    --arch "resnet18" \
    --batch-size 64 \
    --epochs 200
    ```

4. *Evaluate the performance of the pretrained network.* Finally, we will evaluate the performance of the pretrained network on the test set in terms of F1 score.

    ```python
   from transplant.evaluation import f1
   from transplant.utils import read_predictions
   test = read_predictions('jobs/af_classification/test_predictions.csv')
   y_true = test['y_true']
   y_prob = test['y_prob']
   print(f1(y_true, y_prob))
    ```