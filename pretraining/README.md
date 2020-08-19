# Pretraining Residual Networks

This tutorial describes the process of pretraining a residual network on the Icentia11k dataset. If you already have a pretrained network, go to [finetuning](../finetuning) for instructions on how to finetune it to a different task.

### Steps

1. *Download the Icentia11k dataset.* Follow the instructions from the paper <https://arxiv.org/abs/1910.09570>. Let's assume now that the downloaded data is in the `data/icentia11k` directory.

2. *Optionally, unzip the data files.* If you don't mind sacrificing storage for faster i/o, you can unzip the downloaded files. After running the code snippet below, your unzipped data should be in the `data/icentia11k_unzipped` directory.

    ```python
   from transplant.datasets import icentia11k
   from pretraining.utils import unzip_icentia11k
   unzip_icentia11k(
       db_dir='data/icentia11k',
       patient_ids=icentia11k.ds_patient_ids,
       out_dir='data/icentia11k_unzipped',
       verbose=True)
    ```

3. *Run the pretraining job of your choice.* Let's run a beat classification job which will produce output files, such as training history or model checkpoints, that can be found in the `jobs/beat_classification` directory. Notice the `--arch` option that we used to specify ResNet-18 as the architecture that we want to pretrain. For more options, see `pretraining/trainer.py`.

    If you decided not to unzip the files, but rather want to unzip them on the fly during training, then remove the `--unzipped` option and change the `--train` option to `data/icentia11k`. 

    ```shell script
    python -m pretraining.trainer \
    --job-dir "jobs/beat_classification" \
    --task "beat" \
    --train "data/icentia11k_unzipped" \
    --unzipped \
    --arch "resnet18"
    ``` 

4. *Save weights for finetuning.* Now that we have pretrained our ResNet-18, let's extract its weights from a model checkpoint into a separate file which we will later use for finetuning. However, first we need to choose the model checkpoint that we want to get the weights from. A solid choice is a checkpoint that scored good on our validation metric. 

    Since we have trained our network for only one epoch in the step above, which produced only one checkpoint `jobs/beat_classification/epoch_01/model.weights`, we will simply use that checkpoint. After running the code snippet below, our weights will be stored in the `jobs/beat_classification/resnet18.weights` file.

    ```python
   from pretraining.utils import get_pretrained_weights
   resnet18 = get_pretrained_weights(
       checkpoint_file='jobs/beat_classification/epoch_01/model.weights',
       task='beat',
       arch='resnet18')
   resnet18.save_weights('jobs/beat_classification/resnet18.weights')
    ```

5. That's it! See [finetuning](../finetuning) for instructions on how to finetune our pretrained network to a different task.