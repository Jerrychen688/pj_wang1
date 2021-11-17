
## Att-HyperGCN-pytorch

This is the Pytorch implementation for our paper:

Author: Peijie Wang (peijie_wang@snnu.edu.cn)


## Introduction

In this work, we aim to apply GAT and HyperGCN to make it more concise and appropriate for relation prediction. We propose a new model named Att-HyperGCN.


## Enviroment Requirement

`pip install -r `

## Dataset

Fixed dataset: FB-AUTO
Unfixed dataset: JF17K-4, WikiPeople-4

## Usage

To train `Att-HyperGCN` you should define the parameters relevant to the given model.
The default values for most of these parameters are the ones that were used to obtain the results in the paper.

- `model`: The name of the model. 

- `dataset`: The dataset you want to run

- `batch_size`: The training batch size.

- `num_iterations`: The total number of training iterations.

- `lr`: The learning rate.

- `nr`: number of negative examples per positive example for each arity. 

- `emb_dim`: embedding dimension.

- `hidden_drop`: drop out rate for hidden layer of all models.

- `no_test_by_arity`: when set, test results will not be saved by arity, but as a whole. This generally makes testing faster. 设置后，测试结果将不会被保存，而是整体保存。 通常，这会使测试更快。

- `test`: when set, this will test a trained model on the test dataset. If this option is present, then you must specify the path to a trained model using `-pretrained` argument.当设置时，它将测试测试数据集上的训练过的模型。如果存在此选项，则必须使用' - pretraining '参数指定经过训练的模型的路径。

- `pretrained`: the path to a pretrained model file. If this path exists, then the code will load a pretrained model before starting the train or test process.
The filename is expected to have the form `model_*.chkpnt`. The directory containing this file is expected to also contain the optimizer as `opt_*.chkpnt`, if training is to resume. 

- `output_dir`: the path to the output directory where the results and models will be saved. If left empty, then a directory will be automatically created.保存结果和模型的输出目录的路径。 如果保留为空，则将自动创建目录。

- `restartable`: when set, the training job will be restartable: it will load the model from the last saved checkpoint in `output_dir`, as well as the `best_model`, and resume training from that point on.
If this option is set, you must also specify `output_dir`.设置后，训练作业将可重新启动：它将从上次保存的`output_dir`和`best_model`的检查点中加载模型，并从该点开始继续训练。如果设置了此选项，则还必须指定ʻoutput_dir`。
  
## Training  
You can train `Att-HyperGCN` by running the following step:
```
python main.py -model Att-HyperGCN -dataset FB-AUTO -num_iterations 1200 -batch_size 256 -lr 0.001  -emb_dim 200 -nr 10
```

## Testing a pretrained model
You can test a pretrained model by running the following:
```console
python main.py -model Att-HyperGCN -dataset FB-AUTO -pretrained output/my_pretrained_model.chkpnt -test
```


