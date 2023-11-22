# vqa-research
Binary answer to the question on the image

![image](qa_visual_logo.png)

# Prerequisites 

**Backend:** please use *application/requirements.txt* in order to compile the environment for the application. 

**Model:** the experiments were conducted with `CUDA 10.1` and `torch 1.8.1`. The following libraries must be compatible with this software setup:
```
- torch-cluster==1.6.0
- torch-geometric==2.1.0.post1
- torch-scatter==2.0.8
- torch-sparse==0.6.12
- torch-spline-conv==1.2.1
```
All other external libraries, which do not depend on `torch` and `CUDA` versions, are mentioned in `requirements.txt`.

# Launch Instructions

You can use the following commands to control the model settings:

- `-e` -- number of epochs.
- `-lr` -- learning rate.
- `--optimizer-name` -- optimizer name from `torch.optim`.
- `-s` -- the index of the split dividing data to train and test.
- `-bs` -- batch size.
- `-nd` -- device number (e.g., for "cuda:0" it is 0).
- `-o` -- logs saving directory 
- `--alpha` --  coefficient of graph impact (from $0$ to $1$, for images it will be $1 - \alpha$)
- `--alpha-feat` -- it is $\beta$ impact coefficient for auxiliary features priority.
- `--path-blind` -- forces the path-blind mode for model if it is set to `True`, otherwise (`False`) will turn on the path-aware mode.
- `--kfold-filename` -- name of the split file needed to use (specified in the dataset description section).
- `--city`-- name of the city: "Abakan" or "Omsk".
- `--graph-layers` -- number of graph convolution layers.
- `--hidden-size` -- the output size of RegNet and GCN layers. If it is set to $n$,  the input for the transformer encoder will be $2n$ $+$ size of auxiliary features.
- `--linear-size`  -- size of auxiliary features vector.
- `--encoder-layers`  -- number of transformer encoder layers.
- `--fuse-layers` -- number of fine-tuned layers for regression task.
- `--seq-len` -- the fixed length of transformer sequence. Each trip will be truncated or padded regarding this parameter.
- `--graph-input-size`  -- the input vector size for the graph convolution layers.
- `--num-heads` -- number of attention heads in a transformer.
- `--use-infomax` -- if it is set to 1/0, deep graph infomax will be used/not used. Implementation: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/deep_graph_infomax.html 

### Quick test quide

1) Please, install PyTorch and Transformers in your environment;
2) unzip the dataset;
4) check the paths' constants in `test.py`;
5) launch test sctipt as `python test.py`, you can choose the training k-fold split via `-s <n>` option, where `n` is the split number.

The metrics will appear after test prrocess bieng finished.

Example for launching on Abakan:

`python test.py --city Abakan --graph-input-size 73 -s 0 --batch-size 16`

# Contact me

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.
