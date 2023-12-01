# vqa-research
Binary answer to the question on the image

![image](vqa_binary.png)

# Prerequisites 

**Backend:** please use *requirements.txt* in order to compile the environment for the application. 

**Model:** the experiments were conducted with `GPU A100 80GB`, `CUDA 11.2` and `torch 1.13.1`. The following libraries must be compatible with this software setup:
```
- torch==1.13.1
- torchvision==0.14.1
- evaluate==0.4.0
- rouge-metric==1.0.1
- transformers==4.34.0
```
All other external libraries, which do not depend on `torch` and `CUDA` versions, are mentioned in `requirements.txt`.

# Launch Instructions

You can use the following commands to control the model settings:

- `--output-dir` -- overwrite output dir.
- `--eval-strat` -- evaluation strategy for training: evaluate every eval_steps.
- `--eval-steps` -- number of update steps between two evaluations.
- `--logging-strat` -- logging strategy.
- `--logging-steps` -- logging steps.
- `--save-strat` -- save strategy.
- `--save-steps` -- save steps.
- `--save-total-limit` -- save only the last n checkpoints at any given time while training.
- `-lr` -- learning rate.
- `-bs` -- batch size on train.
- `-nte` -- num train epochs.
- `--load-best-model-at-end` -- loads the best model based on the evaluation metric at the end of training.
- `--report` -- report results to (wandb).
- `-esp` -- early stopping patience.
- `--random-state` -- random state.
- `--num-device` -- index of device.
- `-expn` -- name of experiment (wandb).
  
### Quick test quide

1) Please, install PyTorch and other libraries in your environment, you can use *requirements.txt*;
2) unzip the dataset;
4) check the paths' constants in `test.py`;
5) launch test sctipt as `python test.py`, you can choose the training hyperparameters using argument-parser.

The metrics will appear after test prrocess bieng finished.

Example for launching with VILT model:

`python test.py --model VILT --logging-steps 200 --batch-size 8 -expn vilt_vqa_model`

### TODO Roadmap
- ✔️ Data processing and EDA
- ✔️ Training module
- ✔️ Inference module, metrics - ROUGE, F1, Accuracy
- ✔️ Fine-tuned [VILT](https://arxiv.org/abs/2102.03334) model
- ✔️ Fine-tuned [BLIP](https://arxiv.org/pdf/2201.12086.pdf) model
- ✔️ Fine-tuned [ROBERTA](https://arxiv.org/pdf/1907.11692.pdf) + [VIT](https://arxiv.org/pdf/2010.11929.pdf) model
- ✔️ Fine-tuned [ROBERTA](https://arxiv.org/pdf/1907.11692.pdf) + [DEIT](https://arxiv.org/pdf/2012.12877.pdf) model
- ⏳ Fine-tune [LLAVA](https://github.com/haotian-liu/LLaVA/tree/main) model 
- ⏳ Fine-tune [KOSMOS-2](https://arxiv.org/pdf/2306.14824.pdf) model

# Contact me

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.
