# vqa-research
Binary answer to the question on the image

![image](qa_visual_logo.png)

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
- `--save-total-limit` -- # Save only the last n checkpoints at any given time while training.
- `-lr` -- learning rate.
- `-nte` -- num train epochs.
- `--load-best-model-at-end` -- loads the best model based on the evaluation metric at the end of training.
- `--report` -- report results to (wandb).
- `-esp` -- early stopping patience.
- `--random-state` -- random state.
- `--num-device` -- index of device.
- `-expn` -- name of experiment (wandb).
  

  
### Quick test quide

1) Please, install PyTorch and Transformers in your environment;
2) unzip the dataset;
4) check the paths' constants in `test.py`;
5) launch test sctipt as `python test.py`, you can choose the training k-fold split via `-s <n>` option, where `n` is the split number.

The metrics will appear after test prrocess bieng finished.

Example for launching on Abakan:

`python test.py --city Abakan --graph-input-size 73 -s 0 --batch-size 16`

### TODO Roadmap
- ✔️ Adaptive truncation of history
- ✔️ User settings in inline keyboard
- ✔️ Thorough token package, usage limit
- ⏳ Reminder system / user interest tracking
- ⏳ Non-passive conversation (sending trigger messages from time to time)
- ⏳ Ability to understand voice messages
- ⏳ Ability to generate images
- ⏳ Ability to answer with voice messages

# Contact me

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.
