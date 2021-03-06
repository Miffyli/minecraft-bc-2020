# Playing Minecraft with behavioural cloning, 2020 edition
This repository contains the final ranked, imitation-learning only of NoActionWasted team to [MineRL 2020 challenge](https://www.aicrowd.com/challenges/neurips-2020-minerl-competition),
reaching fourth place.

Team members: [Anssi Kanervisto](https://github.com/Miffyli), [Christian Scheller](https://github.com/metataro/) and [Yanick Schraner](https://github.com/YanickSchraner).

Core ingredients:
- Written in PyTorch.
- In the competition actions were obfuscated into continuous vectors. We used k-means clustering on human dataset to create representative actions, which works quite well (thanks to [organizers for suggesting this method](https://minerl.io/docs/tutorials/k-means.html)).
  - Continuous action vectors are mapped to the closest centroids, and all learning is done on these discrete actions.
- Behavioural cloning (i.e. train network to predict what actions human takes).
- Network model: Small ResNet with LSTM layers (see [IMPALA's network](https://arxiv.org/abs/1802.01561) for a similar architecture)

## Manual data cleaning for ObtainDiamond dataset

We also ran experiments by manually going through the ObtainDiamond dataset and classifying games by their quality.
We did not use this data in the competition, only to understand better the quality of the data.
Only including succesfull games from ObtainDiamond we obtained significantly
higher results (20 and above, vs. 12-20 here.). This filtering is _not_ included in this code.

This data is included in the `MineRL-ObtainDiamond-labels.csv`, and also available on [Google Sheets](https://docs.google.com/spreadsheets/d/1XqI5dIQEvmfSzujHL7aom1GTEJbwDK9zuRT7_HbPAfQ).

## Code contents

Code is in the submission format, and can be ran with the instructions at [submission template repository](https://github.com/minerllabs/competition_submission_template).
`environment.yml` contains an Anaconda environment specifications, and `apt.txt` includes any Debian packages required (used by the Docker image in AICrowd evaluation server).

The core of our submission resides in `train_bc_lstm.py`, which contains the main training loop. 

## Running

[Download](http://minerl.io/dataset/) and place MineRL dataset under `./data`. Alternatively point environment variable `MINERL_DATA_ROOT` to the downloaded dataset.

Run `train.py` to train the model. Afterwards run `test.py` to run the evaluation used in the AICrowd platform. This code prints out per-episode rewards.

After 200 games, the average episodic reward should be between 12-20 (the variance between results is rather high).
