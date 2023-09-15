# PersonalityDetection

#### ***ADF : An Attention-based Denoising Framework for Personality Detection from Social Media Text***

### Dataset
- Essay-BigFive (https://github.com/SenticNet/personality-detection)
- Twitter-MBTI (https://www.kaggle.com/datasnaek/mbti-type)

### Performance
Twitter-MBTI Dataset (10.2% average accuracy improvement)

| Model  | E/I  | S/N  | T/F  |  J/P | Ave  |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  Baseline |  77.0 | 85.3  | 54.1  | 60.4  | 69.2  |
| Mehta et al | 78.8  | 86.3  | 76.1  | 67.2  |  77.1 |
|  SOTA | 83.4  | 85.6  | 82.2  | 76.3  | 81.9  |
|  ADF |  92.1 | 93.6  |  **89.7** | 91.2  |  91.7 |
|  ADF (&Multi-task) | **92.7**  |  **94.7** | **89.7**  | 91.4  | **92.1** |
|  ADF (&Self-Ens.) | 92.4  |  94.2 | **89.7**  | **91.6**  | 92.0  |

Essay-BigFive Dataset

| Model  | EXT  | NEU  | AGR  |  CON | OPN |Ave |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  Baseline |  51.7 | 50.0   |  53.0  |  50.8  |  51.5  |  51.4  |
| Majumder et al  |  58.1  |  57.3  | 56.7   |  56.7   | 61.1   | 58.0   |
|  SOTA |  61.1  |   62.2  | 60.8   |  59.5 |  65.6  |  61.9  |
|  ADF |  61.8  |  61.9  |  60.7  | 60.9  |  66.4  |  62.3  |
|  ADF (&Multi-task) |  63.1  |  63.6 |  61.4  |  **61.2** |  66.4 |  63.1  |
|  ADF (&Self-Ens.) |  **63.6**  |  **63.8**  |  **61.5** |  61.0  | **66.7** |   **63.3** |

### Requirements
- python 3.8.5
- pytorch 2.0.1
- transformers 4.29.2
- fairseq 0.12.2

### Preprocessing
#### data preprocess
1.  generate original data files : `python data-prepare/process_twitter.py`;
2.  binarize training data : `sh preprocess_PD_tasks.sh`.

#### model prepare
1. download RoBERTa-base model from https://huggingface.co/roberta-base/tree/main;
2. put `model.pt` under `download/roberta/`.

### Training
1. make sure fairseq (https://github.com/facebookresearch/fairseq) has been correctly installed;
2. create config files (.yaml) and set training hyper-parameters;
3. code `hydra_train_PD_tasks.sh` and start training by `sh hydra_train_PD_tasks.sh`.

#### fairseq mainly modified
- Model: `fairseq/models/personality_detection/`
- Criterions: `fairseq/criterions/personality_detection.py`
- Task: `fairseq/tasks/personality_detection.py`


### Sketch Map of ADF

<img src="https://github.com/Once2gain/PersonalityDetection/assets/66986397/251f87c4-3012-42c2-a333-ac7a93bfa65f" width="450px">





