# 3rd Place Solution of the Benetech - Making Graphs Accessible Kaggle Competition

**Authors :** [Theo Viel](https://github.com/TheoViel), [Andrija Milicevic](https://github.com/CroDoc)

### Introduction and details about the solution on [Kaggle](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/418420)

### Prerequisites
- Clone the repository
- Prepare the environment (see Dockerfile-benetech)
- Download the data and extract it in the `data` folder:
  - [Competition data](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data)
- Replace the annotations with manually cleaned ones from:
  - [Cleaned annotations](https://www.kaggle.com/datasets/crodoc/benetech-fixed-train-annotations)
- Run 
  - `cd matcha`
  - `sh run_me.sh`

#### Inference

Inference is done on Kaggle, notebook is [here](https://www.kaggle.com/code/crodoc/benetech-mit-ensemble?scriptVersionId=134055662). The notebook also contains trained checkpoints.

### Code structure
The final code structure (after adding the competition data) should look like this:
```
benetech
├── data
│   ├── benetech-making-graphs-accessible   #competition data
│   │   ├── test
│   │   ├── train
│   │   │   ├── annotations    #replace these annotations
│   │   │   └── images
│   │   └──  sample_submission.py
├── matcha           
└── Dockerfile-benetech
