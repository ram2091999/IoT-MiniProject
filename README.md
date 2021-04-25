<div align="center">

# Detection of IoT Botnet Attacks - The MiraiShield Project

| **[ [```Initial Dataset```](<http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT>) ]** | **[ [```Project Proposal```](<documentation/IoT Mini Project - Proposal.pdf>) ]** 
|:-------------------:|:-------------------:|

# Model Diagram

<img src="documentation/IoT_Botnet_models.png">

</div>

# Project Structure
```                   
├── README.md                   
├── EDA                          # EDA, NAS and Self-Organising Maps
│    ├── bilinear.py
│    ├── features.py
│    ├── nas.py
│    └── som.py
├── Model_A                      # Base model #1
│    └── model.py
├── Model B                      # Base model #2
│    └── model.py
├── colab                        # Playground for notebooks
│    ├── IOT_IntrusionDetection_EDA_BaseModels.ipynb
│    └── Main_Model_Deep_AE.ipynb
├── docs                         # Deploy docs
│    ├── _config.yml
│    ├── about.md
│    └── index.md
├── documentation                # Documentation of Project
│    ├── IoT Mini Project - Proposal.pdf
│    ├── IoT_Botnet_models.png
│    ├── IoT_Modelkey.png
│    ├── model2.png
│    └── nn.svg
└── Model_C                      # Proposed model - Baseline
     ├── test.py
     └── train.py
```     




