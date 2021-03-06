<div align="center">

<samp>
     
# :radioactive:  The MiraiShield Project :radioactive: 

</samp>

| **[ [```Base Dataset```](<http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT>) ]** | **[ [```Project Proposal```](<documentation/IoT Mini Project - Proposal.pdf>) ]** | **[ [```Project Video```](<https://www.youtube.com/watch?v=In_BqB0dU_0>) ]** | **[ [```Project Page```](<https://ram2091999.github.io/IoT-MiniProject/>) ]** | **[ [```Project Report```](<https://docs.google.com/document/d/1PIZ1pvgTJY73DOb_40rkB8c1JKGxfWrxrl535DA9DQw/edit?usp=sharing>) ]** 
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|

<samp>
 
---
     
# Overall Pipeline

<img src="docs/assets/Overall_Task.jpg">
          
---
     
# Model Diagram

<img src="docs/assets/IoT_Botnet_models_new.png">
     
---
     
# Model Stats   
 
<img src="docs/assets/repo_time.png">
     
<img src="docs/assets/repo_size.png">

---
  
# Hardware Flow and Implementation  
 
<img src="docs/assets/hardware_mirai.png">     
     
--- 
     
<img src="docs/assets/hardware.png">     
     
</samp>
     
</div>
     
<samp>  
     
---     

# Project Structure - Detecting IoT Botnet Attacks
```                   
├── README.md  
├── .idea                       
│    ├── .gitignore
│    ├── IoT-MiniProject.iml
│    ├── misc.xml
│    ├── modules.xml
│    ├── vcs.xml
│    └── inspectionProfiles
│       └── profiles_settings.xml
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
│    └── Main_Model_DeepAE.ipynb
├── docs                         # Deploy docs
│    ├── _config.yml
│    ├── index.md
│    └── assets
│       ├── Figure_1.png
│       ├── IoTPipeline.png
│       ├── IoT_Botnet_models.png
│       ├── IoT_Botnet_models_new.png
│       ├── IoT_Modelkey.png
│       ├── abcd.png
│       ├── conf_C.png
│       ├── conf_D.png
│       ├── hardware.png
│       ├── hardware_mirai.png     
│       ├── repo_size.png
│       ├── repo_time.png
│       ├── size_plots.png
│       ├── time_plots.png   
│       └── r.jpeg
├── documentation                # Documentation of Project
│    ├── IoT Mini Project - Proposal.pdf
│    ├── IoT_Botnet_models.png
│    ├── IoT_Modelkey.png
│    ├── model2.png
│    └── nn.svg
├── Model_D                      # Proposed model - Attack Classifier
│    ├── test.py
│    └── train.py
├── hardware                     # Botnet Server and Socket Connections
│    ├── arduino.ino
│    ├── deepLearningCheck.py
│    └── mirai.py     
└── Model_C                      # Proposed model - Anomaly Detector
     ├── test.py
     └── train.py
```     

---    
     
<!-- # EDA Demo and Base Model Interactive Notebook 
<div align="center">
<a href="https://colab.research.google.com/drive/1Ierv-R_v7x1V-qxIzqcekYGGN1EZzqaA?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/ width="200" height="30">
</a>
</div>     -->

</samp>     
     
## Demo
     
<samp>  
     
We provide an easy-to-get-started demo using Google Colab!

</samp> 
      
This will allow you to train a basic version of our method using 
GPUs on Google Colab. 

<div align = "center">
     

| Description      | Link |
| ----------- | ----------- |
| EDA Demo and Base Model Interactive Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ierv-R_v7x1V-qxIzqcekYGGN1EZzqaA?usp=sharing)|   
    
</div>
     
---

<samp>     
     
# References

</samp>
     
```BibTeX
@misc{Dua:2019 ,
author = "Dua, Dheeru and Graff, Casey",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
```
---

```BibTeX
@ARTICLE{8490192,  
author={Meidan, Yair and Bohadana, Michael and Mathov, Yael and Mirsky, Yisroel and Shabtai, Asaf and Breitenbacher, Dominik and Elovici, Yuval},  
journal={IEEE Pervasive Computing},   
title={N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders},   
year={2018},  
volume={17},  
number={3},  
pages={12-22},  
doi={10.1109/MPRV.2018.03367731}}
```

