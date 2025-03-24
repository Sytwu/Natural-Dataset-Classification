# NYCU Computer Vision 2025 Spring HW1
StudentID: 111550159 \
Name: Li-Zhong Szu-Tu (司徒立中)

## Introduction
In this task, the dataset comprises multiple categories of plants, insects, and birds, with a total of 21,024 images for training/validation and 2,344 images for testing.
In my implementation, I utilize the ResNeSt-200 architecture and apply techniques such as CutMix, fine-tuning, cross-validation, periodic merging, and more.

## How to install
How to install dependences
```
conda env create -f environment.yml
conda activate env
```

## How to run
How to execute the code
```
# Training
python main.py

# Inference
python inference.py
```

## Performance snapshot
A shapshot of the leaderboard
![image](https://github.com/user-attachments/assets/d425744c-96ec-4a17-b2d1-39615ae12325)
