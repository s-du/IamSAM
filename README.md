# IamSAM
A small project for comparing the outputs of SAM (Segment Anything [[1]](#1)) and FastSAM [[2]](#2), with a practical GUI (Pyside6)

**The project is still in pre-release, so do not hesitate to send your recommendations or the bugs you encountered!**

<a href="https://ibb.co/JyspQFK"><img src="https://i.ibb.co/BrC6NKt/Sam.png" alt="Sam" border="0"></a>

## Principle
The user can upload any picture, then add 'positive' and 'negative' points that will be used as prompt in FASTSAM or SAM algorithm.
The tool creates a small output mask (jpg) in the folder, and allow the user to directly visualize the segmentation results.

## Installation instructions
1. Clone the repository:
```
git clone https://github.com/s-du/IamSAM
```

2. Navigate to the app directory:
```
cd IamSAM
```
3. (Optional) Install and activate a virtual environment

   
4. Install the required dependencies:
Please note that SEGMENT ANYTHING requires a CUDA environment (the user must manually install torch with the appropriate CUDA version).
Models checkpoints should be downloaded manually and placed in resources/other:
- FastSAM-x.pt
- sam_vit_h_4b8939.pth

6. Run the app:
```
python main.py
```

## References

<a id="1">[1]</a> See https://github.com/facebookresearch/segment-anything

<a id="2">[2]</a> See https://github.com/CASIA-IVA-Lab/FastSAM

```
@misc{zhao2023fast,
title={Fast Segment Anything},
author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
year={2023},
eprint={2306.12156},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
