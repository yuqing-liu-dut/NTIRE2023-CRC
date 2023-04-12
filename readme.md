# CRC: A 400+ FPS Real-Time Image Super-Resolution Network for 4K Images
This is our solution of NTIRE 2023 Real-Time Super-Resolution- Track 1 (X2), which wins the 6th place.

# Model Details

1080P to 4K, runtime tested on 3090/3060 (FP16)

Parameters: 0.828 K

FLOPs: 1.6921 G

Runtime: 2.38 ms

# Usage
python xswl.py <LRImage.png> <SRImage.png>

# Why CRC
Because this model is composed of one **C**onvolution, one **R**eLU activation and one **C**onvolution!!!
