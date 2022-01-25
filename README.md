# 1. Summary
This task is to improve the performance of text detection, which detects letters in an image at OCR(Optical Character Recognition) to find areas of letters. To improve the performance of Text detection, we first identified the characteristics of the given eval_data and built additional English and Korean data from standard datasets and synthetic datasets so that learning can proceed centering on data mixed with Korean and English. In addition, we implemented code for learning by referring to the paper of the CRAFT model, a character-level detection method that is up-to-date and seems to be suitable for a given environment, and improved performance through hyper-parameter and dataset adjustment.

 
# 2. Experimental Results
Model                      |dataset                     |Recall |precision | F1   | checkpoints
---------------------------|----------------------------|-------|----------|------|-------------
EAST                        |ICDAR17_Korean              |0.3830 |0.6729    |0.4881 | [link](https://drive.google.com/file/d/13c1FU1TpCgXtzhUQ4PLPi0-VXlU5m7HS/view?usp=sharing)              
EAST                       |ICDAR13,15,17,19            |0.4215 |0.6673    |0.5166  | [link](https://drive.google.com/file/d/1_pjaaVz_puefa_q8Xf0nuUMYFblcPupu/view?usp=sharing)           
CRAFT without finetuning   |SynthText                   |0.4600 |0.6364    |0.5340 | [link](https://drive.google.com/file/d/1YXToxjcx7zG5bACx-nrJ_ddtgBzH1fcr/view?usp=sharing)                            
CRAFT with finetuning      |SynthText, ICDAR15           |0.4879  |0.7266    |0.5838 | [link](https://drive.google.com/file/d/12ajKSHscwLMd1tQWRJWMJwJylL3-Mvu3/view?usp=sharing)                        
CRAFT with finetuning      |SynthText, ICDAR{17,19}_korean |0.6158|0.8576     |0.7168        | [link](https://drive.google.com/file/d/1YXToxjcx7zG5bACx-nrJ_ddtgBzH1fcr/view?usp=sharing)

- ICDAR{17,19}_Korean dataset is a dataset reconstructed by collecting Korean samples from ICDAR{2017,2019}-MLT.
- The test result is an experimental result based on text detection data containing Hangul, not ICDAR.

# 3. Environment settings
## Requirements :
- python==3.7.0
- pytorch==1.5.0
- torchvision==0.6.0
- opencv-python==4.5.4.58
- scipy==1.7.3
- scikit-image==0.18.3
- shapely==1.7.1
- Polygon3==3.0.9.1
- tqdm

```bash
pip install -r requirements.txt
```

##  Directory Structure
```
CRAFT/
│
├──train.py - main script to start training
├──finetuning.py - main script to start finetuning
├──val.py - evaluation of trained model
├──test.py - inference trained model and get output.csv,images include bboxes 
│
├──craft.py - CRAFT model
├──craft_utils.py
│
├──file_utils.py
├──data_loader.py - dataloader for ICDAR
├──convert_ICDAR_to_UFO.py - convert ICDAR data to UFO format json file
├──mep.py
├──watershed.py - watershed algorithm that get char bbox
│
├──basenet/
│   └──vgg16_bn.py
│
├──data/ 
│   ├──boxEnlarge.py
│   ├──dataset.py - dataloader for SynthText 
│   ├──imgaug.py
│   ├──imgproc.py
│   ├──load_icdar.py - load ICDAR ground truth label
│   ├──pointColockOrder.py
│   └──SynData.py
│
├──utils/
│   └──inferences_boxes.py   
│
├──loss/
│   └──mseloss.py 
│
└──metrics/
    └──eval_det_iou.py - calculate F1,precision,recall

```
## Instruction
### Run training code for text detection with Synthtext 
```bash
python train.py --Synth_Dir=$PATH_TO_SYNTHTEXT
```

### Run finetuning code for text detection with Synthtext and ICDAR
```bash
python finetuning.py --Synth_Dir=$PATH_TO_SYNTHTEXT --ICDAR_Dir=$PATH_TO_ICDAR --checkpoint=$WEIGHT_FILE 
```

### Run test
```bash
python test.py --trained_model=$WEIGHT_FILE --test_folder=$PATH_TO_TEST_DATA
```

# Approach
Unlike top-down approach in text detection, which directly recognizes words in images, the CRAFT model follows bottom-up approach, which recognizes characters first and connects characters to obtain words.

The CRAFT model predicts the probability region score, which is the center of the character, for each pixel of the image, and the probability score, which is the center of two adjacent characters, resulting in character boxes and word boxes, and is the best performance model in most standard datasets. 

CRAFT goes through a two-step learning process because there is no character-level annotation in standard datasets such as ICDAR. 
First, ground-truth labeling for learning is performed using the SynthText database, which is notated by character-level. 
Create Affinity boxes with Character boxes, apply a 2D Gaussian map to each box to assign them to locations corresponding to the image, and use them as ground-truth of Affinity score and region score to learn the model primarily. 
However, learning only with SynthText dataset is not fully applied to data used in real-world, so additional learning is performed by obtaining pseudoground-truth from ICDAR, an existing standard dataset, and others through this learned model. 
Once the character area is cropped with word-level presentation, the trained model is used to create the pseudo ground-truth of the region score and the affinity score. 
After that, divide the characters using the watershed algorithm and determine the character box. 
In addition, confidence score is obtained to determine whether pseudo GT negatively affects learning and used for fine-tuning of the model. 
During Fine-tuning, Synthtext data and ICDAR data were used together and batch was configured at a ratio of 1:5.


# References
- Zhou, Xinyu, et al. "East: an efficient and accurate scene text detector." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017.

- Baek, Youngmin, et al. "Character region awareness for text detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

- Gupta, Ankush, Andrea Vedaldi, and Andrew Zisserman. "Synthetic data for text localisation in natural images." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

- Karatzas, Dimosthenis, et al. "ICDAR 2015 competition on robust reading." 2015 13th International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2015.

- https://github.com/clovaai/CRAFT-pytorch

- https://github.com/backtime92/CRAFT-Reimplementation

- https://aihub.or.kr/aidata/133
