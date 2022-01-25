# 1. Summary
 본 task는 이미지 속 글자를 검출하여 글자의 영역을 찾아내는 text detection의 성능을 개선하기 위한 task입니다.
Text detection의 성능을 개선하기 위하여 먼저 주어진 eval_data의 특성을 파악해 한국어와 영어가 섞인 데이터를 중심으로 학습이 진행될 수 있도록 표준 데이터셋과 합성 데이터셋으로부터 영어와 한글 데이터를 추가로 구축하였습니다. 
또한, 기존에 주어진 EAST보다 최신 모델이면서 주어진 환경에 적합할 것으로 보이는 character-level detection method인 CRAFT 모델의 paper를 참조하여 학습을 위한 코드를 구현하였고, hyper-parameter와 데이터셋 조정 등을 통해 성능을 개선시켰습니다.   

 
# 2. Experimental Results
Model                      |dataset                     |Recall |precision | F1   | checkpoints
---------------------------|----------------------------|-------|----------|------|-------------
EAST (baseline)                       |ICDAR17_Korean              |0.3830 |0.6729    |0.4881 | [link](https://drive.google.com/file/d/13c1FU1TpCgXtzhUQ4PLPi0-VXlU5m7HS/view?usp=sharing)              
EAST (baseline)                      |ICDAR13,15,17,19            |0.4215 |0.6673    |0.5166  | [link](https://drive.google.com/file/d/1_pjaaVz_puefa_q8Xf0nuUMYFblcPupu/view?usp=sharing)           
CRAFT without finetuning   |SynthText                   |0.4600 |0.6364    |0.5340 | [link](https://drive.google.com/file/d/1YXToxjcx7zG5bACx-nrJ_ddtgBzH1fcr/view?usp=sharing)                            
CRAFT with finetuning      |SynthText, ICDAR15           |0.4879  |0.7266    |0.5838 | [link](https://drive.google.com/file/d/12ajKSHscwLMd1tQWRJWMJwJylL3-Mvu3/view?usp=sharing)                        
CRAFT with finetuning      |SynthText, ICDAR{17,19}_korean |0.6158|0.8576     |0.7168        | [link](https://drive.google.com/file/d/1YXToxjcx7zG5bACx-nrJ_ddtgBzH1fcr/view?usp=sharing)

- ICDAR{17,19}_Korean 데이터셋은 ICDAR {2017,2019}-MLT에서 한글인 샘플을 모아서 재구성한 데이터셋입니다.


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
먼저 본 task가 목표로 하고 있는 text detection의 성능을 높이기 위해 주어진 eval_data의 특성을 파악하고자 하였습니다. 
Eval_data는 real-world scene에서 영어 및 한국어 글자가 포함된 이미지로 구성되어 있으며, 전체 영역에서 글자가 차지하는 비율이 아주 작은 상황과 장난감에 새겨진 것과 같이 글자의 색이나 특성 등이 배경과 차이가 나지 않는 상황 등이 다수 포함되어 있습니다. 
이를 위해 기존에 OCR을 위해 나온 표준 데이터셋인 ICDAR dataset만을 사용하기 보다는 Synth dataset을 같이 활용하는 것이 좋겠다고 판단하였습니다. 
Synth dataset은 image dataset에 다양한 폰트와 크기를 가진 글자를 합성하여 만든 글자 검출 및 인식을 위한 합성 데이터셋입니다. 배경과 유사한 색상의 글자가 포함된 경우가 많고 한 단어에 다양한 크기의 글자가 포함될 수 있도록 굴곡을 주는 등의 변형이 포함되어 있어 ICDAR dataset보다 더 어려운 환경을 가지고 있습니다. 
또한, word-level과 character-level bounding boxes annotation을 모두 포함하고 있어 character-level detection이나 word-level detection이 모두 가능하다는 장점을 가지고 있습니다. 
그 다음으로 고려했던 부분은 모델입니다. 주어진 baseline은 CVPR 2017에 나온 EAST라는 모델인데,  EAST는 Fully Convolutional Network (FCN)을 이용해 word나 text-level prediction을 중간 단계없이 진행한 후 Non-Maximum Suppression (NMS)를 통해 최종 결과를 산출하는 빠르고 정확한 text detector 입니다. 
그러나 eval_data에는 다양한 크기의 글자가 한 단어를 이루는 경우가 많아서 CVPR 2019에 나온 character-level detection method인 [Character Region Awareness for Text detection](https://arxiv.org/abs/1904.01941) 논문의 CRAFT 모델도 EAST 모델과 함께 선정하였습니다.

먼저 baseline으로 주어진 EAST 모델을 여러 학습 데이터셋을 모아서 학습을 진행했는데, ICDAR 2017의 한국어 데이터셋이나 표준 데이터셋을 모두 합친 데이터셋으로 학습해도 50% 전후의 성능을 보였습니다. 
문자 검출이기 때문에 다양한 언어의 학습 데이터셋을 많이 사용하면 검출에 도움이 될 것이라고 생각했는데, 아랍어나 중국어 등의 다른 언어들은 한국어와 영어를 검출하는데 큰 도움이 되지는 못했던 것 같습니다. 
특히 단어 내의 글자 크기가 다양한 환경에 대해서는 거의 대처를 하지 못하는 문제점이 있어서 character-level detector인 CRAFT 모델을 학습시키는 데에 집중했습니다.
CRAFT 모델은 이미지에서 직접적으로 단어를 인식하는 top-down approach와는 다르게 character들을 먼저 인식하고 문자들을 연결하여 단어를 얻는 bottom-up approach를 따릅니다.
CRAFT 모델은 이미지의 각 픽셀에 대해 character의 중심인 확률 region score와 인접한 두 문자의 중심일 확률인 affinity score를 예측하여 character boxes, word boxes를 최종 결과로 얻게 하며 대부분의 표준 데이터셋에서 가장 좋은 성능을 얻은 모델입니다. 
Github을 통해 official 코드가 공개되어 있지만 pre-trained checkpoint와 모델, test 코드만 제공할 뿐 학습을 위한 코드 및 detail은 공개되어 있지 않아서 paper의 내용과 github issues 등의 내용을 참조해 학습을 위한 코드를 구현하였습니다. 

CRAFT는 ICDAR과 같은 표준 데이터셋에 character-level annotation이 없어 두 단계의 학습 과정을 거칩니다. 
먼저 character-level로 annotation이 되어있는 SynthText dataset을 이용하여 학습을 위한 ground-truth labeling을 진행합니다. 
Character box들로 affinity box들을 생성해주고, 각각의 박스들에 2D gaussian map을 적용하여 이미지에 대응되는 위치에 할당하고 이것들을 affinity score와 region score의 ground-truth로 사용해 모델을 1차적으로 학습합니다. 
그러나 합성 데이터셋인 SynthText dataset으로만 학습하는 것은 real-world에서 사용하는 데이터에 완벽하게 적용되지 않아 이렇게 학습된 모델을 통해 기존의 표준 데이터셋인 ICDAR 등으로부터 pseudo ground-truth를 얻어 추가 학습을 진행합니다. 
일단 word-level prediction으로 문자 영역을 crop한 후 학습한 모델을 사용하여 region score와 affinity score의 pseudo ground-truth를 생성합니다. 
그 후 watershed 알고리즘을 사용하여 문자를 분할하고 character box를 결정하여 줍니다. 
또한 pseudo GT가 학습에 부정적인 영향을 미치는 지 판단하기 위해 confidence score를 구하여 모델의 fine-tuning에 이용합니다. 
Fine-tuning 하는 동안에는 Synthtext 데이터와 ICDAR 데이터를 같이 이용하고 1:5의 비율로 batch를 구성하여 학습을 진행하였습니다. 
한글 정보가 없는 ICDAR15 데이터셋을 fine-tuning에 이용하게 되니 ICDAR15 데이터셋의 테스트셋에 대해서는 성능이 높게 나왔지만 한글이 포함된 eval_data에 대해서는 성능이 높지 않아 ICDAR 2017과 2019의 MLT에서 한글이 포함된 데이터셋을 가져와 한글 데이터셋을 구축하여 학습을 진행해 최종 score로 제출하였습니다.

과제를 진행하면서 진행해보고 싶었던 future works로는 fine-tuning data로 간판과 표지판 등이 다수 포함된 [Text-in-the-wild](http://101.101.175.217:8080/static/dataset/text_wild)라는 한국어 데이터셋과 AI-Hub에서 제공하는 손글씨 데이터셋을 사용해보는 것입니다. 
ICDAR 2017과 2019 MLT dataset에서는 한글이 포함된 이미지 수도 많지 않았고, eval_data에 존재하는 붓글씨나 간판과 같은 데이터와 비슷한 경우가 많아 성능 향상에 도움이 될 것이라고 생각합니다. 
또한, CRAFT의 첫번째 training을 위한 SynthText dataset을 한국어와 영어가 포함된 데이터로 직접 생성해서 사용하는 방법도 고려해보았습니다. 
마지막으로는, evaluation을 통해 이미지에서 실제로 bounding box가 생성된 결과물을 여러개 보았는데, affinity score를 예측해도 띄어쓰기 등에는 robust 하지 못한 경우가 많은 것 같아 역으로 recognition의 결과를 이용해서 단어의 글자 수 등을 고려할 수 있게 학습하는 등을 추가하면 text detector의 성능이 더 향상될 수 있지 않을까 생각해보았습니다.



# References
- Zhou, Xinyu, et al. "East: an efficient and accurate scene text detector." Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. 2017.

- Baek, Youngmin, et al. "Character region awareness for text detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

- Gupta, Ankush, Andrea Vedaldi, and Andrew Zisserman. "Synthetic data for text localisation in natural images." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

- Karatzas, Dimosthenis, et al. "ICDAR 2015 competition on robust reading." 2015 13th International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2015.

- https://github.com/clovaai/CRAFT-pytorch

- https://github.com/backtime92/CRAFT-Reimplementation

- https://aihub.or.kr/aidata/133