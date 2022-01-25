import scipy.io as scio
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import re
import itertools
from PIL import Image
from data import imgproc


from gaussianMap.gaussian import GaussianTransformer
from data.boxEnlarge import enlargebox
from data.imgaug import random_scale, random_crop
from data.load_icdar import load_icdar2013_gt, load_icdar2015_gt
from mep import mep
from watershed import watershed


class SynthTextDataLoader(data.Dataset):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, vis=False):
        assert 'synthtext' in data_dir_list.keys()

        self.target_size = target_size
        self.data_dir_list = data_dir_list
        self.vis = vis

        self.charbox, self.image, self.imgtxt = self.load_synthtext()
        self.gen = GaussianTransformer(200, 1.5)
        # self.gen.gen_circle_mask()
        
    def load_synthtext(self):
        gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]
        return charbox, image, imgtxt

    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        image = random_scale(image, _charbox, self.target_size)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, img_path
    
    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences, img_path = self.load_synthtext_image_gt(index)

        region_scores = self.gen.generate_region(image.shape, character_bboxes)
        affinities_scores, _ = self.gen.generate_affinity(image.shape, character_bboxes, words)

        random_transforms = [image, region_scores, affinities_scores, confidence_mask]
        # randomcrop = eastrandomcropdata((768,768))
        # region_image, affinity_image, character_bboxes = randomcrop(region_image, affinity_image, character_bboxes)

        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        image, region_image, affinity_image, confidence_mask = random_transforms
        image = Image.fromarray(image)
        image = image.convert('RGB')
        # image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = image.transpose(2, 0, 1)

        #resize label
        region_image = self.resizeGt(region_image)
        affinity_image = self.resizeGt(affinity_image)
        confidence_mask = self.resizeGt(confidence_mask)

        region_image = region_image.astype(np.float32) / 255
        affinity_image = affinity_image.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32)
        return image, region_image, affinity_image, confidence_mask, confidences

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.pull_item(index)
    
    
    
class ICDAR15DataLoader(data.Dataset):
    def __init__(self, net, icdar2015_folder, target_size=768, viz=False, debug=False):
        self.net = net
        self.net.eval()
        self.img_folder = os.path.join(icdar2015_folder, 'ch4_training_images')
        self.gt_folder = os.path.join(icdar2015_folder, 'ch4_training_localization_transcription_gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            area, p0, p3, p2, p1, _, _ = mep(box)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
            words.append(word)
        return bboxes, words

    def pull_item(self, index):
        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)

        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = random_scale(image, word_bboxes, self.target_size)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        character_bboxes = []
        new_words = []
        confidences = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    continue
                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i],
                                                                                               viz=self.viz)
                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)
        return image, character_bboxes, new_words, confidence_mask, confidences

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        return self.pull_item(index)
    
    def get_imagename(self, index):
        return self.images_path[index]
    
    

if __name__ == "__main__":
    data_dir_list = {"synthtext":"/gallery_moma/minjung.kim/dataset/SynthText"}
    craft_data = SynthTextDataLoader(768, data_dir_list)
    for index in range(10000):
        image, character_bboxes, words, confidence_mask, confidences, img_path = craft_data.load_synthtext_image_gt(index)
        gaussian_map = np.zeros(image.shape, dtype=np.uint8)
        gen = GaussianTransformer(200, 1.5)
        gen.gen_circle_mask()

        region_image = gen.generate_region(image.shape, character_bboxes)
        affinity_image, affinities = gen.generate_affinity(image.shape, character_bboxes, words)

        random_transforms = [image, region_image, affinity_image, confidence_mask]


        random_transforms = random_crop(random_transforms, (768, 768), character_bboxes)
        image, region_image, affinity_image, confidence_mask = random_transforms
        region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
        affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)

        region_image = cv2.addWeighted(region_image, 0.3, image, 1.0, 0)
        affinity_image = cv2.addWeighted(affinity_image, 0.3, image, 1.0, 0)

        for boxes in character_bboxes:
            for box in boxes:
                enlarge = enlargebox(box.astype(np.int), image.shape[0], image.shape[1])
                print("enlarge:", enlarge)
        stack_image = np.hstack((region_image, affinity_image))
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", stack_image)
        cv2.waitKey(0)
