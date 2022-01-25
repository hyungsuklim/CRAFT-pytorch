#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import json
from PIL import Image
import shutil

# In[2]:


dataset_list = ["ICDAR2015","ICDAR2017","ICDAR2019"]
setting = ["training","val","test"]
train = {}
val = {}
val["images"] = {}
train["images"] = {}
license_tag = { "usability" : True,
                "public" : True,
                "commercial" : True,
                "type" : "CC-BY-SA",
                "holder" : None }
data_len = 0

# In[ ]:


"""
# Training data for ICDAR 2013
dataset_path = "/data/ICDAR/" 
path = dataset_path + dataset_list[0]
gt_folder = os.path.abspath(os.path.join(path,"Challenge2_Training_Task1_GT"))
img_folder = os.path.abspath(os.path.join(path,"Challenge2_Training_Task12_Images"))
gt_list = os.listdir(gt_folder)
data_len = 0
for file in sorted(gt_list) :
    num = (file.split(".")[0]).split("_")[1]
    src_img_file = "img_" + num + ".jpg"
    src_img_file2 = num + ".jpg"
    det_img_file = "img_" + str(data_len) + ".jpg"
    data_len += 1
    with open(img_folder+"/"+src_img_file2,'rb') as ip : 
        im = Image.open(ip)
        img_w,img_h = im.size
    shutil.copyfile(img_folder+"/"+src_img_file2,"/data/ICDAR/train_data/ch4_training_images/"+det_img_file)
    train["images"][det_img_file] = {"img_h":img_h,"img_w":img_w,"words":{},"tags":None,"license_tag":license_tag}    
    words = {}
    with open(gt_folder+"/"+file,'r',encoding="utf-8-sig") as f :
        for index,line in enumerate(f.readlines()) :
            points = {}
            xmin,ymin,xmax,ymax,transcription = line.split(" ")
            points["points"] = [[int(xmin),int(ymin)],[int(xmax),int(ymin)],
            [int(xmin),int(ymax)],[int(xmax),int(ymax)]]
            words = points
            words["transcription"] = transcription.replace("\"","")[:-1]
            words["language"] = ["en"]
            words["illegibility"] = False
            words["orientation"] = "Horizontal"
            words["word_tags"] = None
            train["images"][det_img_file]["words"][index] = words
print(data_len)
"""

# In[3]:


# Training data for ICDAR 2015
dataset_path = "/data/ICDAR/" 
path = dataset_path + dataset_list[0]
gt_folder = os.path.abspath(os.path.join(path,"ch4_training_localization_transcription_gt"))
img_folder = os.path.abspath(os.path.join(path,"ch4_training_images"))
gt_list = os.listdir(gt_folder)
save_folder = "/data/ICDAR/train_data/ch4_training_images"
save_gt_folder = "/data/ICDAR/train_data/ch4_training_localization_transcription_gt"
for file in sorted(gt_list) :
    num = (file.split(".")[0]).split("_")[2]
    src_img_file = "img_" + num + ".jpg"
    det_img_file = "img_" + str(data_len) + ".jpg"
    data_len += 1
    with open(img_folder+"/"+src_img_file,'rb') as ip : 
        im = Image.open(ip)
        img_w,img_h = im.size
        
    if not os.path.isdir(save_folder) :
        os.mkdir(save_folder)
    if not os.path.isdir(save_gt_folder) :
        os.mkdir(save_gt_folder)
    shutil.copyfile(os.path.join(img_folder,src_img_file),os.path.join(save_folder,det_img_file))
    train["images"][det_img_file] = {"img_h":img_h,"img_w":img_w,"words":{},"tags":None,"license_tag":license_tag}    
    words = {}
    with open(gt_folder+"/"+file,'r',encoding="utf-8-sig") as f :
        for index,line in enumerate(f.readlines()) :
            points = {}
            line_list = line.split(",")
            if len(line_list) > 9 :
                transcription = ""
                x1,y1,x2,y2,x3,y3,x4,y4,transcription = line_list[0:9]
                for i in range(9,len(line_list)) :
                    transcription += ("," + str(line_list[i]))
            else : 
                x1,y1,x2,y2,x3,y3,x4,y4,transcription = line.split(",")
            points["points"] = [[int(x1),int(y1)],[int(x2),int(y2)],
            [int(x3),int(y3)],[int(x4),int(y4)]]
            words = points
            words["transcription"] = transcription.replace("\"","")[:-1]
            words["language"] = ["en"]
            words["illegibility"] = False
            words["orientation"] = "Horizontal"
            words["word_tags"] = None
            train["images"][det_img_file]["words"][index] = words


# In[4]:


# Training data for ICDAR 2017
dataset_path = "/data/ICDAR/" 
path = dataset_path + dataset_list[1]
gt_folder = os.path.abspath(os.path.join(path,"ch8_training_localization_transcription_gt_v2"))
img_folder = os.path.abspath(os.path.join(path,"ch8_training_images")) # integrate all image files(1~8)
gt_list = os.listdir(gt_folder)
for file in sorted(gt_list) :
    num = (file.split(".")[0]).split("_")[2]
    src_img_file = "img_" + num + ".jpg"
    det_img_file = "img_" + str(data_len) + ".jpg"
    data_len += 1
    if not os.path.isfile(img_folder+"/"+src_img_file) :
        if not os.path.isfile(img_folder+"/"+src_img_file.replace("jpg","png")) :
            src_img_file = src_img_file.replace("jpg","gif")
            det_img_file = det_img_file.replace("jpg","gif")
        else : 
            src_img_file = src_img_file.replace("jpg","png")
            det_img_file = det_img_file.replace("jpg","png")
    with open(img_folder+"/"+src_img_file,'rb') as ip : 
        im = Image.open(ip)
        img_w,img_h = im.size
    shutil.copyfile(os.path.join(img_folder,src_img_file),os.path.join(save_folder,det_img_file))
    train["images"][det_img_file] = {"img_h":img_h,"img_w":img_w,"words":{},"tags":None,"license_tag":license_tag}    
    words = {}
    with open(gt_folder+"/"+file,'r',encoding="utf-8-sig") as f :
        for index,line in enumerate(f.readlines()) :
            points = {}
            line_list = line.split(",")
            if len(line_list) > 10 :
                transcription = ""
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list[0:10]
                for i in range(10,len(line_list)) :
                    transcription += ("," + str(line_list[i]))
            else : 
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list
            points["points"] = [[int(x1),int(y1)],[int(x2),int(y2)],
            [int(x3),int(y3)],[int(x4),int(y4)]]
            words = points
            words["transcription"] = transcription.replace("\"","")[:-1]
            words["language"] = [lang]
            words["illegibility"] = False
            words["orientation"] = "Horizontal"
            words["word_tags"] = None
            train["images"][det_img_file]["words"][index] = words


# In[5]:


# Training data for ICDAR 2019
dataset_path = "/data/ICDAR/" 
path = dataset_path + dataset_list[2]
gt_folder = os.path.abspath(os.path.join(path,"train_gt_t13"))
img_folder = os.path.abspath(os.path.join(path,"ImagesPart")) # integrate ImagePart1 and ImagePart2
gt_list = os.listdir(gt_folder)
for file in sorted(gt_list) :
    num = (file.split(".")[0]).split("_")[2]
    src_img_file = "tr_img_" + num + ".jpg"
    det_img_file = "img_" + str(data_len) + ".jpg"
    data_len += 1
    if not os.path.isfile(img_folder+"/"+src_img_file) :
        if not os.path.isfile(img_folder+"/"+src_img_file.replace("jpg","png")) :
            src_img_file = src_img_file.replace("jpg","gif")
            det_img_file = det_img_file.replace("jpg","gif")
        else : 
            src_img_file = src_img_file.replace("jpg","png")
            det_img_file = det_img_file.replace("jpg","png")
    with open(img_folder+"/"+src_img_file,'rb') as ip : 
        im = Image.open(ip)
        img_w,img_h = im.size
    shutil.copyfile(os.path.join(img_folder,src_img_file),os.path.join(save_folder,det_img_file))
    train["images"][det_img_file] = {"img_h":img_h,"img_w":img_w,"words":{},"tags":None,"license_tag":license_tag}    
    words = {}
    with open(gt_folder+"/"+file,'r',encoding="utf-8-sig") as f :
        for index,line in enumerate(f.readlines()) :
            points = {}
            line_list = line.split(",")
            if len(line_list) > 10 :
                transcription = ""
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list[0:10]
                for i in range(10,len(line_list)) :
                    transcription += ("," + str(line_list[i]))
            else : 
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list
            points["points"] = [[int(x1),int(y1)],[int(x2),int(y2)],
            [int(x3),int(y3)],[int(x4),int(y4)]]
            words = points
            words["transcription"] = transcription.replace("\"","")[:-1]
            words["language"] = [lang]
            words["illegibility"] = False
            words["orientation"] = "Horizontal"
            words["word_tags"] = None
            train["images"][det_img_file]["words"][index] = words


# In[6]:


with open("/data/ICDAR/train_data/ufo/train.json","w") as f :
    json.dump(train,f)

# In[23]:


# make data that contain ICDAR 2015 2017 2019 
det_path = "/data/ICDAR/train_data/ch4_training_localization_transcription_gt"
det_img_path = "/data/ICDAR/train_data/ch4_training_images"
from tqdm import tqdm

for index,file in tqdm(enumerate(train["images"])) :
    gt_file = "gt_" + file.split(".")[0] + ".txt"
    with open(os.path.join(det_path,gt_file), 'w',encoding='utf-8-sig') as outfile :
        for keys,value in train["images"][file]["words"].items() :
            points = value["points"]
            x1,y1 = points[0];x2,y2 = points[1];x3,y3 = points[2];x4,y4 = points[3];trans = value["transcription"]
            outfile.write("%d,%d,%d,%d,%d,%d,%d,%d,%s\n"%(x1,y1,x2,y2,x3,y3,x4,y4,trans))  
        

# In[26]:


# make data that contain korean data
ko_det_path = "/data/ICDAR/ko_train_data/ch4_training_localization_transcription_gt"
ko_det_img_path = "/data/ICDAR/ko_train_data/ch4_training_images"

for index,file in tqdm(enumerate(train["images"])) :
    gt_file = "gt_" + file.split(".")[0] + ".txt"
    k_flag = 0
    with open(os.path.join(ko_det_path,gt_file), 'w',encoding='utf-8-sig') as outfile :
        for keys,value in train["images"][file]["words"].items() :
            if value["language"][0] == "Korean" :
                    k_flag = 1
        if k_flag == 1 :
            for keys,value in train["images"][file]["words"].items() :
                points = value["points"]
                x1,y1 = points[0];x2,y2 = points[1];x3,y3 = points[2];x4,y4 = points[3];trans = value["transcription"]
                outfile.write("%d,%d,%d,%d,%d,%d,%d,%d,%s\n"%(x1,y1,x2,y2,x3,y3,x4,y4,trans)) 

                    
    if k_flag == 1 :
        shutil.copyfile(os.path.join(det_img_path,file),os.path.join(ko_det_img_path,file))
    else :
        os.remove(os.path.join(ko_det_path,gt_file))

# In[13]:


"""dataset_list = ["ICDAR2015","ICDAR2017","ICDAR2019"]


data = {}
data["ICDAR2015_img"] = ["ch4_training_images"]
data["ICDAR2015_gt"] = "ch4_training_localization_transcription_gt"
data["ICDAR2017_img"] = ["ch8_training_images_{}".format(i) for i in range(1,9)]
data["ICDAR2017_gt"] = "ch8_training_localization_transcription_gt_v2"
data["ICDAR2019_img"] = ["ImagesPart1","ImagesPart2"]
data["ICDAR2019_gt"] = "train_gt_t13"
"""

# In[ ]:


""" import os 
import json
from PIL import Image
import shutil
dataset_list = ["ICDAR2015","ICDAR2017","ICDAR2019"]


# Training data for ICDAR 2015,ICDAR 2017, ICDAR 2019
dataset_path = "/data/ICDAR/" 
img_save_path = "/data/ICDAR/ICDAR_151719/ch4_training_images"
gt_save_path = "/data/ICDAR/ICDAR_151719/ch4_training_localization_transcription_gt"
img_num = 0 
for index,dataset_name in enumerate(dataset_list) :
    img_folders = data[dataset_name + "_img"]
    gt_folder = data[dataset_name + "_gt"]
    for img_folder in img_folders : 
        f = os.path.join(dataset_path+dataset_name,img_folder)
        img_list = os.listdir(f)
        for img in sorted(img_list) : 
            img_name = img.split(".")[0]
            gt_file_name = "gt"+img_name+"txt"
            if index == 0 : #for ICDAR 2015
                shutil.copyfile(os.path.join(f,img),os.path.join(img_save_path,img))
                shutil.copyfile(os.path.join(f,gt_file_name),os.path.join(gt_save_path, gt_file_name)
            elif index == 1 :
                img_file_name = "img" + str(img_num) + img.split(".")[1]
                gt_file_name = "gt_img" + str(img_num) + "txt"
                shutil.copyfile(os.path.join(f,img),os.path.join(img_save_path,img_file_name))
                with open(os.path.join(f,gt_file_name), 'r') as infile, open(os.path.join(gt_save_path,gt_file_name, 'w') as outfile:
                    lines = infile.readlines()
                    for line in lines:
                        d = line.split(',')[:8]
                        tmp = [x for x in line.split(',')[:8]]
                        tmp.append(
            img_num += 1
"""

# In[ ]:


"""
# validation data for ICDAR 2017
data_len = 0
dataset_path = "/data/ICDAR/" 
path = dataset_path + dataset_list[2]
gt_folder = os.path.abspath(os.path.join(path,"ch8_validation_localization_transcription_gt_v2"))
img_folder = os.path.abspath(os.path.join(path,"ch8_validation_images")) # integrate ImagePart1 and ImagePart2
gt_list = os.listdir(gt_folder)
for file in sorted(gt_list) :
    num = (file.split(".")[0]).split("_")[2]
    src_img_file = "img_" + num + ".jpg"
    det_img_file = "img_" + str(data_len) + ".jpg"
    data_len += 1
    if not os.path.isfile(img_folder+"/"+src_img_file) :
        if not os.path.isfile(img_folder+"/"+src_img_file.replace("jpg","png")) :
            src_img_file = src_img_file.replace("jpg","gif")
            det_img_file = det_img_file.replace("jpg","gif")
        else : 
            src_img_file = src_img_file.replace("jpg","png")
            det_img_file = det_img_file.replace("jpg","png")
    shutil.copyfile(img_folder+"/"+src_img_file,"/data/ICDAR/val_data/images/"+det_img_file)

    with open(img_folder+"/"+src_img_file,'rb') as ip : 
        im = Image.open(ip)
        img_w,img_h = im.size
    val["images"][det_img_file] = {"img_h":img_h,"img_w":img_w,"words":{},"tags":None,"license_tag":license_tag}    
    words = {}
    with open(gt_folder+"/"+file,'r',encoding="utf-8-sig") as f :
        for index,line in enumerate(f.readlines()) :
            points = {}
            line_list = line.split(",")
            if len(line_list) > 10 :
                transcription = ""
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list[0:10]
                for i in range(10,len(line_list)) :
                    transcription += ("," + str(line_list[i]))
            else : 
                x1,y1,x2,y2,x3,y3,x4,y4,lang,transcription = line_list
            points["points"] = [[int(x1),int(y1)],[int(x2),int(y2)],
            [int(x3),int(y3)],[int(x4),int(y4)]]
            words = points
            words["transcription"] = transcription.replace("\"","")[:-1]
            words["language"] = [lang]
            words["illegibility"] = False
            words["orientation"] = "Horizontal"
            words["word_tags"] = None
            val["images"][det_img_file]["words"][index] = words
print(data_len)
with open("val.json","w") as f :
    json.dump(val,f)
    """

# 

# In[ ]:



