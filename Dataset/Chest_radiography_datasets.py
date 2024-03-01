import numpy as np
import torch
import os
import torchvision
from PIL import Image
from torchvision import transforms as T
import pandas as pd
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
import h5py


class MIMIC_CXR_DATASET(Dataset):
    y_names = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    flag_subset = None
    def __init__(self, cfg_m, cfg_proj, flag_train, root_path = "/storage/yjpang/datasets", flag_preprocess = True):
        self.flag_preprocess = flag_preprocess
        self.flag_split = flag_train if isinstance(flag_train, str) else "train" if flag_train else "test"
        self.mimic_cxr_path, self.mimic_cxr_jpg_path, self.label_path, self.meta_path, self.split_path = self.init_path(root_path)
        self.file_path_preprocess = os.path.join(self.mimic_cxr_jpg_path, "mimic_jpg_preprocess_%s.h5"%(self.flag_split))
        self.data_list = self.find_data(flag_split = self.flag_split)
        if self.flag_preprocess: self.img_preprocess = self.imgs_to_hdf5(self.data_list, self.file_path_preprocess)
        self.classes = [["no %s"%(n), "%s"%(n)] for n in self.y_names]
        self.classes = [val for sublist in self.classes for val in sublist]
        self.transform = TransformsClinical(size = cfg_m.data.img_size)
        self.convert_subset(None)
        self.stats(cfg_m.diseases_focus)

    def init_path(self, root_path):
        mimic_cxr_path = os.path.join(root_path, "mimic-cxr")
        mimic_cxr_jpg_path = os.path.join(root_path, "mimic-cxr-jpg")
        label_path = os.path.join(mimic_cxr_jpg_path, "2.0.0", "mimic-cxr-2.0.0-chexpert.csv")
        meta_path = os.path.join(mimic_cxr_jpg_path, "2.0.0", "mimic-cxr-2.0.0-metadata.csv")
        split_path = os.path.join(mimic_cxr_jpg_path, "2.0.0", "mimic-cxr-2.0.0-split.csv")
        return mimic_cxr_path, mimic_cxr_jpg_path, label_path, meta_path, split_path

    def convert_subset(self, name):
        if name in self.y_names:
            id  = self.y_names.index(name)
            self.data_list_sub = [[img_path,imp_path,labels] for img_path,imp_path,labels in self.data_list if labels[id] == 0 or labels[id] == 1]
        else:
            self.data_list_sub = self.data_list

    def stats(self, d_focus):
        print("statistics of mimic_%s"%(self.flag_split))
        for [id_disease, name] in [[self.y_names.index(name), name] for name in d_focus]:
            self.convert_subset(name)
            num_p = sum([1 for _, _, labels in self.data_list_sub if labels[id_disease] == 1])
            print("%s - num of data points = %d (p-%d,n-%d)"%(name, len(self.data_list_sub), num_p, len(self.data_list_sub) - num_p))
        self.convert_subset(None)

    def __len__(self):
        return len(self.data_list_sub)

    def __getitem__(self, idx):
        img_path, imp_path, labels = self.data_list_sub[idx]
        img = self.img_preprocess[idx] if self.flag_preprocess else self.handle_img(img_path)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float() # torch, (3, 320, 320)
        img = self.transform(img) if self.transform is not None else img
        imp = self.handle_imp(imp_path)
        labels2 = [[1, 0] if i == 0 else [0, 1] if i == 1 else [-1, -1]  for i in labels]
        return img, imp, torch.tensor(labels), torch.tensor(labels2)

    def getIndexOfLast(self, l, element):
        """ Get index of last occurence of element"""
        return max(loc for loc, val in enumerate(l) if val == element)

    def preprocess(self, img, desired_size=320):
        old_size = img.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new('L', (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        return new_img

    def handle_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)
        img = np.expand_dims(img, axis=0)
        return img

    def handle_imp(self, imp_path):
        with open(imp_path, 'r') as f:
            s = f.read()
            s_split = s.split()
            if "IMPRESSION:" in s_split:
                begin = self.getIndexOfLast(s_split, "IMPRESSION:") + 1
                end = None
                end_cand1 = None
                end_cand2 = None
                # remove recommendation(s) and notification
                if "RECOMMENDATION(S):" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATION(S):")
                elif "RECOMMENDATION:" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATION:")
                elif "RECOMMENDATIONS:" in s_split:
                    end_cand1 = s_split.index("RECOMMENDATIONS:")
                if "NOTIFICATION:" in s_split:
                    end_cand2 = s_split.index("NOTIFICATION:")
                elif "NOTIFICATIONS:" in s_split:
                    end_cand2 = s_split.index("NOTIFICATIONS:")
                if end_cand1 and end_cand2:
                    end = min(end_cand1, end_cand2)
                elif end_cand1:
                    end = end_cand1
                elif end_cand2:
                    end = end_cand2            
                if end == None:
                    imp = " ".join(s_split[begin:])
                else:
                    imp = " ".join(s_split[begin:end])
            else:
                imp = 'NO IMPRESSION'
        return imp

    def find_data(self, flag_split):
        imp_path_func = lambda path, sbj_id, study_id : "%s/2.0.0/files/p%s/p%s/s%s.txt"%(path, sbj_id[:2], sbj_id, study_id)
        img_path_func = lambda path, sbj_id, study_id, img_id: "%s/2.0.0/files/p%s/p%s/s%s/%s.jpg"%(path, sbj_id[:2], sbj_id, study_id, img_id)

        #get the split info
        assert flag_split in ["train", "test", "val"]
        flag_split = flag_split if flag_split != "val" else "validate"
        split_df = pd.read_csv(self.split_path)
        split_index = split_df.index[split_df["split"] == flag_split]

        #get the meta data
        meta_df = pd.read_csv(self.meta_path)
        meta_df = meta_df.iloc[split_index]#split the data
        #---filter the data---
        # meta_df = meta_df[meta_df['subject_id'].astype(str).str.contains("^1[0-9].*")] #sub-folder P10
        meta_df = meta_df[meta_df['ViewPosition'].str.contains("PA", na=False)]  #front view
        #---filter the data---
        img_id = meta_df["dicom_id"].astype(str).values.tolist()
        sbj_id = meta_df["subject_id"].astype(str).values.tolist()
        study_id = meta_df["study_id"].astype(str).values.tolist()

        #construct the label dic
        label_df = pd.read_csv(self.label_path)
        label_df = label_df.fillna(-1).copy()
        study_id_unique = label_df["study_id"].astype(str).values.tolist()
        Y = (label_df.loc[:, self.y_names]).to_numpy()
        label_dic = {a:b for a,b in zip(study_id_unique, Y)}

        #create the final data list
        data_list = [[img_path_func(self.mimic_cxr_jpg_path, a,b,c),     #img path
                            imp_path_func(self.mimic_cxr_path, a, b),    #impression path
                            label_dic[b]                            #the label according to the study_id
                            ]     
                            for a,b,c in zip(sbj_id, study_id, img_id) if b in label_dic]
        return data_list

    def imgs_to_hdf5(self, data_list, file_path_preprocess):
        if self.flag_preprocess:
            if not os.path.isfile(file_path_preprocess):
                with h5py.File(file_path_preprocess,'w') as h5f:
                    num_imgs = len(data_list)
                    img_shape = list(self.handle_img(data_list[0][0]).shape)
                    img_dset = h5f.create_dataset('cxr', shape= [num_imgs] + img_shape)    
                    for idx, (img_path, _, _) in enumerate(tqdm(data_list)):
                        img = self.handle_img(img_path)
                        img_dset[idx] = img
                print("img preprocess - %d images are added to %s"%(num_imgs, file_path_preprocess))
            return h5py.File(file_path_preprocess, 'r')['cxr']
        else:
            return None


class CHEXPERT_DATASET(Dataset):
    y_names = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                        'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
    def __init__(self, cfg_m, cfg_proj, flag_train, root_path = "/storage/yjpang/datasets", flag_preprocess = True):
        self.flag_preprocess = flag_preprocess
        self.flag_split = flag_train if isinstance(flag_train, str) else "train" if flag_train else "test"
        self.chexpert_path, self.path_dic = self.init_path(root_path)
        self.file_path_preprocess = os.path.join(self.chexpert_path, "CheXpert_preprocess_%s.h5"%(self.flag_split))
        self.data_list = self.find_data(flag_split = self.flag_split)
        if self.flag_preprocess: self.img_preprocess = self.imgs_to_hdf5(self.data_list, self.file_path_preprocess)
        self.classes = [["no %s"%(n), "%s"%(n)] for n in self.y_names]
        self.classes = [val for sublist in self.classes for val in sublist]
        self.transform = TransformsClinical(size = cfg_m.data.img_size)
        self.convert_subset(None)
        self.stats(cfg_m.diseases_focus)

    def init_path(self, root_path):
        chexpert_path = os.path.join(root_path, "CheXpert")
        test_path = os.path.join(chexpert_path, "test_labels.csv")
        val_path = os.path.join(chexpert_path, "val_labels.csv")
        train_path = os.path.join(chexpert_path, "val_labels.csv")    #no training data
        return chexpert_path, {"train":train_path, "val":val_path, "test":test_path}

    def convert_subset(self, name):
        if name in self.y_names:
            id  = self.y_names.index(name)
            self.data_list_sub = [[img_path, -1, labels] for img_path, _, labels in self.data_list if labels[id] == 0 or labels[id] == 1]
        else:
            self.data_list_sub = self.data_list

    def stats(self, d_focus):
        print("statistics of CheXpert_%s"%(self.flag_split))
        for [id_disease, name] in [[self.y_names.index(name), name] for name in d_focus]:
            self.convert_subset(name)
            num_p = sum([1 for _, _, labels in self.data_list_sub if labels[id_disease] == 1])
            print("%s - num of data points = %d (p-%d,n-%d)"%(name, len(self.data_list_sub), num_p, len(self.data_list_sub) - num_p))
        self.convert_subset(None)

    def __len__(self):
        return len(self.data_list_sub)

    def __getitem__(self, idx):
        img_path, _, labels = self.data_list_sub[idx]
        img = self.img_preprocess[idx] if self.flag_preprocess else self.handle_img(img_path)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float() # torch, (3, 320, 320)
        img = self.transform(img) if self.transform is not None else img
        labels2 = [[1, 0] if i == 0 else [0, 1] if i == 1 else [-1, -1]  for i in labels]
        return img, -1, torch.tensor(labels), torch.tensor(labels2)

    def preprocess(self, img, desired_size = 320):
        old_size = img.size
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new('L', (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        return new_img

    def handle_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)
        img = np.expand_dims(img, axis=0)
        return img

    def find_data(self, flag_split, filter = "view1"):
        file_path = self.path_dic[flag_split]  #train_path == test temp
        y_df = pd.read_csv(file_path)
        y_df = y_df[y_df['Path'].str.contains(filter)]
        X = y_df["Path"].tolist()
        X = X if flag_split != "val" else ["val"+i[19:] for i in X]
        X = [os.path.join(self.chexpert_path, a) for a in X]
        y_df = y_df.loc[:, self.y_names]
        Y = y_df.to_numpy()
        data_list = [[a, -1, b] for a,b in zip(X, Y)]
        return data_list

    def imgs_to_hdf5(self, data_list, file_path_preprocess):
        if self.flag_preprocess:
            if not os.path.isfile(file_path_preprocess):
                with h5py.File(file_path_preprocess,'w') as h5f:
                    num_imgs = len(data_list)
                    img_shape = list(self.handle_img(data_list[0][0]).shape)
                    img_dset = h5f.create_dataset('cxr', shape= [num_imgs] + img_shape)    
                    for idx, (img_path, _, _) in enumerate(tqdm(data_list)):
                        img = self.handle_img(img_path)
                        img_dset[idx] = img
                print("img preprocess - %d images are added to %s"%(num_imgs, file_path_preprocess))
            return h5py.File(file_path_preprocess, 'r')['cxr']
        else:
            return None


class TransformsClinical:
    def __init__(self, size):
        self.test_transform = T.Compose([
            # T.ToTensor(),
            T.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            T.Resize(size),
        ])
    def __call__(self, x):
        return self.test_transform(x)