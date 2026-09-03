import blobfile as bf
from mpi4py import MPI
import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pydicom
import random
import torch

dose_dict = {
    "D100": '2.886 x 600 WB D100',
    "D50" : '2.886 x 600 WB D50',
    "D20" : '2.886 x 600 WB D20',
    "D10" : '2.886 x 600 WB D10',
    "D4"  : '2.886 x 600 WB D4',
    "Normal" : '2.886 x 600 WB NORMAL',
}
# 类别与计量的对应
#   DRF100  1 
#   DRF50   2
#   DRF20   3
#   DRF10   4
#   DRF4    5

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, ifResize=False, dose="D100"
):
    # /home/uPET/dataset/training/PART2
    assert dose in {'D100', 'D50', 'D20', 'D10', 'D4', 'ALL', 'G10'}
    DRF0_data = sorted(glob(f"{data_dir}/*/{dose_dict['Normal']}/*"))
    if dose == "ALL":
        DRF100_data = sorted(glob(f"{data_dir}/*/{dose_dict['D100']}/*"))
        DRF50_data = sorted(glob(f"{data_dir}/*/{dose_dict['D50']}/*"))
        DRF20_data = sorted(glob(f"{data_dir}/*/{dose_dict['D20']}/*"))
        DRF10_data = sorted(glob(f"{data_dir}/*/{dose_dict['D10']}/*"))
        DRF4_data = sorted(glob(f"{data_dir}/*/{dose_dict['D4']}/*"))
        lq_data = []
        lq_data.extend(DRF100_data); lq_data.extend(DRF50_data); lq_data.extend(DRF20_data)
        lq_data.extend(DRF10_data); lq_data.extend(DRF4_data)
        hq_data = []
        hq_data.extend(DRF0_data); hq_data.extend(DRF0_data); hq_data.extend(DRF0_data);
        hq_data.extend(DRF0_data); hq_data.extend(DRF0_data);
        if class_cond:
            dose_label = []
            dose_label.extend([1 for i in range(len(DRF100_data))])
            dose_label.extend([2 for i in range(len(DRF50_data))])
            dose_label.extend([3 for i in range(len(DRF20_data))])
            dose_label.extend([4 for i in range(len(DRF10_data))])
            dose_label.extend([5 for i in range(len(DRF4_data))])
        assert len(lq_data) == len(hq_data)
        index_list = list(range(len(hq_data)))
        index_list = random.sample(index_list, len(hq_data) // 5)
        hq_data = [hq_data[index] for index in index_list]
        lq_data = [lq_data[index] for index in index_list]
        if class_cond:
            dose_label = [dose_label[index] for index in index_list]
    elif dose == "G10":
        lq_data = sorted(glob(f"{data_dir}/G10/*"))
        hq_data = sorted(glob(f"{data_dir}/G180/*"))
        dose_label = [2 for i in range(len(lq_data))]
    else:
        lq_data = sorted(glob(f"{data_dir}/*/{dose_dict[dose]}/*"))
        if class_cond:
            dose_label = []
            if dose == "D100":
                dose_label = [1 for i in range(len(lq_data))]
            elif dose == "D50":
                dose_label = [2 for i in range(len(lq_data))]
            if dose == "D20":
                dose_label = [3 for i in range(len(lq_data))]
            if dose == "D10":
                dose_label = [4 for i in range(len(lq_data))]
            if dose == "D4":
                dose_label = [5 for i in range(len(lq_data))]
        hq_data = DRF0_data
    assert len(hq_data) == len(lq_data) == len(dose_label) if class_cond else len(lq_data) == len(hq_data)

    dataset = ImageDataset(
        resolution=image_size,
        hq_image_paths=hq_data,
        lq_image_paths=lq_data,
        classes=dose_label,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        ifResize=ifResize,
    ) if class_cond else \
    ImageDataset(
        resolution=image_size,
        hq_image_paths=hq_data,
        lq_image_paths=lq_data,
        classes=None,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        ifResize=ifResize,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(self, resolution, hq_image_paths, lq_image_paths, classes=None, shard=0, num_shards=1, ifResize=False):
        super().__init__()
        self.resolution = resolution
        self.hq_images = hq_image_paths[shard:][::num_shards]
        self.classes = None if classes is None else classes[shard:][::num_shards]
        self.lq_images = lq_image_paths[shard:][::num_shards]

        self.dataProc = transforms.Compose([
                transforms.Lambda(lambda t : get_SUV(t)),
                transforms.Lambda(lambda t: np.pad(t, [(3, 3), (3, 3)], mode="constant", constant_values=0)), # 分辨率250x250,填充到256x256
                transforms.ToTensor(),
                # transforms.CenterCrop((resolution, resolution)), # 分辨率大于256x256,裁切到256x256
            ])

        if ifResize:
            self.dataProclq = transforms.Compose([
                transforms.Lambda(lambda t : get_SUV(t)),
                transforms.ToTensor(),
                transforms.CenterCrop((resolution, resolution)),
                transforms.Resize((resolution // 4, resolution // 4)),
                transforms.Resize((resolution, resolution)),
            ])
        else:
            self.dataProclq = transforms.Compose([
                transforms.Lambda(lambda t : get_SUV(t)),
                transforms.ToTensor(),
                transforms.CenterCrop((resolution, resolution)),
            ])
        self.num_classes = len(dose_dict) - 1

    def onehotEncoding(self, dose_tag):
        dose_class = torch.zeros((self.num_classes, ))
        dose_class[dose_tag-1] = 1
        return dose_class

    def __len__(self):
        return len(self.hq_images)
    
    def __getitem__(self, idx):
        hq_path = self.hq_images[idx]
        with bf.BlobFile(hq_path, 'rb') as f:
            hq_image = self.dataProc(f)
        hq_image = hq_image.to(torch.float32)

        lq_path = self.lq_images[idx]
        with bf.BlobFile(lq_path, 'rb') as f:
            lq_image = self.dataProclq(f)
        lq_image = lq_image.to(torch.float32)

        if self.classes is not None:
            classes_tag = self.classes[idx]
            onehot = self.onehotEncoding(classes_tag)
            dose_label = onehot.to(torch.float32)
            return hq_image, {"lq": lq_image, "dose": dose_label}

        else:
            return hq_image, {"lq": lq_image}


def get_SUV(path):
    ds = pydicom.dcmread(path)
    image_data = ds.pixel_array
    
    # 提取患者体重 (kg)
    patient_weight = ds.PatientWeight  # 例如: 70.0 kg
    
    # 提取注射的放射性剂量 (MBq)
    radiopharmaceutical_info = ds.RadiopharmaceuticalInformationSequence[0]
    injected_dose = radiopharmaceutical_info.RadionuclideTotalDose  # 例如: 50000000 Bq
    
    # 提取图像的放射性活度 (Bq/mL)
    # 这里假设 DICOM 文件中存储的是活度浓度
    activity_concentration = image_data  # 例如: 3.5 Bq/mL
    
    # 计算 SUV
    suv = (activity_concentration * patient_weight * 1000) / injected_dose
    
    return suv



# if __name__ == "__main__":
#     class_cond = True
#     data = load_data(data_dir="/home/uPET/dataset/training/PART2",
#               batch_size=2,
#               image_size=256,\
#               class_cond=class_cond,
#               deterministic=False,
#               dose="ALL"
#               )
#     if class_cond:
#         hq_image, lq_image, dose_label = next(data)
#     else:
#         hq_image, lq_image = next(data)
#     pass

