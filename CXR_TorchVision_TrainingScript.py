import sys
sys.path.append('D:\\GitHub\\Mariuki\\DiseaseDetector\\DiseaseDetector_TorchVision')

from typing import List, Dict
import pathlib
from multiprocessing import Pool
from skimage.io import imread

from tools import get_transform, get_instance_segmentation_model, ObjectDetectionDataSet
from tools import map_class_to_int, save_json, read_json, get_filenames_of_path

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image

import time
import datetime

import json

def main():

    # Directorio donde se enceuentran las imágenes y etiquetas para entrenamiento
    root = pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512')

    # Cargar las imágenes y las etiquetas
    inputs = get_filenames_of_path(root / 'ChestBBImages')
    targets = get_filenames_of_path(root / 'ChestBBLabels')

    # Ordenar entradas y objetivos
    inputs.sort()
    targets.sort()

    # Mapear las etiquetas con valores enteros
    mapping = {'Atelectasis': 1,
               'Cardiomegaly': 3,
               'Effusion': 4,
               'Infiltrate': 8,
               'Mass': 6,
               'Nodule': 7,
               'Pneumonia': 2,
               'Pneumothorax': 5}

    # Participación estratificada: misma cantidad de instancias respecto a sus etiquetas en cada subconjunto
    StratifiedPartition = read_json(pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/DatasetSplits/ChestXRay8/split1.json'))

    inputs_train = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Train'].keys())]
    targets_train = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Train'].keys())]

    inputs_valid = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Val'].keys())]
    targets_valid = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Val'].keys())]

    inputs_test = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBImages/' + i[:-4] + '.png') for i in list(StratifiedPartition['Test'].keys())]
    targets_test = [pathlib.Path('D:/GitHub/Mariuki/DiseaseDetector/data/ChestXRay8/512/ChestBBLabels/' + i[:-4] + '.json') for i in list(StratifiedPartition['Test'].keys())]

    lt = len(inputs_train)+len(inputs_valid)+len(inputs_test)
    ltr,ptr,lvd,pvd,lts,pts = len(inputs_train), len(inputs_train)/lt, len(inputs_valid), len(inputs_valid)/lt, len(inputs_test), len(inputs_test)/lt
    print('Total de datos: {}\nDatos entrenamiento: {} ({:.2f}%)\nDatos validación: {} ({:.2f}%)\nDatos Prueba: {} ({:.2f}%)'.format(lt,ltr,ptr,lvd,pvd,lts,pts))

    # dataset_train = ObjectDetectionDataSet(inputs=inputs_train,
    #                                        targets=targets_train,
    #                                        transform=None,
    #                                        add_dim = None,
    #                                        use_cache=True,
    #                                        convert_to_format=None,
    #                                        mapping=mapping,
    #                                        tgt_int64=False)
    inputs_train

    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



    # Logging metadata
    import neptune.new as neptune
    from neptune.new.types import File

    # Llave personal de usuario obtenida de Neptune.ai
    NEPTUNE_API_TOKEN = str(sys.argv[1]) #os.getenv("NEPTUNE")
    # Se puede copiar y poner directamente la llave. O configurar como variable de entorno
    run = neptune.init(project='rubsini/CXR-Fine-Tunning',
                       api_token=NEPTUNE_API_TOKEN)

    # run.stop()

    from engine import train_one_epoch#, evaluate
    from engineMod import evaluate
    import utils
    import transforms as T


    # use our dataset and defined transformations
    # dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    dataset = ObjectDetectionDataSet(inputs=inputs_train + inputs_valid,
                                           targets=targets_train + targets_valid,
                                           transform=get_transform(train=True),
                                           add_dim = None,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=False)
    print('Dataset Train Ready!')

    # dataset_valid = ObjectDetectionDataSet(inputs=inputs_valid,
    #                                        targets=targets_valid,
    #                                        transform=get_transform(train=True),
    #                                        add_dim = None,
    #                                        use_cache=True,
    #                                        convert_to_format=None,
    #                                        mapping=mapping,
    #                                        tgt_int64=False)

    dataset_test = ObjectDetectionDataSet(inputs=inputs_test,
                                           targets=targets_test,
                                           transform=get_transform(train=True),
                                           add_dim = None,
                                           use_cache=True,
                                           convert_to_format=None,
                                           mapping=mapping,
                                           tgt_int64=False)
    print('Dataset Test Ready!')

    # split the dataset in train and test set
    #torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has nine classes - background and 8 diseases types
    num_classes = 9

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    from torch.optim.lr_scheduler import StepLR
    num_epochs = 50

    for epoch in range(num_epochs):
       # train for one epoch, printing every 10 iterations
       npl = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
       # print("Loger: ",npl)
       run["logs/lr"].log(npl.lr.value)
       run["logs/loss"].log(npl.loss.global_avg)
       run["logs/loss_classifier"].log(npl.loss_classifier.global_avg)
       run["logs/loss_box_reg"].log(npl.loss_box_reg.global_avg)
       run["logs/loss_objectness"].log(npl.loss_objectness.global_avg)
       run["logs/loss_rpn_box_reg"].log(npl.loss_rpn_box_reg.global_avg)
       # update the learning rate
       lr_scheduler.step()
       # evaluate on the test dataset
       cc= evaluate(model, data_loader_test, device=device, epoch=epoch,run = run, torch_mets= [['macro','micro'],'global', True])
       run["logs/AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[0])
       run["logs/AP @[ IoU=0.50      | area=   all | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[1])
       run["logs/AP @[ IoU=0.75      | area=   all | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[2])
       run["logs/AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[3])
       run["logs/AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[4])
       run["logs/AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[5])
       run["logs/AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]"].log(cc.coco_eval['bbox'].stats[6])
       run["logs/AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]"].log(cc.coco_eval['bbox'].stats[7])
       run["logs/AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[8])
       run["logs/AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[9])
       run["logs/AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[10])
       run["logs/AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"].log(cc.coco_eval['bbox'].stats[11])
       # print("Coco Eval:", cc)

if __name__ == "__main__":
    main()
