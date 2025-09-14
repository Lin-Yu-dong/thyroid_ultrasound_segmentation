import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from mobileTrans import *
from archs import UNext
from SETR.transformer_seg import SETRModel
from transunet.utils.transunet import TransUNet
from Swimunet import *
from MISSFormer.networks.MISSFormer import *
from Axial_trans.lib.models.axialnet import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/ICOS_MISSFormer_woDS/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    # model = UNext(num_classes=1, input_channels=3, deep_supervision=False,img_size=224)


    # model = MISSFormer(num_classes=1, token_mlp_mode="mix_skip", encoder_pretrained=True)
    # model = SETRModel(patch_size=(32, 32), 
    #             in_channels=3, 
    #             out_channels=1, 
    #             hidden_size=1024, 
    #             num_hidden_layers=8, 
    #             num_attention_heads=16, 
    #             decode_features=[512, 256, 128, 64])

    # model = TransUNet(img_dim=256,
    #                       in_channels=3,
    #                       out_channels=256,
    #                       head_num=4,
    #                       mlp_dim=512,
    #                       block_num=8,
    #                       patch_dim=16,
    #                       class_num=1)

    # model = SwinTransformer(hidden_dim=96,layers = (2, 2, 6, 2),heads = (3, 6, 12, 24), n_channels=3, n_classes=1,
    # head_dim = 32,
    # window_size = 7,
    # downscaling_factors = (4, 2, 2, 2),
    # relative_pos_embedding = True
# )


    # model = MedT(img_size = 256, imgchan = 3)

    model = MobileViT_S(img_size = 224, num_classes = 1) 
    
    # archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    val_img_ids = glob(os.path.join('/media/mostafa/Elements/ICOS/codes/Heatmaps/test/', 'images', '*'))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    # print(val_img_ids)

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/ICOS_MISSFormer_woDS/model.pth'))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('/media/mostafa/Elements/ICOS/codes/Heatmaps/test/','images'),
        mask_dir=os.path.join('/media/mostafa/Elements/ICOS/codes/Heatmaps/test/','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda() 
            model = model.cuda()
            # compute output
            output = model(input) # don't compute loss


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1  # bina
            output[output<0.5]=0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('/media/mostafa/Elements/ICOS/codes/RESULT/Miss/', meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
