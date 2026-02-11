import torch
import torch.nn.functional as F
from torchvision import transforms
from model import ICM
import os
import sys
import argparse
from PIL import Image
import numpy as np
import cv2
from utils import *


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script (VBR variants).")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input images")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save images")
    parser.add_argument("--d", type=float, required=True, help="Parameter to control the rate-mAP tradeoff")

    args = parser.parse_args(argv)
    return args

def load_model(device, ckpt_path=None):
    icm = ICM().to(device).eval()
    if ckpt_path:
        print("Loading", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = {}
        for k, v in checkpoint["state_dict"].items():
            state[k.replace("module.", "")] = v
        icm.load_state_dict(state)
    return icm

def main(argv):
    print('\n::: compress images for Machines :::\n')
    print(torch.cuda.is_available())

    args = parse_args(argv)
    p = 128
    path = args.input
    out_path = args.save_path
    os.makedirs(out_path, exist_ok=True)
    img_list = []
    for file in os.listdir(path):
        img_list.append(file)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    icm = ICM(d=args.d)
    icm = icm.to(device)
    icm.eval()
    Bit_rate = 0
    
    dictory = {}
    if args.checkpoint:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        icm.load_state_dict(dictory)

    Bit_rate = 0.
    
    print('\n::real compression::\n')
    icm.update()
    for img_name in img_list:
        img_path = os.path.join(path, img_name)
        print(img_path[-16:])
        img = Image.open(img_path).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        x_padded, padding = pad(x, p)
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            out_enc = icm.compress(x_padded)
            out_dec = icm.decompress(out_enc["strings"], out_enc["shape"])
            
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
            
            Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            yi = out_dec["x_hat"].detach().cpu().numpy()
            yi = np.squeeze(yi[0,:,:,:])*255
            yi = yi.transpose(1,2,0)
            yi = cv2.cvtColor(yi, cv2.COLOR_RGB2BGR)
            img_PATH = img_path[-16:-4]
            cv2.imwrite(out_path + '/%s.png'%(img_PATH),yi.astype(np.uint8))

    print('\n---Result bpp---')
    print(f'Average_Bit-rate: {(Bit_rate/len(img_list)):.3f} bpp')
    print('--- Save image ---')
    print(f'Compressed images are saved in {out_path}' )

if __name__ == "__main__":
    main(sys.argv[1:])
