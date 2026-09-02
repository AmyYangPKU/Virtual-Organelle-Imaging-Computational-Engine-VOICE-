import argparse
import os
import random

import numpy as np
import tifffile as tf
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import kornia.losses as losses
from pytorch_msssim import ms_ssim, MS_SSIM
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr

from RCAN2D import RCAN2D


class PairedTransform:
    def __init__(self, hflip=True, vflip=True, rot90=True):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90

    def __call__(self, lq, gt):
        if self.hflip and random.random() < 0.5:
            lq = torch.flip(lq, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if self.vflip and random.random() < 0.5:
            lq = torch.flip(lq, dims=[1])
            gt = torch.flip(gt, dims=[1])
        if self.rot90:
            k = random.choice([0, 1, 2, 3])
            lq = torch.rot90(lq, k, [1, 2])
            gt = torch.rot90(gt, k, [1, 2])
        return lq, gt


class RandomRotate90:
    def __call__(self, x):
        k = random.choice([0, 1, 2, 3])
        return torch.rot90(x, k, [1, 2])


class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        self.lq_files = sorted([
            f for f in os.listdir(lq_dir)
            if f.lower().endswith(('.tif', '.tiff'))
            and '._' not in f
        ])

        self.gt_files = sorted([
            f for f in os.listdir(gt_dir)
            if f.lower().endswith(('.tif', '.tiff'))
            and '._' not in f
        ])

        assert len(self.lq_files) == len(self.gt_files),\
            f"LQ/GT file count mismatch: {len(self.lq_files)} vs {len(self.gt_files)}"

        self.transform = PairedTransform()

    def __len__(self):
        return len(self.lq_files)

    def __getitem__(self, idx):
        lq_path = os.path.join(self.lq_dir, self.lq_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])

        lq_img = tf.imread(lq_path)
        gt_img = tf.imread(gt_path)


        lq_img = torch.from_numpy(lq_img)
        gt_img = torch.from_numpy(gt_img)

        lq_img, gt_img = lq_img.unsqueeze(0).float(), gt_img.unsqueeze(0).float()
        lq_img, gt_img = self.transform(lq_img, gt_img)

        return lq_img, gt_img


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        loss = mse_loss


        return loss


def _batch_minmax_norm(x: torch.Tensor):

    x_min = x.amin(dim=[1, 2, 3], keepdim=True)
    x_max = x.amax(dim=[1, 2, 3], keepdim=True)
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    return x_norm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_ms_ssim = 0.0


    progress_bar = tqdm(dataloader, desc="Training", ncols=80)

    for lq, gt in progress_bar:
        lq, gt = lq.to(device), gt.to(device)


        optimizer.zero_grad()
        output = model(lq)

        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


        progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})


        with torch.no_grad():
            out_norm = _batch_minmax_norm(output)
            gt_norm = _batch_minmax_norm(gt)
            ms_val = ms_ssim(
                out_norm,
                gt_norm,
                data_range=1.0,
                size_average=True
            )
            total_ms_ssim += ms_val.item()

    avg_loss = total_loss / len(dataloader)
    avg_ms_ssim = total_ms_ssim / len(dataloader)
    return avg_loss, avg_ms_ssim


def validate_val(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_ms_ssim = 0.0
    num_images = 0
    total_psnr = 0.0

    progress_bar = tqdm(dataloader, desc="Validating", ncols=80)

    with torch.no_grad():
        for lq, gt in progress_bar:

            lq, gt = lq.to(device), gt.to(device)


            output = model(lq)


            B = output.size(0)


            for b in range(B):
                out_b = output[b:b+1]
                gt_b  = gt[b:b+1]
                out_b[out_b < 0] = 0

 


                loss_b = criterion(out_b, gt_b)
                total_loss += loss_b.item()


                out_n = (out_b - out_b.min()) / (out_b.max() - out_b.min())
                gt_n  = (gt_b  - gt_b.min())  / (gt_b.max()  - gt_b.min())


                ms_val = ms_ssim(out_n, gt_n, data_range=1, size_average=True)

                out_np = out_n.squeeze().cpu().numpy()
                gt_np = gt_n.squeeze().cpu().numpy()
                psnr_val = psnr(gt_np, out_np, data_range=1.0)
                total_psnr += psnr_val


                total_ms_ssim += ms_val.item()

                num_images += 1

            progress_bar.set_postfix({
                "val_loss": f"{total_loss / num_images:.6f}",
                "val_psnr": f"{total_psnr / num_images:.2f}",
                "val_ms_ssim": f"{total_ms_ssim / num_images:.4f}"
            })

    avg_loss = total_loss / num_images
    avg_psnr = total_psnr / num_images
    avg_ms_ssim = total_ms_ssim / num_images

    return avg_loss, avg_psnr, avg_ms_ssim

def infer_and_save_test_sr(model, test_loader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        idx = 0
        for lq, _ in tqdm(test_loader, desc="Infer TEST (best SSIM)"):
            lq = lq.to(device)

            sr = model(lq)


            sr = sr.squeeze(0).squeeze(0)
            sr = sr - sr.min()

            sr = sr / (sr.max() + 1e-8)

            sr_np = (sr.cpu().numpy() * 65535).astype(np.uint16)

            save_path = os.path.join(save_dir, f"{idx+1}.tif")
            tf.imwrite(save_path, sr_np)

            idx += 1

    print(f">>> Saved {idx} SR test images to {save_dir}")


def main():


    parser = argparse.ArgumentParser(description="Train RCAN")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset root, containing train/LR, train/SR, val/LR, val/SR")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="directory to save checkpoints and logs")
    parser.add_argument("--pretrained", type=str, default=None,
                    help="path to pretrained RCAN checkpoint (.pth)")
    parser.add_argument("--finetune", action="store_true",
                    help="whether this run is finetuning")

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="number of training epochs (default: 20)"
    )


    args = parser.parse_args()
    dataset = args.dataset
    save_dir = args.save_dir


    os.makedirs(save_dir, exist_ok=True)

    best_ssim_test_sr_dir = os.path.join(save_dir, "best_ssim_test_SR")
    os.makedirs(best_ssim_test_sr_dir, exist_ok=True)


    train = os.path.join(dataset, "train")
    val   = os.path.join(dataset, "val")
    test  = os.path.join(dataset, "test")

    lq_train = os.path.join(train, "LR")
    gt_train = os.path.join(train, "SR")
    lq_val   = os.path.join(val,   "LR")
    gt_val   = os.path.join(val,   "SR")
    lq_test  = os.path.join(test,  "LR")
    gt_test  = os.path.join(test,  "SR")


    has_test = os.path.isdir(test) and os.path.isdir(lq_test) and len(os.listdir(lq_test)) > 0

    batch_size = 8
    lr_rate = 2e-4
    num_epochs = args.epochs

    scale_factor = 2


    train_dataset = SRDataset(lq_train, gt_train)
    val_dataset   = SRDataset(lq_val, gt_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, num_workers=1, shuffle=False)


    if has_test:
        test_dataset = SRDataset(lq_test, gt_test)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(">>> Test dataset detected. SR images will be exported when validation MS-SSIM improves.")
    else:
        test_loader = None
        print(">>> No test dataset found. Skipping TEST inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCAN2D(input_channels=1, output_channels=1, scale=scale_factor).to(device)


    if args.pretrained is not None:
        print(f">>> Loading pretrained model from {args.pretrained}")
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict)
        print(">>> Pretrained weights loaded.")


    criterion = CombinedLoss(alpha=0.6)


    if args.finetune:
        print(">>> Finetuning mode: using smaller learning rate")
        optimizer = optim.Adam(model.parameters(), lr=lr_rate * 0.1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)


    log_path = os.path.join(save_dir, "loss_log.txt")
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "w") as f:
        f.write("===== Training Configuration =====\n")
        f.write(f"Timestamp: {now}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"SaveDir: {save_dir}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr_rate}\n")
        f.write(f"Scale Factor: {scale_factor}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Augmentation: Hflip/Vflip/Rot90\n")
        f.write("=====================================\n\n")
        f.write("epoch,train_ms_ssim,val_ms_ssim,val_psnr,val_loss\n")


    best_val_loss = float("inf")
    best_ms_ssim = 0.0

    for epoch in range(num_epochs):
        print(f"\n===== Epoch [{epoch + 1}/{num_epochs}] =====")

        train_loss, train_ms_ssim = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_psnr, val_ms_ssim   = validate_val(model, val_loader, criterion, device)


        print(f"Epoch [{epoch + 1}/{num_epochs}]  "
              f"Train Loss: {train_loss:.6f}  Train MS-SSIM: {train_ms_ssim:.6f}  "
              f"Val Loss: {val_loss:.6f}  Val PSNR: {val_psnr:.6f}  Val MS-SSIM: {val_ms_ssim:.6f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch + 1},{train_ms_ssim:.6f},{val_ms_ssim:.6f},{val_psnr:.6f},{val_loss:.6f}\n")


        torch.save(model.state_dict(), os.path.join(save_dir, f"rcan_epoch{epoch + 1}.pth"))


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_rcan.pth"))


        if val_ms_ssim > best_ms_ssim:
            best_ms_ssim = val_ms_ssim

            best_ckpt_path = os.path.join(save_dir, "best_ssim_rcan.pth")
            torch.save(model.state_dict(), best_ckpt_path)

            if has_test:
                print(">>> New BEST MS-SSIM achieved. Running TEST inference...")

                state_dict = torch.load(best_ckpt_path, map_location=device)
                model.load_state_dict(state_dict)

                infer_and_save_test_sr(
                    model,
                    test_loader,
                    device,
                    best_ssim_test_sr_dir
                )

if __name__ == "__main__":
    main()
