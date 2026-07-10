"""OFA evaluation."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from pgra.datasets import MstarComponents  # noqa: E402
from pgra.models import densenet121_pgra  # noqa: E402


@torch.no_grad()
def evaluate(loader, model, device, tag):
    model.eval()
    correct = 0
    total = 0
    for images, asc_part, target, _ in tqdm(
            loader, file=sys.stdout, desc=tag, ncols=80):
        images = images.float().to(device)
        target = target.to(device)
        asc_part = asc_part.float().to(device)
        logits = model(images, asc_part)
        correct += int((logits.argmax(dim=1) == target).sum().item())
        total += target.size(0)
    return correct / max(total, 1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a PGRA / baseline checkpoint on OFA test sets")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--attention_setting", action="store_true")
    p.add_argument("--num_classes",   type=int,   default=10)
    p.add_argument("--num_parts",     type=int,   default=4)
    p.add_argument("--angle_count",   type=int,   default=5)
    p.add_argument("--angle_interval", type=float, default=2.5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--device",        default="0")
    p.add_argument("--val_list",  default=None,
                   help="(optional) validation set for sanity check")
    p.add_argument("--ofa1_list", default=None)
    p.add_argument("--ofa2_list", default=None)
    p.add_argument("--ofa3_list", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    angle_range = [i * args.angle_interval
                   for i in range(-args.angle_count, args.angle_count + 1)]

    model = densenet121_pgra(
        num_classes=args.num_classes,
        part_num=args.num_parts,
        attention_setting=args.attention_setting,
        angle_range=angle_range,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print("Loaded:", args.ckpt)
    print("Attention setting (PGRA enabled):", args.attention_setting)

    transform = transforms.Compose([transforms.ToTensor()])
    results = []

    def _run(list_path, tag):
        if not list_path:
            return
        ds = MstarComponents(list_path, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        print("\n{}: {} samples".format(tag, len(ds)))
        acc = evaluate(loader, model, device, tag)
        print("{} acc = {:.4f}".format(tag, acc))
        results.append((tag, len(ds), acc))

    _run(args.val_list,  "VAL")
    _run(args.ofa1_list, "OFA-1")
    _run(args.ofa2_list, "OFA-2")
    _run(args.ofa3_list, "OFA-3")

    print("\n========== Summary ==========")
    for tag, n, acc in results:
        print("{:>6}  n={:>4d}  acc={:.4f}".format(tag, n, acc))


if __name__ == "__main__":
    main()
