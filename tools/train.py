"""Training entrypoint."""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from pgra.datasets import MstarComponents  # noqa: E402
from pgra.models import densenet121_pgra  # noqa: E402
from pgra.utils import EarlyStopping  # noqa: E402

try:
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:
    SummaryWriter = None


def batch_topk_correct(logits, target):
    with torch.no_grad():
        return int((logits.argmax(dim=1) == target).sum().item())


@torch.no_grad()
def evaluate_dataset(loader, model, device, criterion=None, desc="Eval"):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    n_batches = 0
    for images, asc_part, target, _ in tqdm(
            loader, file=sys.stdout, desc=desc, ncols=80):
        images = images.float().to(device)
        target = target.to(device)
        asc_part = asc_part.float().to(device)

        logits = model(images, asc_part)
        if criterion is not None:
            loss_sum += float(criterion(logits, target).item())
            n_batches += 1
        correct += batch_topk_correct(logits, target)
        total += target.size(0)

    return {
        "acc": correct / max(total, 1),
        "loss": (loss_sum / n_batches) if n_batches else float("nan"),
    }


def build_model(args, angle_range):
    tag = "+ PGRA" if args.attention_setting else "baseline"
    print("Building DenseNet121 {}".format(tag))
    return densenet121_pgra(
        num_classes=args.num_classes,
        part_num=args.num_parts,
        attention_setting=args.attention_setting,
        angle_range=angle_range,
    )


def build_loader(list_path, transform, batch_size, num_workers, shuffle=False):
    if not list_path:
        return None
    ds = MstarComponents(list_path, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train DenseNet121 (+ PGRA) on MSTAR")

    p.add_argument("--train_list", required=True, help="Train index file")
    p.add_argument("--val_list",   required=True, help="Validation index file")
    p.add_argument("--ofa1_list",  default=None, help="OFA-1 index file")
    p.add_argument("--ofa2_list",  default=None, help="OFA-2 index file")
    p.add_argument("--ofa3_list",  default=None, help="OFA-3 index file")
    p.add_argument("--num_parts",  type=int, default=4,
                   help="Number K of ASC component channels")

    p.add_argument("--save_path",  required=True,
                   help="Output directory (checkpoints + result.txt)")
    p.add_argument("--result_file", default="result.txt")

    p.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")

    p.add_argument("--num_epochs",  type=int,   default=250)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=100)
    p.add_argument("--num_runs",    type=int,   default=5,
                   help="Independent training runs (5 by paper convention)")
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--num_classes", type=int,   default=10)

    p.add_argument("--attention_setting", action="store_true",
                   help="Enable PGRA blocks (omit for baseline)")
    p.add_argument("--pretrain", default=None,
                   help="Optional path to a pre-trained state_dict")

    p.add_argument("--angle_count",    type=int,   default=5,
                   help="k in theta = {-k*delta, ..., k*delta}")
    p.add_argument("--angle_interval", type=float, default=2.5,
                   help="delta (deg)")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, num_epochs, run_id, num_runs):
    model.train()
    running = 0.0
    pbar = tqdm(loader, file=sys.stdout, ncols=100,
                desc="run {}/{}  epoch {}/{}".format(
                    run_id + 1, num_runs, epoch + 1, num_epochs))
    for images, asc_part, labels, _ in pbar:
        images = images.float().to(device)
        labels = labels.to(device)
        asc_part = asc_part.float().to(device)

        logits = model(images, asc_part)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        pbar.set_postfix({"loss": "{:.5f}".format(loss.item())})
    return running / max(len(loader), 1)


def run_single(args, run_id, angle_range, device, transform, loaders):
    log_dir = os.path.join(args.save_path, "log{}".format(run_id))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else None

    ckpt = os.path.join(args.save_path, "{}.pth".format(run_id))
    stopper = EarlyStopping(ckpt, patience=args.patience, mode="max")

    train_loader = build_loader(args.train_list, transform,
                                args.batch_size, args.num_workers, shuffle=True)
    val_loader = loaders["val"]

    model = build_model(args, angle_range).to(device)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain, map_location=device))
        print("Loaded pre-trained weights from", args.pretrain)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    last_epoch = 0
    for epoch in range(args.num_epochs):
        last_epoch = epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.num_epochs, run_id, args.num_runs)

        val_metrics = evaluate_dataset(val_loader, model, device,
                                       criterion=criterion, desc="val")
        val_acc = val_metrics["acc"]

        if writer is not None:
            writer.add_scalar("accuracy", val_acc, epoch + 1)
            writer.add_scalars("loss", {
                "train": train_loss,
                "val": val_metrics["loss"],
            }, epoch + 1)
        print("epoch [{}/{}] train_loss={:.5f}  val_acc={:.5f}".format(
            epoch + 1, args.num_epochs, train_loss, val_acc))

        stopper.step(val_acc, model, epoch=epoch)
        if writer is not None:
            writer.add_scalar("no_improve_count", stopper.counter, epoch + 1)
        if stopper.should_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    final = {
        "val":  evaluate_dataset(val_loader, model, device, desc="VAL")["acc"],
        "stop_epoch": last_epoch,
    }
    for tag in ("ofa1", "ofa2", "ofa3"):
        loader = loaders.get(tag)
        if loader is None:
            continue
        acc = evaluate_dataset(loader, model, device, desc=tag.upper())["acc"]
        final[tag] = acc
        print("[run {}] {} acc = {:.5f}".format(run_id + 1, tag.upper(), acc))
    return final


def dump_results(args, records):
    def _row(key, extract):
        vals = [str(extract(r)) for r in records if extract(r) is not None]
        return "{}:{}\n".format(key, "\t".join(vals))

    lines = [
        _row("val",        lambda r: r.get("val")),
        _row("stop_epoch", lambda r: r.get("stop_epoch")),
    ]
    for tag_lc, tag_uc in (("ofa1", "OFA1"), ("ofa2", "OFA2"), ("ofa3", "OFA3")):
        if any(r.get(tag_lc) is not None for r in records):
            lines.append(_row(tag_uc, lambda r, t=tag_lc: r.get(t)))

    out = os.path.join(args.save_path, args.result_file)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    angle_range = [i * args.angle_interval
                   for i in range(-args.angle_count, args.angle_count + 1)]
    print("Rotation angles ({}):".format(len(angle_range)), angle_range)

    transform = transforms.Compose([transforms.ToTensor()])

    loaders = {
        "val":  build_loader(args.val_list,  transform, args.batch_size, args.num_workers),
        "ofa1": build_loader(args.ofa1_list, transform, args.batch_size, args.num_workers),
        "ofa2": build_loader(args.ofa2_list, transform, args.batch_size, args.num_workers),
        "ofa3": build_loader(args.ofa3_list, transform, args.batch_size, args.num_workers),
    }
    for tag, loader in loaders.items():
        if loader is not None:
            print("{}: {} samples".format(tag, len(loader.dataset)))

    records = []
    for run_id in range(args.num_runs):
        record = run_single(args, run_id, angle_range, device, transform, loaders)
        records.append(record)
        dump_results(args, records)

    print("Done. Per-run val:", [r["val"] for r in records])
    for tag in ("ofa1", "ofa2", "ofa3"):
        vals = [r.get(tag) for r in records if r.get(tag) is not None]
        if vals:
            print("Per-run {}: {}".format(tag.upper(), vals))


if __name__ == "__main__":
    main()
