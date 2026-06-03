"""Uczenie i pomiar referencji PyTorch — do porównania z biblioteką Julia w artykule KM2.

Uruchomienie:
    pip install -r requirements.txt
    python train.py

Porównanie z Julią prowadzimy na CPU (biblioteka Julia działa na CPU), żeby liczby
czasu były porównywalne. Mierzymy czas/epokę i końcową dokładność na zbiorze testowym.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FlexNet, he_init


def evaluate(net, loader, device):
    net.eval()
    correct = total = 0
    loss_sum = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss_sum += loss_fn(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total, loss_sum / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tf = transforms.ToTensor()  # -> FloatTensor (1, 28, 28) w [0, 1]
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=1000, shuffle=False)

    net = FlexNet().to(device)
    net.apply(he_init)
    opt = torch.optim.SGD(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()  # cross-entropy na logitach

    total_t = 0.0
    acc = loss = 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.perf_counter()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        dt = time.perf_counter() - t0
        total_t += dt
        acc, loss = evaluate(net, test_dl, device)
        print(f"epoch={epoch}  time={dt:.1f}s  test_acc={acc:.2f}%")

    tpe = total_t / args.epochs
    print(f"\nRAZEM: {total_t:.1f}s ({total_t/60:.1f} min), {args.epochs} epok, device={device}")
    print(f"  time_per_epoch_s  {tpe:.2f}")
    print(f"  test_acc          {acc:.2f}")
    print(f"  test_loss         {loss:.4f}")

    # dopisz wiersz do reference/results.csv (tabela porownawcza)
    # alloc_gib puste — pomiar pamieci wg polecenia dotyczy tylko rozwiazan w Julii
    results = Path(__file__).resolve().parent.parent / "results.csv"
    header = "solution,framework,time_per_epoch_s,total_time_s,alloc_gib,test_acc,test_loss\n"
    if not results.exists():
        results.write_text(header)
    with results.open("a") as f:
        f.write(f"pytorch-reference,Python / PyTorch,{tpe:.2f},{total_t:.2f},,{acc:.2f},{loss:.4f}\n")
    print(f"\nzapisano wiersz 'pytorch-reference' do {results}")


if __name__ == "__main__":
    main()
