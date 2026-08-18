"""Does the CLASS survive when the pixels do not?

Pixel MSE says recovery fails past t*~0.45.  But the claim in the active-diffusion
paper is about semantic memory: eta stores class identity, which should outlive
pixel identity.  This measures both on the same samples.

Reference class = the classifier's own prediction on the ORIGINAL, not the CIFAR
label, so classifier error cancels out of the agreement rate.

  python classify_recovery.py DownloadsFromCluster/recover_t0.*.npz
"""
import argparse, glob, re, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--data_dir", default="/home/svaikunt/CIFAR10/data")
ap.add_argument("--epochs", type=int, default=12, help="fallback classifier training")
ap.add_argument("--local", action="store_true",
                help="skip torch.hub and train the small CNN locally")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- classifier
def get_classifier():
    try:
        if a.local:
            raise RuntimeError('--local requested')
        m = torch.hub.load("chenyaofo/pytorch-cifar-models",
                           "cifar10_resnet20", pretrained=True,
                           verbose=False, trust_repo=True)
        print("classifier: pretrained cifar10_resnet20 (92.6% top-1)", flush=True)
        return m.to(dev).eval()
    except Exception as e:
        print(f"torch.hub unavailable ({type(e).__name__}); training a small CNN", flush=True)
    from torchvision import datasets, transforms
    tf_tr = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2470, 0.2435, 0.2616))])
    tr = torch.utils.data.DataLoader(
        datasets.CIFAR10(a.data_dir, train=True, download=False, transform=tf_tr),
        batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    def blk(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o),
                             nn.ReLU(True), nn.Conv2d(o, o, 3, 1, 1, bias=False),
                             nn.BatchNorm2d(o), nn.ReLU(True))
    m = nn.Sequential(blk(3, 64), nn.MaxPool2d(2), blk(64, 128), nn.MaxPool2d(2),
                      blk(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                      nn.Linear(256, 10)).to(dev)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, epochs=a.epochs, steps_per_epoch=len(tr))
    for ep in range(a.epochs):
        m.train(); c = n = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = m(x); loss = F.cross_entropy(out, y)
            loss.backward(); opt.step(); sch.step()
            c += (out.argmax(1) == y).sum().item(); n += y.numel()
        print(f"  epoch {ep+1}/{a.epochs}  train acc {c/n:.3f}", flush=True)
    return m.eval()

net = get_classifier()
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)

@torch.no_grad()
def predict(x_pm1):                       # images arrive in [-1, 1]
    x = torch.as_tensor(x_pm1, device=dev, dtype=torch.float32)
    x = ((x + 1) / 2).clamp(0, 1)
    out = []
    for i in range(0, len(x), 512):
        out.append(net((x[i:i+512] - MEAN) / STD).argmax(1).cpu())
    return torch.cat(out).numpy()

# ---------------------------------------------------------------- evaluate
print(f"\n{'t*':>6} {'n':>5} {'pixel corr':>11} {'class kept':>11} "
      f"{'shuffled class':>15} {'chance':>7}")
for f in sorted(a.npz):
    T = re.search(r"t?(0\.\d+)\.npz", f)
    T = T.group(1) if T else f
    d = np.load(f)
    o = d["orig"].astype(np.float32)
    m = d["matched"].astype(np.float32)[-1][:, [0, 2, 4]]
    s = d["shuffled"].astype(np.float32)[-1][:, [0, 2, 4]]
    ref, pm, ps = predict(o), predict(m), predict(s)
    corr = np.median([np.corrcoef(m[i].ravel(), o[i].ravel())[0, 1] for i in range(len(o))])
    print(f"{T:>6} {len(o):5d} {corr:11.3f} {(pm == ref).mean():11.1%} "
          f"{(ps == ref).mean():15.1%} {0.1:7.1%}")
print("\n'class kept' = recovered image gets the same predicted class as the original.")
print("Reference is the classifier's prediction on the original, so its errors cancel.")
