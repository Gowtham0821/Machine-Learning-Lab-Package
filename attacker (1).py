import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image



DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ITERATIONS  = 300          
RESTARTS    = 3
SAVE_ITERS  = [25, 50, 100, 200, 300]

PATIENCE    = 50           
REL_IMPROVE = 0.99        
GOOD_ENOUGH = 1e-9         
STOP_LOSS   = 1e-14        

print("Device:", DEVICE)



# Model  (identical to client.py)

class LeakNet(nn.Module):
    def __init__(self, patch=32, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding=2), nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, padding=2), nn.Sigmoid(),
        )
        self.fc = nn.Linear(12 * patch * patch, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))



# Gradient distance 

def grad_dist(dummy_grads, target_grads):
    return sum(((dg - tg) ** 2).mean()
               for dg, tg in zip(dummy_grads, target_grads))



# Attack one patch

def attack_patch(model, target_grads, P, num_classes, init=None):
    best = None

    for restart in range(RESTARTS):
        
        if restart == 0 and init is not None:
            dummy_x = init.clone().unsqueeze(0).to(DEVICE).requires_grad_(True)
        else:
            dummy_x = torch.randn(1, 3, P, P,
                                  device=DEVICE, requires_grad=True)
        dummy_y = torch.randn(1, num_classes,
                              device=DEVICE, requires_grad=True)

        
        opt = torch.optim.LBFGS(
            [dummy_x, dummy_y],
            lr=1, history_size=20, max_iter=10,
            tolerance_grad=1e-10, tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            out = model(dummy_x)
            ce  = -(torch.softmax(dummy_y, -1)
                    * torch.log_softmax(out, -1)).sum()
            dg  = torch.autograd.grad(ce, model.parameters(),
                                      create_graph=True)
            dist = grad_dist(dg, target_grads)
            dist.backward(inputs=[dummy_x, dummy_y])
            return dist

        
        snaps = {}
        last_loss       = float("inf")
        best_in_restart = float("inf")
        no_improve      = 0

        for it in range(ITERATIONS):
            val = opt.step(closure)
            if val is not None:
                last_loss = val.item()

            
            if last_loss < best_in_restart * REL_IMPROVE:
                best_in_restart = last_loss
                no_improve      = 0
            else:
                no_improve += 1

            if it in SAVE_ITERS:
                snaps[it] = dummy_x.detach()[0].cpu().clamp(0, 1)

            if no_improve >= PATIENCE:
                break
            if last_loss < STOP_LOSS:
                break

        
        pred_label = torch.softmax(dummy_y.detach(), -1).argmax(-1).item()

        if best is None or last_loss < best["loss"]:
            best = {
                "x":     dummy_x.detach()[0].cpu().clamp(0, 1),
                "y":     pred_label,
                "loss":  last_loss,
                "snaps": snaps,
            }

        del dummy_x, dummy_y, opt

        
        if best["loss"] < GOOD_ENOUGH:
            break

    return best



# Load data

data        = torch.load("gradients.pt", map_location="cpu")
P           = data["patch_size"]
num_classes = data["num_classes"]
patches     = data["patches"]
all_grads   = data["gradients"]
H, W        = data["original_size"]

model = LeakNet(P, num_classes).to(DEVICE)
model.load_state_dict(data["model_state"])
model.eval()

print(f"Image {W}x{H}  |  patch {P}x{P}  |  {len(patches)} patches")

canvas = {s: torch.zeros(3, H, W) for s in SAVE_ITERS}
canvas["final"] = torch.zeros(3, H, W)



# Reconstruct every patch

prev_result = None
labels_seen = []
t_start     = time.time()

for n, (info, tg) in enumerate(zip(patches, all_grads)):
    tg_dev = [g.to(DEVICE) for g in tg]

    t0  = time.time()
    res = attack_patch(model, tg_dev, P, num_classes, init=prev_result)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    per_patch = time.time() - t0

    prev_result = res["x"]
    labels_seen.append(res["y"])

    x, y   = info["x"], info["y"]
    pw, ph = info["width"], info["height"]

    canvas["final"][:, y:y+ph, x:x+pw] = res["x"][:, :ph, :pw]
    for s, snap in res["snaps"].items():
        canvas[s][:, y:y+ph, x:x+pw] = snap[:, :ph, :pw]

    elapsed = time.time() - t_start
    eta     = (elapsed / (n + 1)) * (len(patches) - n - 1)
    print(f"Patch {n+1:4d}/{len(patches)}  |  "
          f"loss {res['loss']:.3e}  |  "
          f"label {res['y']}  |  "
          f"pos ({x},{y})  |  "
          f"{per_patch:.1f}s  |  ETA {eta/60:.1f} min")

    del tg_dev
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()



# Save outputs

def save_canvas(t, path):
    arr = (t.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(arr, "RGB").save(path)
    print(f"  Saved {path}")

os.makedirs("progress", exist_ok=True)
save_canvas(canvas["final"], "reconstructed.png")
for s in SAVE_ITERS:
    save_canvas(canvas[s], f"progress/iter_{s:04d}.png")

print("\nProgress images saved in  progress/")


print(f"\nRecovered labels ({len(labels_seen)}): {labels_seen}")
print(f"Label histogram: {dict(Counter(labels_seen))}")



# Quality metrics

if os.path.exists("original_reference.png"):
    ref = np.asarray(Image.open("original_reference.png").convert("RGB"),
                     dtype=np.float32) / 255.0
    rec = np.asarray(Image.open("reconstructed.png").convert("RGB"),
                     dtype=np.float32) / 255.0
    mse  = float(((ref - rec) ** 2).mean())
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"\nMSE : {mse:.5f}")
    print(f"PSNR: {psnr:.2f} dB")
    print("(Paper reports MSE < 0.03 for successful reconstruction)")

print(f"\nTotal runtime: {(time.time() - t_start)/60:.2f} min")
print("\nAttacker done.")