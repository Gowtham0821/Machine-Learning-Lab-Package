
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image



IMAGE_PATH = "original.jpg"
SAVE_SIZE = (256, 256)
PATCH_SIZE = 32
NUM_CLASSES = 10    
TRUE_LABEL = 0 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)




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


# Load + resize

img = Image.open(IMAGE_PATH).convert("RGB").resize(SAVE_SIZE, Image.LANCZOS)
img.save("original_reference.png")
print(f"Image resized to {img.size[0]} x {img.size[1]}")

arr = np.asarray(img, dtype=np.float32) / 255.0
image = torch.from_numpy(arr).permute(2, 0, 1)
c, H, W = image.shape



# Patch splitting

def make_patches(image, ps):
    c, H, W = image.shape
    patches = []
    for r in range(math.ceil(H / ps)):
        for col in range(math.ceil(W / ps)):
            y, x = r * ps, col * ps
            patch = image[:, y:y+ps, x:x+ps]
            ph, pw = patch.shape[1], patch.shape[2]
            padded = torch.zeros(c, ps, ps)
            padded[:, :ph, :pw] = patch
            patches.append({"data": padded,
                            "x": x, "y": y,
                            "width": pw, "height": ph})
    return patches

patches = make_patches(image, PATCH_SIZE)
print(f"Patches: {len(patches)} ({math.ceil(H/PATCH_SIZE)} rows x "
      f"{math.ceil(W/PATCH_SIZE)} cols)")



# Model + gradients

torch.manual_seed(42)
model = LeakNet(PATCH_SIZE, NUM_CLASSES).to(DEVICE)
model.eval()
criterion = nn.CrossEntropyLoss()
label = torch.tensor([TRUE_LABEL], dtype=torch.long, device=DEVICE)

all_gradients = []
for i, p in enumerate(patches):
    x_in = p["data"].unsqueeze(0).to(DEVICE)
    loss = criterion(model(x_in), label)
    grads = torch.autograd.grad(loss, model.parameters())
    all_gradients.append([g.detach().cpu() for g in grads])
    del x_in, loss, grads
    if (i + 1) % 16 == 0 or i + 1 == len(patches):
        print(f" gradient computed for patch {i+1}/{len(patches)}")



# Save (label intentionally omitted)

torch.save({
    "model_state" : model.state_dict(),
    "gradients" : all_gradients,
    "original_size": (H, W),
    "channels" : c,
    "patch_size" : PATCH_SIZE,
    "num_classes" : NUM_CLASSES,
    "patches" : [{"x": p["x"], "y": p["y"],
                       "width": p["width"], "height": p["height"]}
                      for p in patches],
}, "gradients.pt")

print("\nClient done. Saved: gradients.pt + original_reference.png")
print(f"(For your own comparison only — the true label is {TRUE_LABEL}.)")






class Solution:
    def solveNQueens(self, n: int) -> list[list[str]]:
        res = []
        board = [['.'] * n for _ in range(n)]

        def isSafe(r, c):
            # upper-left diagonal
            i, j = r - 1, c - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1

            # left side
            j = c - 1
            while j >= 0:
                if board[r][j] == 'Q':
                    return False
                j -= 1

            # lower-left diagonal
            i, j = r + 1, c - 1
            while i < n and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i += 1
                j -= 1

            return True

        def backtrack(col):
            if col == n:
                res.append([''.join(row) for row in board])
                return

            for row in range(n):
                if isSafe(row, col):
                    board[row][col] = 'Q'
                    backtrack(col + 1)
                    board[row][col] = '.'

        backtrack(0)
        return res
