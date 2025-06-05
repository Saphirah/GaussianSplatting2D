import math
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import os
from PIL import Image
import pyperclip
import matplotlib.pyplot as plt


class GS2D:
    def __init__(self,
                 img_size=(100, 100, 3),
                 num_epoch=20,
                 num_iter_per_epoch=100,
                 num_samples=300,
                 num_max_samples=3000,
                 sigma_thre=10,
                 grad_thre=0.06,
                 device="cuda"
                 ):
        self.img_size = img_size
        self.num_epoch = num_epoch
        self.num_iter_per_epoch = num_iter_per_epoch
        self.num_samples = num_samples
        self.device = device
        self.sigma_thre = sigma_thre
        self.grad_thre = grad_thre
        self.num_max_samples = num_max_samples

        h, w = self.img_size[:2]
        xx, yy = torch.arange(w), torch.arange(h)
        x, y = torch.meshgrid(xx, yy, indexing="xy")  # (h,w)
        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def draw_gaussian(self, w):
        sigma, rho, mean, color, alpha = self.parse_param(w)
        r = rho.view(-1, 1, 1)
        sx = sigma[:, :1, None]
        sy = sigma[:, -1:, None]
        dx = self.x.unsqueeze(0) - mean[:, 0].view(-1, 1, 1)
        dy = self.y.unsqueeze(0) - mean[:, 1].view(-1, 1, 1)
        v = -0.5 * (((sy * dx) ** 2 + (sx * dy) ** 2) - 2 * dx * dy * r * sx * sy) / (
                    sx ** 2 * sy ** 2 * (1 - r ** 2) + 1e-8)
        v = torch.exp(v)
        v = v * alpha.view(-1, 1, 1)
        img = torch.sum(v.unsqueeze(1) * color.view(-1, 3, 1, 1), dim=0)
        return torch.clamp(img, 0, 1)

    def random_init_param(self):
        # Sample size, aspect, color, alpha as before
        sigma = torch.rand(self.num_samples, 2, device=self.device) - 2.5

        # Rotation
        rho = torch.rand(self.num_samples, 1, device=self.device) * 2 - 1
        color = torch.atanh(torch.rand(self.num_samples, 3, device=self.device))
        alpha = torch.zeros(self.num_samples, 1, device=self.device) - 0.01

        # Center-biased mean sampling via Beta(2,2)
        beta_dist = torch.distributions.Beta(2.5, 2.5)
        u_centered = beta_dist.sample((self.num_samples, 2)).to(self.device)  # in [0,1], peaks at 0.5
        mean = torch.atanh((u_centered * 2.0 - 1.0).clamp(-0.999, 0.999))

        # Assemble parameter tensor
        w = torch.cat([sigma, rho, mean, color, alpha], dim=1)
        return nn.Parameter(w)

    def parse_param(self, w):
        sigma = (torch.sigmoid(w[:, 0:2])) * torch.tensor(self.img_size[:2][::-1]).to(self.device) * 0.25
        rho = torch.tanh(w[:, 2:3])
        mean = (0.5 * torch.tanh(w[:, 3:5]) + 0.5) * torch.tensor(self.img_size[:2][::-1]).to(self.device)
        color = 0.5 * torch.tanh(w[:, 5:8]) + 0.5
        alpha = 0.5 * torch.tanh(w[:, 8:9]) + 0.5
        return sigma, rho, mean, color, alpha

    def update_w(self, w_old: torch.nn.Parameter, _grad: torch.Tensor) -> torch.Tensor:
        # 1) Detach
        w = w_old.detach()
        grad = _grad.detach()

        # 2) compute sigma in pixels and aspect ratio
        size_t = torch.tensor(self.img_size[:2][::-1], device=self.device)
        sigma_px = torch.sigmoid(w[:, :2]) * size_t * 0.25
        sx, sy = sigma_px[:, 0], sigma_px[:, 1]
        asp = sx / sy
        asp = torch.where(asp < 1, 1 / asp, asp)

        # 3) prune extreme aspect ratios <1:10
        keep = asp <= 5.0
        pruned_count = (~keep).sum().item()
        print(f"Removed Splats based on A/R: {pruned_count}")
        if not keep.any():
            return w
        w, grad = w[keep], grad[keep]

        # 4) recompute masks
        grad_norm = torch.norm(2 * grad[:, 3:5] / (1 - torch.tanh(w[:, 3:5]) ** 2 + 1e-8), dim=1, p=2)
        sigma_px = torch.sigmoid(w[:, :2]) * size_t * 0.25
        sigma_norm = torch.norm(sigma_px, dim=1, p=2)
        grad_mask = grad_norm > self.grad_thre
        sigma_mask = sigma_norm > self.sigma_thre

        # 5) split paths
        w_save = w[~grad_mask]
        w_scale = w[grad_mask & sigma_mask].clone()
        w_split = w[grad_mask & ~sigma_mask].clone()

        # 6) logit inversion for scale branch
        inv = (sigma_px[grad_mask & sigma_mask] / (size_t * 0.25)).clamp(1e-6, 1 - 1e-6)
        w_scale[:, :2] = torch.log(inv) - torch.log(1 - inv)

        # 7) copies for resample
        w_scale_copy = w_scale.clone()
        w_scale_copy[:, 3:5] -= grad[grad_mask & sigma_mask, 3:5]
        w_split_copy = w_split.clone()

        # 8) assemble
        w1 = torch.cat([w_save, w_scale, w_split], dim=0)
        w2 = torch.cat([w_scale_copy, w_split_copy], dim=0)
        if w2.numel():
            w2 = w2[torch.randperm(w2.size(0), device=self.device)]
        total = w1.size(0) + w2.size(0)
        if total > self.num_max_samples:
            w2 = w2[: self.num_max_samples - w1.size(0)]
        return torch.cat([w1, w2], dim=0)

    def save_splat_data(self, splat_file, w):
        base_dir = os.path.dirname(splat_file)
        base_name = os.path.basename(splat_file)
        os.makedirs(base_dir, exist_ok=True)
        torch.save(w, splat_file)
        torch.save(w.half(), os.path.join(base_dir, "16-bit-quantized-" + base_name))
        torch.save(w.to(torch.int8), os.path.join(base_dir, "8-bit-quantized-" + base_name))
        print(f"Saved splat data in {splat_file}")

    def load_splat_data(self, splat_file):
        print("Loading splat data from", splat_file)
        w_loaded = torch.load(splat_file)
        w_loaded = w_loaded.to(torch.float32)
        return torch.nn.Parameter(w_loaded.requires_grad_())

    def train(self, target, splat_file):
        if os.path.exists(splat_file):
            w = self.load_splat_data(splat_file)
            print(f"Loaded splat data from {splat_file}")
        else:
            w = self.random_init_param()

        for epoch in range(self.num_epoch):
            torch.cuda.empty_cache()
            optimizer = torch.optim.AdamW([w], lr=0.005)
            bar = tqdm(range(self.num_iter_per_epoch))
            for _iter in bar:
                optimizer.zero_grad()
                predicted = self.draw_gaussian(w)
                predicted_img = predicted.permute(1, 2, 0).cpu().detach().numpy()

                plt.imshow(predicted_img)
                plt.title(f"Epoch {epoch}")
                plt.pause(0.001)  # Non-blocking show
                plt.clf()  # Clear the figure for the next image

                loss = nn.functional.l1_loss(predicted, target)

                sigma, *_ = self.parse_param(w)
                sx, sy = sigma[:, 0], sigma[:, 1]
                ratio = sx / (sy + 1e-6)
                ratio = torch.where(ratio < 1.0, 1.0 / ratio, ratio)
                penalty = (ratio * 0.2) ** 6
                loss += penalty.mean()

                loss.backward()

                optimizer.step()
                bar.set_description(f"[Ep@{epoch}] [Loss@{loss.item():.6f}] [Current Samples@{w.size(0)}] [AR Penalty@{penalty.mean()}]")
            _grad = w.grad.data
            optimizer.zero_grad()
            self.save_splat_data(splat_file, w)
            with torch.no_grad():
                w = self.update_w(w.detach(), _grad.detach())
                w = torch.nn.Parameter(w)

                pred_out = torchvision.utils.make_grid([predicted, target], nrow=2)
                base_dir = os.path.dirname(splat_file)
                os.makedirs(base_dir + "/images", exist_ok=True)
                torchvision.utils.save_image(pred_out, base_dir + f"/images/{epoch}.jpg")

def test_quantized_draw(splat_file: str, gs: GS2D):
    w = gs.load_splat_data(splat_file)
    img = gs.draw_gaussian(w)

    torchvision.utils.save_image(img, "test.jpg")
    plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy())
    plt.show()


def encode_splats_to_uvec4(splat_file: str, gs: GS2D):
    """
    splat_file: Pfad zur .pt–Datei mit float32-Parametern
    img_size: (H, W, C)
    Ausgabe: uvec4–Daten zum Einfügen ins ShaderToy
    """
    H, W, _ = gs.img_size
    w = torch.load(splat_file, map_location="cuda")
    sigma, rho, mean, color, alpha = gs.parse_param(
        torch.nn.Parameter(w))
    N = w.shape[0]
    output = ""
    #print(f"const uvec4 gaussian_data[{N}] = uvec4[{N}](")
    for i in range(N):
        # --- Colors 6 Bit ---
        R6 = int((color[i,0] *  63).clamp(0,63).item())
        G6 = int((color[i,1] *  63).clamp(0,63).item())
        B6 = int((color[i,2] *  63).clamp(0,63).item())
        # --- Orientation 8 Bit ---
        sx, sy = sigma[i].tolist()
        r = rho[i].item()
        theta = 0.5 * math.atan2(2*r*sx*sy, sx*sx - sy*sy)
        Th8 = int((theta / math.pi) * 255) & 0xFF
        # --- Amplitude 8 Bit ---
        A8 = int((alpha[i].item() * 255)) & 0xFF

        # u0: [ B6 | G6 | R6 | Th8 | A6 ] = 6+6+6+8+6 = 32 Bit
        u0 = (B6 <<   0) \
           | (G6 <<   6) \
           | (R6 <<  12) \
           | (Th8 << 18) \
           | ((A8 >> 2) << 26)   # nur die obersten 6 Bit von A8

        # quant24 for Position und Scale
        def quant24(x_norm):
            return int(round(x_norm * (2**24 - 1))) & 0xFFFFFF

        # Mean
        qmx = quant24(mean[i,0].item() / W)
        qmy = quant24(mean[i,1].item() / H)
        mu_base = (qmx >>  8) & 0xFFFF
        mu_ext  =  qmx &  0xFF
        my_base = (qmy >>  8) & 0xFFFF
        my_ext  =  qmy &  0xFF
        u1 = (mu_base <<  0) | (my_base << 16)
        u3 = (mu_ext  <<  0) | (my_ext  <<  8)

        # Sigma
        qsx = quant24(sx / W)
        qsy = quant24(sy / H)
        sx_base = (qsx >> 8) & 0xFFFF
        sx_ext  =  qsx &  0xFF
        sy_base = (qsy >> 8) & 0xFFFF
        sy_ext  =  qsy &  0xFF
        u2 = (sx_base <<  0) | (sy_base << 16)
        u3 |= (sx_ext << 16) | (sy_ext << 24)

        #print(f"    uvec4({u0}u, {u1}u, {u2}u, {u3}u),")
        #output += f"{u0},{u1},{u2},{u3},"
        output += f"uvec4({u0}u, {u1}u, {u2}u, {u3}u),"

    #Copy to clipboard
    pyperclip.copy(output)
    print("Splats copied to clipboard")

if __name__ == "__main__":
    device = "cuda"
    img_file = "CuteGirlA.jpg"
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    splat_file = "training/" + base_name + "/" + base_name + ".pt"
    img = Image.open(img_file).convert("RGB")
    tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(512, 512)),
        torchvision.transforms.ToTensor()
    ])
    img = tsfm(img)
    h, w_dim = img.size()[1:]
    gs = GS2D(
        num_epoch=1000,
        img_size=(h, w_dim, 3),
        device=device,
        num_iter_per_epoch=100,
        num_samples=1000,
        num_max_samples=1000,
        #grad_thre=0.0005
    )
    img = img.to(device)
    try:
        gs.train(img, splat_file)
    except KeyboardInterrupt:
        print("Training interrupted. Saving splat data...")
    #test_quantized_draw("training/" + base_name + "/" + "16-bit-quantized-" + base_name + ".pt", gs=gs)
    encode_splats_to_uvec4(splat_file, gs)
