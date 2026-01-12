import torch
from deeplens import GeoLens

def grad_sanity_check():
    torch.manual_seed(0)

    lens = GeoLens(filename="./datasets/lenses/camera/ef35mm_f2.0.json", device="cuda")
    # 确保输入尺寸与 sensor_res 匹配（GeoLens 的 render 要求匹配）
    lens.set_sensor_res(sensor_res=(256, 256))  # (W, H)

    optim = lens.get_optimizer(lrs=[1e-3, 1e-4, 0, 0])
    optim.zero_grad()

    W, H = lens.sensor_res  # (W, H)
    img = torch.rand(1, 3, H, W, device="cuda")  # 注意张量是 (B,C,H,W)

    img_render = lens.render(img, depth=-1000.0, method="ray_tracing", spp=8)

    # 任意标量 loss 都可以；这里用一个“让输出接近输入”的简单目标
    loss = (img_render - img).abs().mean()
    loss.backward()

    # 打印梯度统计
    print("loss =", loss.item())
    has_grad = 0
    for name, p in lens.named_parameters():
        if p.requires_grad:
            g = p.grad
            if g is not None:
                has_grad += 1
                print(f"{name:60s} grad_mean={g.abs().mean().item():.3e} grad_max={g.abs().max().item():.3e}")
            else:
                print(f"{name:60s} grad=None")
    print("n_params_with_grad =", has_grad)

if __name__ == "__main__":
    grad_sanity_check()
