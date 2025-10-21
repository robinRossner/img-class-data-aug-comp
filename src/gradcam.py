import torch
import torch.nn.functional as F

def grad_cam(model, img_tensor, target_layer, class_idx=None):
    model.eval()
    activations, gradients = [], []

    # --- Hooks ---
    def forward_hook(module, input, output):
        activations.append(output.detach())
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Use safe backward hook version
    if hasattr(target_layer, "register_full_backward_hook"):
        handle_b = target_layer.register_full_backward_hook(backward_hook)
    else:
        handle_b = target_layer.register_backward_hook(backward_hook)
    handle_f = target_layer.register_forward_hook(forward_hook)

    # --- Forward ---
    img_batch = img_tensor.unsqueeze(0)
    output = model(img_batch)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # --- Backward ---
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    # --- Compute CAM ---
    grads = gradients[0].mean(dim=[2, 3], keepdim=True)   # GAP over spatial dims
    act = activations[0]
    cam = (grads * act).sum(dim=1).squeeze()              # weighted sum
    cam = torch.relu(cam)

    # Normalize and resize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=img_tensor.shape[1:],  # (H, W)
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # --- Clean up hooks ---
    handle_f.remove()
    handle_b.remove()

    return cam
