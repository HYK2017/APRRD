import torch

class APR:
    def __init__(self, size=3):
        self.size = (size - 1) // 2
        self.window = None

    def _generate_window(self, device):
        window = torch.tensor([((a, b), (c, d))
                               for a in range(-self.size, self.size + 1)
                               for b in range(-self.size, self.size + 1)
                               for c in range(-self.size, self.size + 1)
                               for d in range(-self.size, self.size + 1)],
                              device=device)
        exception = ~((window[:, 0, 0] == window[:, 1, 0]) & (window[:, 0, 1] == window[:, 1, 1]))
        self.window = window[exception]

    def apply(self, y, fy):
        B, C, H, W = y.shape
        device = y.device

        if self.window is None or self.window.device != device:
            self._generate_window(device)

        random_indices = torch.randint(0, len(self.window), (B, H, W), device=device)

        dx_h1 = torch.clamp(torch.arange(W, device=device).repeat(H, 1).unsqueeze(0).expand(B, -1, -1) + self.window[random_indices, 0, 0], 0, W - 1)
        dy_h1 = torch.clamp(torch.arange(H, device=device).unsqueeze(1).repeat(1, W).unsqueeze(0).expand(B, -1, -1) + self.window[random_indices, 0, 1], 0, H - 1)
        dx_h2 = torch.clamp(torch.arange(W, device=device).repeat(H, 1).unsqueeze(0).expand(B, -1, -1) + self.window[random_indices, 1, 0], 0, W - 1)
        dy_h2 = torch.clamp(torch.arange(H, device=device).unsqueeze(1).repeat(1, W).unsqueeze(0).expand(B, -1, -1) + self.window[random_indices, 1, 1], 0, H - 1)

        idx_h1 = dy_h1 * W + dx_h1
        idx_h2 = dy_h2 * W + dx_h2

        h1_y = torch.gather(y.view(B, C, -1), 2, idx_h1.view(B, 1, -1).expand(-1, C, -1)).view(B, C, H, W)
        h2_y = torch.gather(y.view(B, C, -1), 2, idx_h2.view(B, 1, -1).expand(-1, C, -1)).view(B, C, H, W)
        h1_fy = torch.gather(fy.view(B, C, -1), 2, idx_h1.view(B, 1, -1).expand(-1, C, -1)).view(B, C, H, W)
        h2_fy = torch.gather(fy.view(B, C, -1), 2, idx_h2.view(B, 1, -1).expand(-1, C, -1)).view(B, C, H, W)
        return h1_y, h2_y, h1_fy, h2_fy
# Usage
# apr = APR()
# h1_y, h2_y, h1_fy, h2_fy = apr.apply(y, fy)
    
class Recharger:
    def __init__(self, percentage=1.0):
        self.percentage = percentage

    def generate_subset_mask(self, y):
        true_indices = torch.nonzero(y, as_tuple=False)

        num_true = true_indices.size(0)
        if num_true == 0:  # 만약 true 값이 없으면 그대로 반환
            return torch.zeros_like(y, dtype=torch.bool)

        num_to_keep = int(num_true * self.percentage)
        shuffled_indices = torch.randperm(num_true, device=y.device)
        keep_indices = true_indices[shuffled_indices[:num_to_keep]]

        subset_mask = torch.zeros_like(y, dtype=torch.bool)
        subset_mask[keep_indices[:, 0], keep_indices[:, 1], keep_indices[:, 2]] = True
        return subset_mask

    def apply(self, y, T_elements):
        T_cat = torch.stack(T_elements, dim=0)  # (distill_no, B, C, H, W)
        distill_no, B, C, H, W = T_cat.shape
        device = y.device

        recharger = torch.randint(0, distill_no, (B, H, W), device=device)

        y = y.permute(1, 0, 2, 3)
        T_cat = T_cat.permute(0, 2, 1, 3, 4)

        for no in range(distill_no):
            recharging_mask = (recharger == no)
            recharging_mask = self.generate_subset_mask(recharging_mask)
            T_cat[no, :, recharging_mask] = y[:, recharging_mask]

        T_RD_cat = T_cat.permute(0, 2, 1, 3, 4)
        T_RD_elements = [t.squeeze(0) for t in torch.chunk(T_RD_cat, chunks=distill_no, dim=0)]
        return T_RD_elements
# Usage
# recharger = Recharger()
# T_RD_elements = recharger.apply(y, T_elements)