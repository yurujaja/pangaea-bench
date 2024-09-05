import torch
from utils.registry import LOSS_REGISTRY
from torch.nn import functional as F


@LOSS_REGISTRY.register()
def CrossEntropy(cfg):
    # if cfg["use_weights"]:
    return torch.nn.CrossEntropyLoss(ignore_index = cfg["ignore_index"]) #, weights = )

@LOSS_REGISTRY.register()
def WeightedCrossEntropy(cfg):
    weights = [1/w for w in cfg["distribution"]]
    loss_weights = torch.Tensor(weights).to("cuda")
    return torch.nn.CrossEntropyLoss(ignore_index = cfg["ignore_index"], weight = loss_weights)

@LOSS_REGISTRY.register()
def MSELoss(cfg):
    return torch.nn.MSELoss()


@LOSS_REGISTRY.register()
class DICELoss(torch.nn.Module):
    def __init__(self, cfg):
        super(DICELoss, self).__init__()
        self.ignore_index = cfg["ignore_index"]
    
    def forward(self, logits, target):
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        
        num_classes = logits.shape[1]
        mask = (target != self.ignore_index)
        #mask_expand = mask.unsqueeze(1).expand_as(probs)
        target_temp = target.clone()
        target_temp[~mask] = 0

        target_one_hot = F.one_hot(target_temp, num_classes=num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot * mask.unsqueeze(1).float()

        intersection = torch.sum(probs * target_one_hot, dim=(2, 3))
        union = torch.sum(probs + target_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        # dice_loss = 1 - dice_score.mean(dim=1).mean()
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score
        
        return dice_loss

    