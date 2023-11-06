import torch

class SALR:
    def __init__(
        self, 
        optimizer,
        learning_rate: float,
        total_epochs: int
    ):
        self.optimizer = optimizer
        self.base = learning_rate
        self.total_epochs = total_epochs
        self.sharpness_step = []
        self.sharpness_history = []

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3/10:
            lr = self.base
        elif epoch < self.total_epochs * 6/10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10:
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3
            
        sharpness_epoch = torch.mean((torch.tensor(self.sharpness_step)))
        self.sharpness_history.append(sharpness_epoch)
        alpha = (sharpness_epoch + 1e-12) / (torch.median(torch.tensor(self.sharpness_history)) + 1e-12)
        # Reset sharpness_step
        self.sharpness_step = []
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * alpha 

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]