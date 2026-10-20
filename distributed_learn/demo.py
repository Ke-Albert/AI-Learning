from accelerate import Accelerator,DeepSpeedPlugin
import torch

from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    accelerator = Accelerator()

    model=SimpleNet()
    device= accelerator.device
    model.to(device)
    model,optimizer,train_dataloader,scheduler= accelerator.prepare(
        model,torch.optim.Adam(model.parameters()),
        DataLoader(TensorDataset(torch.randn(1000,10), torch.randn(1000,1)),
        batch_size=32, shuffle=True),torch.optim.lr_scheduler.StepLR(torch.optim.Adam(model.parameters()), step_size=10, gamma=0.1)
        )

    model.train()
    for epoch in range(100):
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")