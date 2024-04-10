import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(device):
    # Generate synthetic data
    input_data = torch.randn(1000, 10)
    target = torch.randn(1000, 1)

    # Instantiate the model
    model = SimpleNN()

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model_move_start = time.time()
    # Move model and data to device
    model.to(device)
    print(f'Model Move Time: {time.time() - model_move_start}')
    input_data = input_data.to(device)
    target = target.to(device)

    # Train the model
    start_time = time.time()
    epochs = 5000
    print(f'Epochs: {epochs}')
    for epoch in range(epochs):
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    end_time = time.time()
    print(f"Training time on {device}: {end_time - start_time} seconds")

# Train with CPU
print("Training with CPU:")
train_model(torch.device('cpu'))

# Train with CUDA if available
if torch.cuda.is_available():
    print("\nTraining with CUDA:")
    train_model(torch.device('cuda'))
else:
    print("\nCUDA is not available, skipping training with CUDA.")
