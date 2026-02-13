# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 4 neurons each, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

<img width="899" height="363" alt="image" src="https://github.com/user-attachments/assets/b31f7100-3068-4a63-a05c-e2a467275fa0" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: KIRUTHIGA.B
### Register Number: 212224040160
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Dataset
dataset1 = pd.read_csv('ex1.csv')
print(dataset1.head(10))
X = dataset1[['x']].values
y = dataset1[['y']].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)
# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # Removed ReLU from the final output layer
        return x

# Initialize Model, Loss and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Training Function
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss:.6f}")

# Train the Model
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

# Test Evaluation
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")
print(" NAME: KIRUTHIGA.B")
print("REG.NO: 212224040160")
# Plot Loss
loss_df = pd.DataFrame(ai_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()

# New Sample Prediction
X_new = torch.tensor([[9]], dtype=torch.float32)
X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)

prediction = ai_brain(X_new_scaled).item()
print(f"Predicted Spending Score: {prediction}")

```
## Dataset Information

<img width="312" height="400" alt="image" src="https://github.com/user-attachments/assets/676defba-8eff-43e9-b64c-a07006ca1f02" />


## OUTPUT

<img width="386" height="528" alt="image" src="https://github.com/user-attachments/assets/5cdaa431-fd5d-4326-9fa4-04b534015dba" />



### Training Loss Vs Iteration Plot

<img width="771" height="605" alt="image" src="https://github.com/user-attachments/assets/e6fe77f0-81bb-4d54-8f6d-d5a41fc36538" />



### New Sample Data Prediction

<img width="533" height="76" alt="image" src="https://github.com/user-attachments/assets/c1c70060-40b2-40db-ae9c-6af96f47a8a1" />


## RESULT

Thus, a neural network regression model for the given dataset is developed successfully.
