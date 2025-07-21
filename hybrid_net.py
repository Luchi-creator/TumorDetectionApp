
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer

from tqdm import tqdm

# ----------------------------- 1. Circuit Cuantic -----------------------------
class QuanvCircuit:
    def __init__(self, kernel_size, backend, shots=10, threshold=0.5):
        self.n_qubits = kernel_size ** 2
        self.theta = [Parameter(f'theta{i}') for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit.h(range(self.n_qubits))
        self.circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        data = data.flatten()
        param_dict = {self.theta[i]: np.pi if data[i] > self.threshold else 0.0 for i in range(self.n_qubits)}
        bound_circuit = self.circuit.assign_parameters(param_dict, inplace=False)
        transpiled = transpile(bound_circuit, self.backend)
        job = self.backend.run(transpiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        total = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return total / (self.shots * self.n_qubits)

# -------------------------- 2. Funcție Forward Quanv --------------------------
class QuanvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, quantum_circuits, kernel_size):
        batch_size = inputs.shape[0]
        length_x = inputs.shape[2] - kernel_size + 1
        length_y = inputs.shape[3] - kernel_size + 1
        outputs = torch.zeros(batch_size, len(quantum_circuits), length_x, length_y).to(inputs.device)

        for i in range(batch_size):
            for c, circuit in enumerate(quantum_circuits):
                for x in range(length_x):
                    for y in range(length_y):
                        patch = inputs[i, 0, x:x+kernel_size, y:y+kernel_size]
                        outputs[i, c, x, y] = circuit.run(patch.cpu().detach().numpy())

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

# ---------------------------- 3. Stratul Cuantic ------------------------------
class QuanvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(QuanvLayer, self).__init__()
        backend = Aer.get_backend('aer_simulator')
        self.quantum_circuits = [QuanvCircuit(kernel_size, backend) for _ in range(out_channels)]
        self.kernel_size = kernel_size

    def forward(self, x):
        return QuanvFunction.apply(x, self.quantum_circuits, self.kernel_size)


class HybridNet(nn.Module):
    def __init__(self):
        super(HybridNet, self).__init__()
        self.quanv = QuanvLayer(1, 1, kernel_size=3)
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)

        # Init temporar pentru a determina automat dimensiunea FC
        self._dummy_input = torch.zeros(1, 1, 28, 28)  # same shape as your dataset images
        with torch.no_grad():
            x = self.forward_features(self._dummy_input)
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward_features(self, x):
        x = self.quanv(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

