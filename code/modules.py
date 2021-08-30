import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionNetwork(nn.Module):
    def __init__(self, module_dim, device):
        super().__init__()

        self.module_dim = module_dim
        self.device = device

        self.stem = nn.Sequential(
            nn.Dropout(p=0.18),
            nn.Conv2d(
                128, self.module_dim, kernel_size=3, stride=1, padding=1
            ),  # the input dimension 512 is for first 2 layers of ResNet # 128 is for first 2 layers of ResNet18
            nn.ELU(),
            nn.Dropout(p=0.18),
            nn.Conv2d(
                self.module_dim, self.module_dim, kernel_size=3, stride=1, padding=1
            ),
            nn.ELU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

    def forward(self, image_feature):
        g_t = self.stem(image_feature)
        g_t = self.avgpool(g_t)
        g_t = g_t.view(g_t.shape[0], g_t.shape[1])
        return g_t


class QuestionNetwork(nn.Module):
    def __init__(self, dict_len, question_dim, device):
        super().__init__()
        self.device = device
        self.weight = torch.rand(
            dict_len, question_dim, dtype=torch.float32, requires_grad=True
        ).to(self.device)
        self.embedding = nn.Embedding.from_pretrained(self.weight, freeze=False)

    def forward(self, question):
        q_t = self.embedding(question)
        return q_t

    # pass


class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class MovementNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        feat = F.relu(
            self.fc(h_t.detach())
        )  # using detach() makes the loss of RL is not able to be backpropogated to update RNNNetwork, QuestionNetwork, and VisionNetwork.
        # m_t = F.log_softmax(self.fc_lt(feat), dim=1)
        m_t = F.softmax(self.fc_lt(feat), dim=1)
        return m_t


class PredictionNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        p_t = F.log_softmax(self.fc(h_t), dim=1)
        return p_t


class BaselineNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t
