import torch
import torch.nn as nn

import modules


class Model(nn.Module):
    def __init__(
        self,
        dict_len,
        question_dim,
        vision_dim,
        rnn_input_dim,
        rnn_hidden_size,
        num_mclasses,
        num_aclasses,
        device,
    ):
        super().__init__()
        self.device = device

        self.question_net = modules.QuestionNetwork(dict_len, question_dim, self.device)
        self.vision_net = modules.VisionNetwork(vision_dim, self.device)
        self.rnn = modules.RNNNetwork(rnn_input_dim, rnn_hidden_size)
        self.movement_net = modules.MovementNetwork(rnn_hidden_size, num_mclasses)
        self.baseliner = modules.BaselineNetwork(rnn_hidden_size, 1)
        self.prediction_net = modules.PredictionNetwork(rnn_hidden_size, num_aclasses)

    def forward(self, question, rgb_image, h_t_prev, last=False):
        q_t = self.question_net(question)
        g_t = self.vision_net(rgb_image)
        cat_t = torch.cat((q_t, g_t), 1)  # concatenate
        h_t = self.rnn(cat_t, h_t_prev)
        m_t = self.movement_net(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            p_t = self.prediction_net(h_t)
            return p_t

        return h_t, m_t, b_t
