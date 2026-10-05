"""whole-episode recurrent replay preserves the policy's hidden-state law."""

import torch
from torch import nn
from torch.distributions import Categorical


def distribution(logits, mask):
    if mask.dtype != torch.bool or mask.shape != logits.shape or not mask.any(dim=-1).all():
        raise ValueError("a categorical policy needs a nonempty matching legal mask")
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite policy logits")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class ActorCritic(nn.Module):
    def __init__(self, features, actions=25, hidden=32, memory="gru"):
        super().__init__()
        self.features, self.actions, self.hidden_size, self.memory_kind = features, actions, hidden, memory
        self.encoder = nn.Linear(features, hidden)
        self.memory = nn.GRU(hidden, hidden, batch_first=True) if memory == "gru" else None
        self.actor, self.critic = nn.Linear(hidden, actions), nn.Linear(hidden, 1)
        self.role_head, self.success_head = nn.Linear(hidden, 6), nn.Linear(hidden, 1)

    def forward(self, features, hidden=None):
        encoded = torch.tanh(self.encoder(features))
        if self.memory is not None:
            encoded, hidden = self.memory(encoded.unsqueeze(0), hidden)
            encoded = encoded.squeeze(0)
        return (self.actor(encoded), self.critic(encoded).squeeze(-1),
                self.role_head(encoded), torch.sigmoid(self.success_head(encoded).squeeze(-1)), hidden)


def returns(rewards, gamma):
    result, value = [], 0.0
    for reward in reversed(rewards):
        value = float(reward)+gamma*value
        result.append(value)
    return torch.tensor(result[::-1], dtype=torch.float32)


def advantages(rewards, values, gamma, trace):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_values = torch.cat((values[1:], values.new_zeros(1)))
    deltas = rewards + gamma*next_values-values
    result, value = [], values.new_zeros(())
    for delta in reversed(deltas):
        value = delta + gamma*trace*value
        result.append(value)
    advantage = torch.stack(result[::-1])
    return advantage, advantage+values
