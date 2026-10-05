"""the auditor differentiates the actual finite sgd update, not exact shaping."""

import torch
from torch import nn
from torch.func import functional_call

from .models import advantages, distribution, returns
from .rollouts import collect, episode_seed


def ppo_update(model, optimizer, trajectories, config, trader=False):
    prepared = []
    for trajectory in trajectories:
        if trajectory.guarded:
            raise ValueError("evaluation-only fallback trajectories are not ppo behavior samples")
        old_values = torch.tensor(trajectory.values)
        advantage, target = advantages(trajectory.rewards, old_values, config.gamma, config.gae_lambda)
        prepared.append((trajectory, advantage, target))
    all_advantages = torch.cat([item[1] for item in prepared])
    mean, scale = all_advantages.mean(), all_advantages.std(unbiased=False).clamp_min(1e-6)
    diagnostics = {}
    for _ in range(config.epochs):
        losses, kls, fractions = [], [], []
        for trajectory, advantage, target in prepared:
            features, masks, actions = trajectory.tensors()
            logits, values, role, success, _ = model(features)
            dist = distribution(logits, masks)
            logp = dist.log_prob(actions)
            old = torch.tensor(trajectory.logp)
            ratio = (logp-old).exp()
            normalized = (advantage-mean)/scale
            weights = config.gamma ** torch.arange(len(actions), dtype=torch.float32)
            surrogate = torch.minimum(ratio*normalized, ratio.clamp(1-config.clip, 1+config.clip)*normalized)
            loss = -(weights*surrogate).mean() + 0.5*(values-target).square().mean()-config.entropy*dist.entropy().mean()
            if not trader:
                if config.role_weight:
                    labels = torch.full((len(actions),), trajectory.role_label, dtype=torch.long)
                    loss += config.role_weight*nn.functional.cross_entropy(role, labels)
                if config.success_weight:
                    loss += config.success_weight*(success-float(trajectory.successful)).square().mean()
            losses.append(loss)
            kls.append(((ratio-1)-(logp-old)).mean().detach())
            fractions.append(((ratio-1).abs() > config.clip).float().mean().detach())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite ppo loss")
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite ppo gradient")
        optimizer.step()
        diagnostics = {"loss": float(loss.detach()), "gradient_norm": float(norm),
                       "approx_kl": float(torch.stack(kls).mean()),
                       "clip_fraction": float(torch.stack(fractions).mean())}
    return diagnostics


def td_loss(model, trajectories, config, gate=None):
    losses = []
    for trajectory in trajectories:
        if trajectory.guarded:
            raise ValueError("evaluation-only fallback trajectories cannot train the base actor")
        features, masks, actions = trajectory.tensors()
        logits, values, role, success, _ = model(features)
        dist = distribution(logits, masks)
        next_values = torch.cat((values[1:].detach(), values.new_zeros(1)))
        delta = torch.tensor(trajectory.rewards)+config.gamma*next_values-values.detach()
        if gate is not None:
            potential = 2*torch.tensor(trajectory.success_predictions)-1
            next_potential = torch.cat((potential[1:], potential.new_zeros(1)))
            coefficient = torch.sigmoid(gate) if config.audit_gate else config.fixed_gate
            delta = delta+config.shaping_alpha*coefficient*(config.gamma*next_potential-potential)
        weights = config.gamma ** torch.arange(len(actions), dtype=torch.float32)
        loss = -(weights*dist.log_prob(actions)*delta).mean()
        target = returns(trajectory.rewards, config.gamma)
        loss += 0.5*(values-target).square().mean()-config.entropy*dist.entropy().mean()
        if gate is not None:
            loss += 0.1*(success-float(trajectory.successful)).square().mean()
        elif config.success_weight:
            loss += config.success_weight*(success-float(trajectory.successful)).square().mean()
        if config.role_weight:
            labels = torch.full((len(actions),), trajectory.role_label, dtype=torch.long)
            loss += config.role_weight*nn.functional.cross_entropy(role, labels)
        losses.append(loss)
    return torch.stack(losses).mean()


def virtual_sgd(model, loss, learning_rate, create_graph=False):
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(named.values()), create_graph=create_graph, allow_unused=True)
    norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients if gradient is not None)+1e-12)
    scale = (0.5/norm).clamp(max=1.0)
    updated = {name: value if gradient is None else value-learning_rate*scale*gradient
               for (name, value), gradient in zip(named.items(), gradients)}
    return updated, norm


def td_update(model, trajectories, config, gate=None, scenario=None, budget=None, outer_index=0):
    loss = td_loss(model, trajectories, config, gate)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite td loss")
    audited = gate is not None and config.audit_gate
    updated, norm = virtual_sgd(model, loss, config.learning_rate, create_graph=audited)
    diagnostics = {"loss": float(loss.detach()), "gradient_norm": float(norm.detach())}
    if audited:
        outer_losses = []
        outer_actions, outer_market = 0, 0
        for index in range(config.outer_episodes):
            trajectory, _, details = collect(scenario, budget, model,
                episode_seed(config.seed, 2, outer_index+index), config.exact_resources, parameters=updated)
            outer_actions += details["actions"]
            outer_market += details["market_updates"]
            features, masks, actions = trajectory.tensors()
            logits, _, _, _, _ = functional_call(model, updated, (features,))
            logp = distribution(logits, masks).log_prob(actions)
            target = returns(trajectory.rewards, config.gamma)
            weights = config.gamma ** torch.arange(len(actions), dtype=torch.float32)
            outer_losses.append(-(weights*logp*target).sum())
        outer_loss = torch.stack(outer_losses).mean()+0.001*torch.sigmoid(gate).square()
        gradient = torch.autograd.grad(outer_loss, gate)[0]
        if not torch.isfinite(gradient):
            raise FloatingPointError("non-finite meta gradient")
        diagnostics.update(gate_before=float(torch.sigmoid(gate).detach()),
                           gate_gradient=float(gradient.detach()), outer_loss=float(outer_loss.detach()),
                           outer_actions=outer_actions, outer_market_updates=outer_market)
        with torch.no_grad():
            gate.sub_(config.gate_lr*gradient).clamp_(-6, 6)
        diagnostics["gate_after"] = float(torch.sigmoid(gate).detach())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(updated[name])
    return diagnostics
