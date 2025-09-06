#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Differentiable Parameter Emulation (DPE) for SUMMA — Config-driven multipath

Supports three gradient paths selectable via config.yaml:
  EMULATOR_SETTING:
    - "EMULATOR"          : train NN emulator; optimize via backprop through NN only
    - "FD"                : optimize parameters using finite-difference gradients on SUMMA
    - "SUMMA_AUTODIFF"    : end-to-end autograd using SUMMA sensitivities (Sundials) when available,
                             with optional small NN head on top of objectives
    - "SUMMA_AUTODIFF_FD" : same autograd wrapper but forcing FD Jacobian (no Sundials)

Additional config keys (with reasonable defaults shown where applicable):
  DPE_HIDDEN_DIMS: [256, 128, 64]
  DPE_TRAINING_SAMPLES: 500
  DPE_VALIDATION_SAMPLES: 100
  DPE_EPOCHS: 300
  DPE_LEARNING_RATE: 1e-3
  DPE_OPTIMIZATION_LR: 1e-2
  DPE_OPTIMIZATION_STEPS: 200

  # AUTODIFF + FD controls
  DPE_USE_NN_HEAD: true
  DPE_USE_SUNDIALS: true   # ignored for SUMMA_AUTODIFF_FD
  DPE_AUTODIFF_STEPS: 200
  DPE_AUTODIFF_LR: 1e-2
  DPE_FD_STEP: 1e-3
  DPE_GD_STEP_SIZE: 1e-1   # for explicit FD gradient descent

"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Reuse your iterative optimizer backend for SUMMA IO/parallelism
from iterative_optimizer import DEOptimizer  # type: ignore


# ------------------------------
# Config + Emulator definition
# ------------------------------
@dataclass
class EmulatorConfig:\n    hidden_dims: List[int] = None
    dropout: float = 0.1
    activation: str = 'relu'
    n_training_samples: int = 1000
    n_validation_samples: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 500
    early_stopping_patience: int = 50
    optimization_lr: float = 1e-2
    optimization_steps: int = 100
    objective_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64, 32]
        if self.objective_weights is None:
            # Maximize KGE/NSE: use negative weights to minimize their negatives
            self.objective_weights = {'KGE': -1.0, 'NSE': -1.0, 'RMSE': 0.1, 'PBIAS': 0.05}


class ParameterEmulator(nn.Module):
    def __init__(self, n_parameters: int, n_objectives: int, config: EmulatorConfig):
        super().__init__()
        layers = []
        input_dim = n_parameters
        for hidden in config.hidden_dims:
            layers += [nn.Linear(input_dim, hidden),
                       nn.SiLU() if config.activation == 'swish' else nn.ReLU(),
                       nn.Dropout(config.dropout),
                       nn.BatchNorm1d(hidden)]
            input_dim = hidden
        layers += [nn.Linear(input_dim, n_objectives)]
        self.network = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ------------------------------
# SUMMA interface via iterative backend
# ------------------------------
class SummaInterface:
    def __init__(self, confluence_config: Dict, logger: logging.Logger):
        self.config = confluence_config
        self.logger = logger
        self.backend = DEOptimizer(confluence_config, logger)
        self.param_manager = self.backend.parameter_manager
        self.model_executor = self.backend.model_executor
        self.calib_target = self.backend.calibration_target
        self.param_names = self.param_manager.all_param_names

    def normalize_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        return self.param_manager.normalize_parameters(params)

    def denormalize_parameters(self, x_norm: np.ndarray) -> Dict[str, np.ndarray]:
        return self.param_manager.denormalize_parameters(x_norm)

    def run_simulations_batch(self, param_samples: List[np.ndarray]):
        tasks = []
        for i, x in enumerate(param_samples):
            params = self.denormalize_parameters(x)
            tasks.append({
                'individual_id': i,
                'params': params,
                'proc_id': i % self.backend.num_processes,
                'evaluation_id': f"dpe_sample_{i:04d}",
            })
        results = self.backend._run_parallel_evaluations(tasks)
        out = [None] * len(param_samples)
        for r in results:
            if r and r.get('metrics') is not None:
                out[r['individual_id']] = r['metrics']
        return out


# ------------------------------
# Autograd op to bridge SUMMA into PyTorch
# ------------------------------
class SummaAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_norm: torch.Tensor, dpe_self, use_sens: bool, fd_step: float):
        x_np = x_norm.detach().cpu().numpy().astype(float)
        objs = dpe_self._evaluate_objectives_real(x_np)
        obj_vec = torch.tensor([objs.get(k, objs.get(f"Calib_{k}", 0.0)) for k in dpe_self.objective_names],
                               dtype=torch.float32, device=x_norm.device)
        if use_sens:
            J_np = dpe_self._summa_objective_jacobian(x_np)  # user must wire to Sundials sensitivities
            J = torch.tensor(J_np, dtype=torch.float32, device=x_norm.device)
            ctx.save_for_backward(J)
            ctx.jac_ready = True
        else:
            ctx.save_for_backward(x_norm.detach().clone())
            ctx.jac_ready = False
            ctx.fd_step = float(fd_step)
            ctx.dpe_ref = dpe_self
        return obj_vec

    @staticmethod
    def backward(ctx, grad_out):
        if getattr(ctx, 'jac_ready', False):
            (J,) = ctx.saved_tensors  # [n_obj, n_params]
        else:
            (x_saved,) = ctx.saved_tensors
            dpe = ctx.dpe_ref
            J_np = dpe._finite_difference_jacobian(x_saved.detach().cpu().numpy(), step=ctx.fd_step)
            J = torch.tensor(J_np, dtype=torch.float32, device=grad_out.device)
        grad_x = grad_out.unsqueeze(0).matmul(J).squeeze(0)
        return grad_x, None, None, None


class ObjectiveHead(nn.Module):
    def __init__(self, n_params: int, n_objs: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_objs + n_params, hidden), nn.SiLU(), nn.Linear(hidden, n_objs)
        )
    def forward(self, obj, x_norm):
        return obj + self.net(torch.cat([obj, x_norm], dim=-1))


# ------------------------------
# Main Optimizer
# ------------------------------
class DifferentiableParameterOptimizer:
    def __init__(self, confluence_config: Dict, domain_name: str, emulator_config: EmulatorConfig = None):
        self.confluence_config = confluence_config
        self.domain_name = domain_name
        self.emulator_config = emulator_config or EmulatorConfig()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"{__name__}.DPE")
        self.summa = SummaInterface(confluence_config, self.logger)
        self.param_names = self.summa.param_names
        self.objective_names = sorted(self.emulator_config.objective_weights.keys())
        self.n_parameters = len(self.param_names)
        self.n_objectives = len(self.objective_names)
        self.emulator = ParameterEmulator(self.n_parameters, self.n_objectives, self.emulator_config)
        self.training_data = {'parameters': [], 'objectives': []}
        self.validation_data = {'parameters': [], 'objectives': []}

    # ===== Shared utilities =====
    def _latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=self.n_parameters).random(n=n_samples)

    def _is_valid_objective(self, obj: Dict[str, float]) -> bool:
        for k in self.objective_names:
            v = obj.get(k, obj.get(f"Calib_{k}"))
            if v is None or np.isnan(v) or np.isinf(v):
                return False
        return True

    def _composite_loss_from_objectives(self, obj: Dict[str, float]) -> float:
        total = 0.0
        for k in self.objective_names:
            v = obj.get(k, obj.get(f"Calib_{k}"))
            if v is None or np.isnan(v) or np.isinf(v):
                return 1e6
            total += self.emulator_config.objective_weights.get(k, 0.0) * float(v)
        return float(total)

    def _evaluate_objectives_real(self, x_norm: np.ndarray) -> Dict[str, float]:
        params = self.summa.denormalize_parameters(x_norm)
        task = {'individual_id': 0, 'params': params, 'proc_id': 0, 'evaluation_id': 'dpe_eval_single'}
        results = self.summa.backend._run_parallel_evaluations([task])
        return results[0].get('metrics', {}) if results else {}

    # ===== Data gen + training =====
    def generate_training_data(self):
        total = self.emulator_config.n_training_samples + self.emulator_config.n_validation_samples
        X = self._latin_hypercube_sampling(total)
        Y_dicts = self.summa.run_simulations_batch(X)
        Xv, Yv = [], []
        for x, y in zip(X, Y_dicts):
            if y and self._is_valid_objective(y):
                Xv.append(x)
                Yv.append([y.get(k, y.get(f"Calib_{k}", 0.0)) for k in self.objective_names])
        if len(Xv) < self.emulator_config.n_training_samples:
            raise RuntimeError("Insufficient valid samples for training")
        ntr = self.emulator_config.n_training_samples
        self.training_data = {'parameters': Xv[:ntr], 'objectives': Yv[:ntr]}
        self.validation_data = {'parameters': Xv[ntr:], 'objectives': Yv[ntr:]}

    def train_emulator(self):
        if not self.training_data['parameters']:
            raise ValueError("No training data. Call generate_training_data() first.")
        Xtr = torch.tensor(np.array(self.training_data['parameters']), dtype=torch.float32)
        Ytr = torch.tensor(np.array(self.training_data['objectives']), dtype=torch.float32)
        Xva = torch.tensor(np.array(self.validation_data['parameters']), dtype=torch.float32)
        Yva = torch.tensor(np.array(self.validation_data['objectives']), dtype=torch.float32)
        opt = optim.Adam(self.emulator.parameters(), lr=self.emulator_config.learning_rate)
        crit = nn.MSELoss()
        best = float('inf'); patience = 0
        for ep in range(self.emulator_config.n_epochs):
            self.emulator.train()
            perm = torch.randperm(Xtr.shape[0])
            for i in range(0, Xtr.shape[0], self.emulator_config.batch_size):
                idx = perm[i:i+self.emulator_config.batch_size]
                xb, yb = Xtr[idx], Ytr[idx]
                opt.zero_grad(); pred = self.emulator(xb); loss = crit(pred, yb); loss.backward(); opt.step()
            self.emulator.eval()
            with torch.no_grad():
                vloss = crit(self.emulator(Xva), Yva).item()
            if vloss < best:
                best = vloss; patience = 0
                torch.save(self.emulator.state_dict(), f"best_emulator_{self.domain_name}.pt")
            else:
                patience += 1
            if patience >= self.emulator_config.early_stopping_patience:
                break
        self.emulator.load_state_dict(torch.load(f"best_emulator_{self.domain_name}.pt"))

    # ===== Optimizers =====
    def _compute_weighted_loss_tensor(self, obj_tensor: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32, device=obj_tensor.device)
        for i, k in enumerate(self.objective_names):
            w = self.emulator_config.objective_weights.get(k, 0.0)
            loss = loss + w * obj_tensor[i]
        return loss

    def optimize_parameters(self, initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        # emulator-only backprop
        if initial_params:
            x0 = self.summa.normalize_parameters(initial_params)
        else:
            x0 = np.full(self.n_parameters, 0.5)
        x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        opt = optim.Adam([x], lr=self.emulator_config.optimization_lr)
        best = float('inf'); best_x = x.detach().clone()
        self.emulator.eval()
        for t in range(self.emulator_config.optimization_steps):
            opt.zero_grad()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
            obj = self.emulator(x.unsqueeze(0)).squeeze(0)
            loss = self._compute_weighted_loss_tensor(obj)
            loss.backward(); opt.step()
            if loss.item() < best:
                best, best_x = loss.item(), x.detach().clone()
        return self.summa.denormalize_parameters(best_x.clamp(0.0,1.0).numpy())

    # ---- Finite-difference gradient descent on real SUMMA ----
    def _finite_difference_jacobian(self, x: np.ndarray, step: float = 1e-3) -> np.ndarray:
        n, m = self.n_parameters, self.n_objectives
        J = np.zeros((m, n), dtype=float)
        base = self._evaluate_objectives_real(x)
        base_vec = np.array([base.get(k, base.get(f"Calib_{k}", 0.0)) for k in self.objective_names])
        plus, minus, kinds = [], [], []
        for i in range(n):
            xp, xm = x.copy(), x.copy()
            if x[i]+step <= 1 and x[i]-step >= 0:
                xp[i]+=step; xm[i]-=step; plus.append(xp); minus.append(xm); kinds.append('central')
            elif x[i]+step <= 1:
                xp[i]+=step; plus.append(xp); minus.append(None); kinds.append('forward')
            elif x[i]-step >= 0:
                xm[i]-=step; plus.append(None); minus.append(xm); kinds.append('backward')
            else:
                plus.append(None); minus.append(None); kinds.append('zero')
        batch, mapidx = [], []
        for i, (xp, xm) in enumerate(zip(plus, minus)):
            if xp is not None: batch.append(xp); mapidx.append(('p', i))
            if xm is not None: batch.append(xm); mapidx.append(('m', i))
        mets = self.summa.run_simulations_batch(batch) if batch else []
        vecs = []
        for met in mets:
            if met is None:
                vecs.append(np.full(m, 1e6))
            else:
                vecs.append(np.array([met.get(k, met.get(f"Calib_{k}", 0.0)) for k in self.objective_names]))
        ptr = 0; pvec = [None]*n; mvec = [None]*n
        for tag, i in mapidx:
            if tag == 'p': pvec[i] = vecs[ptr]; ptr += 1
            else: mvec[i] = vecs[ptr]; ptr += 1
        for i, kd in enumerate(kinds):
            if kd=='central' and pvec[i] is not None and mvec[i] is not None:
                J[:, i] = (pvec[i] - mvec[i]) / (2*step)
            elif kd=='forward' and pvec[i] is not None:
                J[:, i] = (pvec[i] - base_vec) / step
            elif kd=='backward' and mvec[i] is not None:
                J[:, i] = (base_vec - mvec[i]) / step
            else:
                J[:, i] = 0.0
        return J

    def optimize_with_fd(self, initial_params: Optional[Dict[str, float]] = None,
                         step_size: float = 1e-1, fd_step: float = 1e-3,
                         max_iters: int = 50, tol: float = 1e-6) -> Dict[str, float]:
        x = self.summa.normalize_parameters(initial_params) if initial_params else np.full(self.n_parameters, 0.5)
        base = self._evaluate_objectives_real(x); L = self._composite_loss_from_objectives(base)
        for t in range(1, max_iters+1):
            G = self._finite_difference_jacobian(x, step=fd_step)  # [m,n]
            # chain rule for scalar loss: dL/dx = w^T * J
            w = np.array([self.emulator_config.objective_weights.get(k, 0.0) for k in self.objective_names])
            g = w @ G  # [n]
            x_new = np.clip(x - step_size * g, 0.0, 1.0)
            met_new = self._evaluate_objectives_real(x_new); L_new = self._composite_loss_from_objectives(met_new)
            if L_new < L:
                x, L = x_new, L_new
            else:
                step_size *= 0.5
                if step_size < 1e-5:
                    break
            if np.linalg.norm(g) < tol:
                break
        return self.summa.denormalize_parameters(x)

    # ---- SUMMA autograd path (Sundials or FD fallback) ----
    def _summa_objective_jacobian(self, x_norm: np.ndarray) -> np.ndarray:
        """Wire this to your Sundials-enabled SUMMA sensitivity outputs.
        Must return shape [n_objectives, n_parameters] wrt normalized params.
        Default raises to force explicit wiring.
        """
        raise NotImplementedError("Implement Sundials sensitivity plumbing here.")

    def optimize_with_summa_autodiff(self, use_sundials: bool = True, fd_step: float = 1e-3,
                                     steps: int = 200, lr: float = 1e-2, use_nn_head: bool = True,
                                     initial_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if initial_params:
            x0 = self.summa.normalize_parameters(initial_params)
        else:
            x0 = np.full(self.n_parameters, 0.5)
        x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)
        head = ObjectiveHead(self.n_parameters, self.n_objectives).to(device) if use_nn_head else None
        opt_params = [x] + (list(head.parameters()) if head else [])
        opt = optim.Adam(opt_params, lr=lr)
        for t in range(1, steps+1):
            opt.zero_grad()
            obj_summa = SummaAutogradOp.apply(x, self, use_sundials, float(fd_step))
            obj_adj = head(obj_summa, x) if head else obj_summa
            loss = self._compute_weighted_loss_tensor(obj_adj)
            loss.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
        return self.summa.denormalize_parameters(x.detach().cpu().numpy())

    # ===== Orchestration from config =====
    def run_from_config(self):
        mode = self.confluence_config.get('EMULATOR_SETTING', 'EMULATOR').upper()
        # Common knobs
        fd_step = float(self.confluence_config.get('DPE_FD_STEP', 1e-3))
        gd_step = float(self.confluence_config.get('DPE_GD_STEP_SIZE', 1e-1))
        ad_steps = int(self.confluence_config.get('DPE_AUTODIFF_STEPS', 200))
        ad_lr = float(self.confluence_config.get('DPE_AUTODIFF_LR', 1e-2))
        use_head = bool(self.confluence_config.get('DPE_USE_NN_HEAD', True))
        use_sens = bool(self.confluence_config.get('DPE_USE_SUNDIALS', True))

        if mode == 'EMULATOR':
            cache_path = Path(self.confluence_config.get("DPE_TRAINING_CACHE",
                                                        f"training_data_{self.domain_name}.json"))
            self.generate_training_data(cache_path=cache_path, force=False)
            self.train_emulator()
            return self.optimize_parameters()
        elif mode == 'FD':
            params = self.optimize_with_fd(step_size=gd_step, fd_step=fd_step, max_iters=self.emulator_config.optimization_steps)
            return params
        elif mode == 'SUMMA_AUTODIFF':
            params = self.optimize_with_summa_autodiff(use_sundials=use_sens, fd_step=fd_step,
                                                       steps=ad_steps, lr=ad_lr, use_nn_head=use_head)
            return params
        elif mode == 'SUMMA_AUTODIFF_FD':
            params = self.optimize_with_summa_autodiff(use_sundials=False, fd_step=fd_step,
                                                       steps=ad_steps, lr=ad_lr, use_nn_head=use_head)
            return params
        else:
            raise ValueError(f"Unknown EMULATOR_SETTING: {mode}")

    # ===== Validation + Save =====
    def validate_optimization(self, optimized_params: Dict[str, float]) -> Dict[str, float]:
        task = {'individual_id': 0, 'params': optimized_params, 'proc_id': 0, 'evaluation_id': 'dpe_final_validation'}
        results = self.summa.backend._run_parallel_evaluations([task])
        actual = results[0].get('metrics') if results else {}
        return actual or {}

    def save_results(self, optimized_params: Dict[str, float], results_dir: Path):
        results_dir.mkdir(parents=True, exist_ok=True)
        params_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in optimized_params.items()}
        (results_dir / 'optimized_parameters.json').write_text(json.dumps(params_to_save, indent=2))
        torch.save(self.emulator.state_dict(), results_dir / 'emulator_model.pt')
        payload = {
            'parameter_names': self.param_names,
            'objective_names': self.objective_names,
            'training_data_sizes': {k: len(v) for k, v in {'train': self.training_data.get('parameters', []),
                                                           'val': self.validation_data.get('parameters', [])}.items()}
        }
        (results_dir / 'training_meta.json').write_text(json.dumps(payload, indent=2))


# ------------------------------
# CLI entry (optional)
# ------------------------------

def main():
    cfg_path = Path('config.yaml')
    if not cfg_path.exists():
        print('Error: config.yaml not found.')
        return
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    domain = cfg.get('DOMAIN_NAME', 'test_domain')
    emu_cfg = EmulatorConfig(
        hidden_dims=cfg.get('DPE_HIDDEN_DIMS', [256, 128, 64]),
        n_training_samples=cfg.get('DPE_TRAINING_SAMPLES', 500),
        n_validation_samples=cfg.get('DPE_VALIDATION_SAMPLES', 100),
        n_epochs=cfg.get('DPE_EPOCHS', 300),
        learning_rate=cfg.get('DPE_LEARNING_RATE', 1e-3),
        optimization_lr=cfg.get('DPE_OPTIMIZATION_LR', 1e-2),
        optimization_steps=cfg.get('DPE_OPTIMIZATION_STEPS', 200),
    )
    dpe = DifferentiableParameterOptimizer(cfg, domain, emu_cfg)
    params = dpe.run_from_config()
    val = dpe.validate_optimization(params)
    outdir = Path(f"results_differentiable_{domain}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    dpe.save_results(params, outdir)
    print("\n✅ DPE completed.")
    print(json.dumps({'optimized_parameters': params, 'validation': val}, indent=2))


if __name__ == '__main__':
    main()
