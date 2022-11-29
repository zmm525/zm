import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import yaml
import gym
import time
from pathlib import Path


class CarDynamics(nn.Module):
    def __init__(self, u, p_body, p_tyre):
        """
        u is batch_size * 2, corresponding to (delta, omega).
        """
        self.u = u
        self.p_body = p_body
        self.p_tyre = p_tyre
        super().__init__()

    def compute_extended_state(self, s):
        """
        s is batch_size * 6, corresponding to (x, y, psi, xd, yd, psid).
        """
        # Decoding
        x = s[:, 0]
        y = s[:, 1]
        psi = s[:, 2]
        xd = s[:, 3]
        yd = s[:, 4]
        psid = s[:, 5]
        delta = self.u[:, 0]
        omega = self.u[:, 1]
        lf = self.p_body[:, 0]
        lr = self.p_body[:, 1]
        m = self.p_body[:, 2]
        h = self.p_body[:, 3]
        g = self.p_body[:, 4]
        Iz = self.p_body[:, 5]
        B = self.p_tyre[:, 0]
        C = self.p_tyre[:, 1]
        D = self.p_tyre[:, 2]
        E = self.p_tyre[:, 3]

        v = torch.hypot(xd, yd)
        beta = torch.atan2(yd, xd) - psi
        vfx = v * torch.cos(beta - delta) + psid * lf * torch.sin(delta)
        vfy = v * torch.sin(beta - delta) + psid * lf * torch.cos(delta)
        vrx = v * torch.cos(beta)
        vry = v * torch.sin(beta) - psid * lr
        eps = 1e-3
        sfx = (vfx - omega) / (omega + eps)
        sfy = (vfy) / (omega + eps)
        srx = (vrx - omega) / (omega + eps)
        sry = (vry) / (omega + eps)
        sf = torch.hypot(sfx, sfy)
        sr = torch.hypot(srx, sry)
        pacejka = lambda slip: D * torch.sin(C * torch.atan(B * slip - E * (B * slip - torch.atan(B * slip))))
        muf = pacejka(sf)
        mur = pacejka(sr)
        alphaf = torch.atan2(sfy, sfx)
        alphar = torch.atan2(sry, srx)
        mufx = -torch.cos(alphaf) * muf
        mufy = -torch.sin(alphaf) * muf
        murx = -torch.cos(alphar) * mur
        mury = -torch.sin(alphar) * mur
        G = m * g
        l = lf + lr
        ffz = (lr * G - h * G * murx) / (l + h * (mufx * torch.cos(delta) - mufy * torch.sin(delta) - murx))
        frz = G - ffz
        ffx = mufx * ffz
        ffy = mufy * ffz
        frx = murx * frz
        fry = mury * frz
        
        xdd = 1 / m * (ffx * torch.cos(psi + delta) - ffy * torch.sin(psi + delta) + frx * torch.cos(psi) - fry * torch.sin(psi))
        ydd = 1 / m * (ffx * torch.sin(psi + delta) + ffy * torch.cos(psi + delta) + frx * torch.sin(psi) + fry * torch.cos(psi))
        psidd = 1 / Iz * ((ffy * torch.cos(delta) + ffx * torch.sin(delta)) * lf - fry * lr)

        # cast from shape [batch_size,] to [batch_size, 1]
        xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry = map(lambda t: torch.unsqueeze(t, 1), [xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry])

        return torch.cat([xd, yd, psid, xdd, ydd, psidd, v, beta, vfx, vfy, vrx, vry, sfx, sfy, srx, sry, sf, sr, muf, mur, alphaf, alphar, mufx, mufy, murx, mury, ffz, frz, ffx, ffy, frx, fry], 1)

    def forward(self, t, s):
        es = self.compute_extended_state(s)
        return es[:, :6]


class GPUVectorizedCarEnv:
    def __init__(self, preset_name, n, dt=0.01, solver="euler", device="cuda:0"):
        self.num_states = 6
        self.num_actions = 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = gym.spaces.Box(low=np.array([-0.6, 0.]), high=np.array([0.6, 7.]), shape=(2,))
        self.state_space = self.observation_space

        self.preset_name = preset_name
        self.n = n
        self.dt = dt
        self.solver = solver
        self.device = torch.device(device)
        file_path = os.path.dirname(__file__)
        with open(os.path.join(file_path, "presets.yaml")) as f:
            presets = yaml.safe_load(f)
            params = presets[preset_name]["parameters"]
        self.p_body = torch.zeros((n, 6), device=self.device)
        self.p_body[:, 0] = params["lF"]
        self.p_body[:, 1] = params["lR"]
        self.p_body[:, 2] = params["m"]
        self.p_body[:, 3] = params["h"]
        self.p_body[:, 4] = params["g"]
        self.p_body[:, 5] = params["Iz"]
        self.p_tyre = torch.zeros((n, 4), device=self.device)
        self.p_tyre[:, 0] = params["B"]
        self.p_tyre[:, 1] = params["C"]
        self.p_tyre[:, 2] = params["D"]
        self.p_tyre[:, 3] = params["E"]
        self.s = torch.zeros((n, 6), device=self.device)
        self.dynamics = None
        self.step_count = 0
        self.saved_data = []

    def obs(self):
        return self.s

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def reset(self):
        self.s = torch.zeros((self.n, 6), device=self.device)
        self.u = torch.zeros((self.n, 2), device=self.device)
        self.dynamics = CarDynamics(self.u, self.p_body, self.p_tyre)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count = 0
        self.saved_data = []
        return self.obs()

    def step(self, u):
        self.u = u
        self.dynamics = CarDynamics(u, self.p_body, self.p_tyre)
        self.s = odeint(self.dynamics, self.s, torch.tensor([0., self.dt]), method=self.solver)[1, :, :]
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count += 1
        obs, reward, done, info = self.obs(), self.reward(), self.done(), self.info()
        if torch.all(done) and self.saved_data:
            filename = time.strftime("%Y%m%d-%H%M%S") + ".pth"
            Path("data").mkdir(parents=True, exist_ok=True)
            torch.save(self.saved_data, os.path.join("data", filename))
        return obs, reward, done, info

    def render(self, **kwargs):
        """Save rollout data to emulate rendering."""
        self.saved_data.append((self.s[0, :].cpu(), self.u[0, :].cpu(), self.es[0, :].cpu()))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import time

    # Compare solvers
    env_euler = GPUVectorizedCarEnv("racecar", 1, solver="euler")
    env_rk = GPUVectorizedCarEnv("racecar", 1, solver="dopri5")
    env_rk8 = GPUVectorizedCarEnv("racecar", 1, solver="dopri8")

    traj_euler = [env_euler.reset().cpu().numpy()]
    traj_rk = [env_rk.reset().cpu().numpy()]
    # traj_rk8 = [env_rk8.reset().cpu().numpy()]
    
    for i in range(600):
        if i < 100:
            u = [0., 1.]
        elif 100 <= i < 200:
            u = [0., 4.]
        elif 200 <= i < 300:
            u = [0., 1.]
        elif 300 <= i < 400:
            u = [0.4, 4.]
        else:
            u = [-0.4, 4.]

        s_euler, _, _, _ = env_euler.step(torch.tensor([u], device=torch.device("cuda:0")))
        s_rk, _, _, _ = env_rk.step(torch.tensor([u], device=torch.device("cuda:0")))
        # s_rk8, _, _, _ = env_rk8.step(torch.tensor([u], device=torch.device("cuda:0")))

        traj_euler.append(s_euler.cpu().numpy())
        traj_rk.append(s_rk.cpu().numpy())
        # traj_rk8.append(s_rk8.cpu().numpy())

    plt.figure(dpi=300)
    plt.plot([s[0][0] for s in traj_euler], [s[0][1] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    # plt.plot([s[0][0] for s in traj_rk8], [s[0][1] for s in traj_rk8], label="RK8")
    plt.legend()
    plt.axis("equal")

    # Test large-scale parallelization
    ns = [10 ** i for i in range(7)]
    def measure_time(n, solver):
        env = GPUVectorizedCarEnv("racecar", n, solver=solver)
        u = torch.tensor([[0.1, 10.] for _ in range(n)], device=torch.device("cuda:0"))
        start_time = time.time()
        for i in tqdm(range(1000)):
            env.step(u)
        elapsed = time.time() - start_time
        return elapsed
    times_euler = [measure_time(n, "euler") for n in ns]
    times_rk = [measure_time(n, "dopri5") for n in ns]

    plt.figure(dpi=300)
    plt.loglog(ns, times_euler, label="Euler")
    plt.loglog(ns, times_rk, label="RK5")
    plt.legend()
    plt.xlabel("# of instances")
    plt.ylabel("Time of performing 10s simulation")
