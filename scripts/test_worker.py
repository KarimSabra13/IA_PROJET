import numpy as np
from main.rl_env import InverterEnv

def main():
    env = InverterEnv(max_steps=1, timeout_s=10.0, restart_every=25, restart_on_reset=True)
    rng = np.random.default_rng(0)

    obs, info = env.reset()
    for i in range(200):
        a = np.array([rng.uniform(0.24, 5.0), rng.uniform(0.48, 10.0)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            obs, info = env.reset()

    print("OK: 200 episodes")
    env.close()

if __name__ == "__main__":
    main()
