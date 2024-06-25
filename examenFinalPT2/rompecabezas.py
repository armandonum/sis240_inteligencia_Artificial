import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

class SaveImageCallback(BaseCallback):
    """
    Callback para guardar una imagen PNG del progreso del entrenamiento.
    """
    def __init__(self, log_dir, eval_freq=1000):
        super().__init__()
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.rewards)
            self.save_png(mean_reward)
        return True

    def _on_rollout_end(self) -> None:
        rewards = self.locals['rewards']
        self.rewards.extend(rewards)

    def save_png(self, mean_reward):
        x = np.arange(0, len(self.rewards) * self.eval_freq, self.eval_freq)
        y = self.rewards[:len(x)]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o')
        plt.xlabel('Steps')
        plt.ylabel('Mean Reward')
        plt.title('Training Progress')
        plt.grid(True)

        plt.savefig(os.path.join(self.log_dir, f'training_progress_{self.num_timesteps}.png'))
        plt.close()

class PuzzleEnv(gym.Env):
    def __init__(self):
        super(PuzzleEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 4 acciones: arriba, abajo, izquierda, derecha
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 5), dtype=float)  # espacio de observación 2D
        self.state = np.zeros((4, 5), dtype=float)
        self.agent_pos = [0, 0]  # posición inicial del agente
        self.goal_pos = [3, 4]   # posición objetivo

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((4, 5), dtype=float)
        self.agent_pos = [0, 0]
        self.state[self.agent_pos[0], self.agent_pos[1]] = 0.5
        self.state[self.goal_pos[0], self.goal_pos[1]] = 1
        return self.state, {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0:  # mover arriba
            x -= 1
        elif action == 1 and x < 3:  # mover abajo
            x += 1
        elif action == 2 and y > 0:  # mover izquierda
            y -= 1
        elif action == 3 and y < 4:  # mover derecha
            y += 1

        self.agent_pos = [x, y]

        self.state = np.zeros((4, 5), dtype=float)
        self.state[self.agent_pos[0], self.agent_pos[1]] = 0.5
        self.state[self.goal_pos[0], self.goal_pos[1]] = 1

        reward = 1 if self.agent_pos == self.goal_pos else -0.1
        terminated = self.agent_pos == self.goal_pos
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):
        for row in self.state:
            print(" ".join(map(str, row)))
        print()

if __name__ == "__main__":
    # Directorio donde se guardará el modelo y los registros
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Crear y envolver el entorno con Monitor
    env = Monitor(PuzzleEnv(), log_dir)

    # Verificar el entorno
    check_env(env)

    # Crear el modelo DQN
    model = DQN('MlpPolicy', env, verbose=1)

    # Callback para evaluar y guardar los resultados del modelo
    eval_callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=1000, deterministic=True, render=False)

    # Callback para guardar la imagen PNG del progreso del entrenamiento
    image_callback = SaveImageCallback(log_dir, eval_freq=900)

    # Entrenar el modelo
    model.learn(total_timesteps=900, callback=[eval_callback, image_callback])

    # Guardar el modelo entrenado
    model.save(os.path.join(log_dir, "puzzle_model"))

    print("Training complete.")

    # Cargar el modelo entrenado
    model = DQN.load(os.path.join(log_dir, "puzzle_model"))

    # Evaluar el modelo
    obs, _ = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()
