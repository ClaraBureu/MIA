from collections import Counter

import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer

class DQNAgent(Base_Agent):
    """Agente que implementa el algoritmo Deep Q-Learning.
    
    Este agente utiliza una red neuronal profunda para aproximar la función Q
    y aprender una política óptima a través de la experiencia. Implementa
    características clave como:
    - Experience replay para romper correlaciones temporales
    - Red neuronal objetivo para estabilizar el entrenamiento
    - Exploración epsilon-greedy para balance exploración-explotación
    """
    agent_name = "DQN"
    def __init__(self, config):
        # Hiperparámetros fijos
        self.hyperparameters = {
            'learning_rate': 0.001,
            'batch_size': 64,
            'buffer_size': 10000,
            'gamma': 0.99,
            'tau': 0.001,
            'update_every': 4,
            'gradient_clipping_norm': 1.0,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'epsilon_decay_rate_denominator': 1.0
        }
        
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], 
                                  self.hyperparameters["batch_size"], 
                                  config.seed, 
                                  self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.optimizer = optim.Adam(self.q_network_local.parameters(),
                                  lr=self.hyperparameters["learning_rate"])
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

    def reset_game(self):
        super(DQNAgent, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self, state, action, reward, next_state, done):
        """Guarda experiencia en memoria y aprende si es el momento.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        # Guardar experiencia en memoria
        self.memory.add_experience(state, action, reward, next_state, done)
        
        # Aprender cada update_every pasos
        self.global_step_number = (self.global_step_number + 1) % self.hyperparameters["update_every"]
        if self.global_step_number == 0 and len(self.memory) > self.hyperparameters["batch_size"]:
            self.learn()

    def pick_action(self, state=None):
        """Selecciona una acción usando la red Q local y una política epsilon-greedy.
        
        Args:
            state: Estado actual del entorno. Si es None, usa self.state
            
        Returns:
            int: Acción seleccionada
        """
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        if isinstance(state, tuple): state = state[0]  # Extract state from (state, info) tuple
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        
        # Get Q-values from local network
        self.q_network_local.eval()  # puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()  # puts network back in training mode
        
        # Use epsilon-greedy strategy to select action
        action_info = {
            "action_values": action_values,
            "turn_off_exploration": self.turn_off_exploration,
            "episode_number": self.episode_number
        }
        action = self.exploration_strategy.perturb_action_for_exploration_purposes(action_info)
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Ejecuta una iteración de aprendizaje para la red Q.
        
        Actualiza los pesos de la red Q usando un batch de experiencias
        del buffer de replay. Implementa el algoritmo DQN estándar.
        
        Args:
            experiences: Tupla opcional de (estados, acciones, recompensas, 
                        siguientes estados, terminados). Si es None, se muestrea
                        del buffer de replay.
        """
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Calcula la pérdida necesaria para entrenar la red Q"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Calcula los valores Q objetivo que compararemos con los valores Q predichos para crear la pérdida para entrenar la red Q"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Calcula los valores Q para el siguiente estado que usaremos para crear la pérdida para entrenar la red Q"""
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Calcula los valores Q objetivo para los estados actuales.
        
        Args:
            rewards: Tensor de recompensas
            Q_targets_next: Tensor de valores Q para los siguientes estados
            dones: Tensor de flags de terminación
            
        Returns:
            Tensor de valores Q objetivo para los estados actuales
        """
        Q_targets_current = rewards + (self.hyperparameters["gamma"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Calcula los valores Q esperados que usaremos para crear la pérdida para entrenar la red Q"""
        Q_expected = self.q_network_local(states).gather(1, actions.long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Guarda la política"""
        torch.save(self.q_network_local.state_dict(), "Models/{}_local_network.pt".format(self.agent_name))

    def time_for_q_network_to_learn(self):
        """Retorna un booleano indicando si se han tomado suficientes pasos para comenzar el aprendizaje y hay
        suficientes experiencias en el buffer de replay para aprender"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Retorna un booleano indicando si se han tomado suficientes pasos para comenzar el aprendizaje"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """Extrae una muestra aleatoria de experiencias del buffer de memoria"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones