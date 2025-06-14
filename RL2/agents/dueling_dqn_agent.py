import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration

class DuelingQNetwork(nn.Module):
    """Red neuronal para Dueling DQN.
    
    Implementa la arquitectura de Dueling DQN con dos streams:
    - Stream de valor (V): estima el valor del estado
    - Stream de ventaja (A): estima la ventaja de cada acción
    """
    def __init__(self, input_dim, output_dim, hidden_units=[64, 64]):
        super(DuelingQNetwork, self).__init__()
        
        # Capas compartidas
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU()
        )
        
        # Stream de valor
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units[1], 1)
        )
        
        # Stream de ventaja
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units[1], output_dim)
        )
        
    def forward(self, x):
        shared = self.shared_layers(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Combinar streams (Q = V + (A - mean(A)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DuelingDQNAgent(Base_Agent):
    """Agente que implementa Dueling DQN.
    
    Utiliza una arquitectura de red neuronal con dos streams separados:
    - Uno para estimar el valor del estado (V)
    - Otro para estimar las ventajas de cada acción (A)
    """
    def __init__(self, config):
        super().__init__(config)
        self.memory = deque(maxlen=config.hyperparameters.get('buffer_size', 2000))
        self.batch_size = config.hyperparameters.get('batch_size', 64)
        self.gamma = config.hyperparameters.get('gamma', 0.99)
        self.tau = config.hyperparameters.get('tau', 0.001)
        self.update_every = config.hyperparameters.get('update_every', 4)
        self.learning_rate = config.hyperparameters.get('learning_rate', 0.001)
        
        # Inicializar estrategia de exploración
        self.exploration_strategy = self.create_exploration_strategy()
        
        # Redes Q
        self.q_network_local = DuelingQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_units=config.hyperparameters.get('linear_hidden_units', [64, 64])
        ).to(self.device)
        
        self.q_network_target = DuelingQNetwork(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_units=config.hyperparameters.get('linear_hidden_units', [64, 64])
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_network_local.parameters(), lr=self.learning_rate)
        
        # Inicializar contador de pasos
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Guarda experiencia en memoria y aprende si es el momento."""
        # Guardar experiencia
        self.memory.append((state, action, reward, next_state, done))
        
        # Aprender cada update_every pasos
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)
    
    def create_exploration_strategy(self):
        """Crea la estrategia de exploración epsilon-greedy."""
        return Epsilon_Greedy_Exploration(self.config)
        
    def act(self, state, epsilon=None):
        """Retorna la acción para un estado dado según la política actual."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()
        
        # Usar estrategia de exploración
        action_info = {"action_values": action_values.cpu().data.numpy()[0],
                      "turn_off_exploration": False,
                      "episode_number": self.episode_number}
        
        action = self.exploration_strategy.select_action(action_info=action_info)
        return action
    
    def learn(self, experiences):
        """Actualiza los parámetros usando un batch de experiencias."""
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.q_network_local(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar red target
        self.soft_update(self.q_network_local, self.q_network_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        """Actualiza el modelo target usando soft update."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def pick_action(self, state=None):
        """Utiliza la red Q local y una política epsilon-greedy para seleccionar una acción"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
            action_values = action_values[:, :-1] #because we treat the last output element as state-value and rest as advantages
        self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        return action

    def compute_q_values_for_next_states(self, next_states):
        """Calcula los valores Q para el siguiente estado que usaremos para crear la función de pérdida para entrenar la red Q.
        DQN Doble usa el índice local para elegir la acción con el valor Q máximo y luego la red objetivo para calcular el valor Q.
        El razonamiento detrás de esto es que ayudará a evitar que la red sobreestime los valores Q"""
        max_action_indexes = self.q_network_local(next_states)[:, :-1].detach().argmax(1)
        duelling_network_output = self.q_network_target(next_states)
        q_values = self.calculate_duelling_q_values(duelling_network_output)
        Q_targets_next = q_values.gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def calculate_duelling_q_values(self, duelling_q_network_output):
        """Calcula los valores Q usando la arquitectura de red dueling. Esta es la ecuación (9) en el paper
        referenciado al inicio de la clase"""
        state_value = duelling_q_network_output[:, -1]
        avg_advantage = torch.mean(duelling_q_network_output[:, :-1], dim=1)
        q_values = state_value.unsqueeze(1) + (duelling_q_network_output[:, :-1] - avg_advantage.unsqueeze(1))
        return q_values

    def compute_expected_q_values(self, states, actions):
        """Calcula los valores Q esperados que usaremos para crear la función de pérdida para entrenar la red Q"""
        duelling_network_output = self.q_network_local(states)
        q_values = self.calculate_duelling_q_values(duelling_network_output)
        Q_expected = q_values.gather(1, actions.long())
        return Q_expected






