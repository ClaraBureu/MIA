import torch
from .dqn_agent import DQNAgent
from networks.qnetwork import QNetwork

class DDQN(DQNAgent):
    """Un agente DQN doble"""
    agent_name = "DDQN"

    def __init__(self, config):
        super().__init__(config)
        # Crear red target con los mismos parámetros que la red local
        self.q_network_target = QNetwork(input_dim=self.state_size, 
                                        output_dim=self.action_size, 
                                        hidden_units=self.hyperparameters.get('linear_hidden_units', [64, 64]))
        # Mover la red target al mismo dispositivo que la red local
        self.q_network_target.to(self.device)
        # Copiar pesos de la red local a la target
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())

    def update_target_model(self):
        """Actualiza los pesos de la red target usando los pesos de la red local"""
        tau = self.hyperparameters["tau"]
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def compute_q_values_for_next_states(self, next_states):
        """Calcula los valores Q para el siguiente estado que usaremos para crear la pérdida para entrenar la red Q.
        DQN doble usa el índice local para elegir la acción con el valor Q máximo y luego la red objetivo para calcular el valor Q.
        El razonamiento detrás de esto es que ayudará a evitar que la red sobreestime los valores Q"""
        with torch.no_grad():
            # Asegurar que next_states esté en el dispositivo correcto
            next_states = next_states.to(self.device)
            # Obtener acciones usando la red local
            max_action_indexes = self.q_network_local(next_states).argmax(1)
            # Calcular valores Q usando la red target
            Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
            return Q_targets_next
