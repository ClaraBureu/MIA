import torch.nn as nn

class QNetwork(nn.Module):
    """Red neuronal que implementa la función Q para el aprendizaje por refuerzo.
    
    Esta red toma un estado como entrada y produce valores Q para cada acción posible.
    La arquitectura consiste en capas lineales completamente conectadas con activaciones ReLU.
    
    Args:
        input_dim (int): Dimensión del espacio de estados
        output_dim (int): Dimensión del espacio de acciones
        hidden_units (list): Lista con el número de unidades en cada capa oculta
    """
    def __init__(self, input_dim, output_dim, hidden_units=[64, 64]):
        super(QNetwork, self).__init__()
        
        # Crear las capas ocultas
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Realiza la pasada hacia adelante de la red.
        
        Args:
            x (torch.Tensor): Tensor de entrada que representa el estado
            
        Returns:
            torch.Tensor: Valores Q para cada acción posible
        """
        return self.network(x)
