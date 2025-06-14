# Reinforcement Learning II - CartPole

Este proyecto implementa y compara tres variantes del algoritmo Deep Q-Network (DQN) para resolver el entorno CartPole-v1 de Gymnasium:

- DQN (Deep Q-Network)
- Double DQN
- Dueling DQN

## Estructura del Proyecto

```
.
├── agents/                     # Implementaciones de los agentes
│   ├── base_agent.py          # Clase base con funcionalidad común
│   ├── dqn_agent.py           # Implementación de DQN
│   ├── double_dqn_agent.py    # Implementación de Double DQN
│   └── dueling_dqn_agent.py   # Implementación de Dueling DQN
├── exploration_strategies/     # Estrategias de exploración
├── networks/                   # Arquitecturas de redes neuronales
├── results/                    # Resultados del entrenamiento
├── train_cartpole.py          # Script principal de entrenamiento
└── requirements.txt           # Dependencias del proyecto
```

## Requisitos

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Para entrenar los tres agentes:

```bash
python train_cartpole.py
```

Esto entrenará secuencialmente DQN, Double DQN y Dueling DQN en el entorno CartPole-v1. Para cada agente:
- Se realizan 500 episodios de entrenamiento
- Se graba un video de evaluación
- Se guarda el modelo entrenado
- Se generan gráficos de rendimiento

Los resultados se guardan en el directorio `results/` organizados por tipo de agente.

## Características

### DQN (Deep Q-Network)
- Implementación estándar de DQN con experience replay
- Red neuronal para aproximar la función Q
- Estrategia de exploración epsilon-greedy

### Double DQN
- Reduce la sobreestimación del valor Q usando dos redes
- Una red para selección de acciones y otra para evaluación
- Mejor estabilidad durante el entrenamiento

### Dueling DQN
- Arquitectura con dos streams: valor del estado (V) y ventajas de acciones (A)
- Mejor estimación del valor de estados
- Más eficiente en entornos con muchas acciones

## Configuración

Los hiperparámetros se pueden modificar en el archivo `train_cartpole.py`:

- Learning rate
- Batch size
- Buffer size
- Gamma (factor de descuento)
- Tau (actualización soft de la red target)
- Parámetros de exploración epsilon-greedy

## Resultados

Los resultados se guardan en el directorio `results/` con la siguiente estructura:

```
results/
├── dqn/
│   ├── evaluation.gif         # Video del agente entrenado
│   └── model_500ep.pth       # Modelo guardado
├── double_dqn/
│   ├── evaluation.gif
│   └── model_500ep.pth
└── dueling_dqn/
    ├── evaluation.gif
    └── model_500ep.pth
```

## Autores

- Alex Barria
- Clara Bureu

## Basado en

Este trabajo se basó en el repositorio [Deep-Reinforcement-Learning-Algorithms-with-PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch), el cual sirvió como punto de partida para la implementación de los agentes DQN, Double DQN y Dueling DQN.
