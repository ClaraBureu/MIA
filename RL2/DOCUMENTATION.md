# Documentación del Trabajo Práctico - Reinforcement Learning II
**Autores:** Clara Bureu y Alex Barria

## Índice
1. [Introducción](#introducción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Implementación de Agentes](#implementación-de-agentes)
4. [Entrenamiento y Evaluación](#entrenamiento-y-evaluación)
5. [Resultados y Análisis](#resultados-y-análisis)
6. [Guía de Uso](#guía-de-uso)

## Introducción
Este trabajo práctico implementa y compara diferentes algoritmos de Deep Reinforcement Learning aplicados al entorno CartPole-v1 de Gymnasium. Se implementaron tres variantes de DQN (Deep Q-Network):
- DQN clásico
- Double DQN (DDQN)
- Dueling DQN

El objetivo es analizar y comparar el rendimiento de estos algoritmos en términos de velocidad de aprendizaje y estabilidad.

## Estructura del Proyecto
```
RL2/
├── agents/                     # Implementaciones de los agentes
│   ├── base_agent.py          # Clase base para todos los agentes
│   ├── dqn_agent.py           # Implementación DQN clásico
│   ├── double_dqn_agent.py    # Implementación Double DQN
│   └── dueling_dqn_agent.py   # Implementación Dueling DQN
├── exploration_strategies/     # Estrategias de exploración
│   └── Base_Exploration_Strategy.py
├── networks/                   # Arquitecturas de redes neuronales
├── utilities/                  # Funciones y clases auxiliares
├── results/                    # Resultados de entrenamiento
├── train_cartpole.py          # Script principal de entrenamiento
└── train_agents.ipynb         # Notebook interactiva de entrenamiento
```

## Implementación de Agentes

### Base Agent
La clase `Base_Agent` proporciona la funcionalidad común para todos los agentes:
- Gestión del entorno y semillas aleatorias
- Buffer de experiencias para replay
- Logging y seguimiento de métricas
- Métodos base para entrenamiento y evaluación

### DQN Agent
Implementación del algoritmo DQN clásico con las siguientes características:
- Red neuronal para aproximar la función Q
- Experience replay para romper correlaciones temporales
- Red objetivo para estabilizar el entrenamiento
- Actualización suave de pesos (soft update)

### Double DQN Agent
Extiende el DQN clásico para reducir la sobreestimación de valores Q:
- Usa dos redes para desacoplar la selección y evaluación de acciones
- Reduce el sesgo optimista presente en DQN
- Mantiene la misma arquitectura de red que DQN

### Dueling DQN Agent
Implementa una arquitectura de red especializada:
- Separa el cálculo del valor de estado (V) y las ventajas de acciones (A)
- Permite evaluar estados sin aprender el efecto de cada acción
- Mejora la eficiencia del aprendizaje en estados donde las acciones no afectan significativamente el resultado

## Entrenamiento y Evaluación

### Hiperparámetros
```python
hyperparameters = {
    "buffer_size": 100000,      # Tamaño del buffer de replay
    "batch_size": 64,           # Tamaño del batch de entrenamiento
    "linear_hidden_units": [64, 64],  # Arquitectura de la red
    "learning_rate": 0.001,     # Tasa de aprendizaje
    "gamma": 0.99,              # Factor de descuento
    "tau": 0.001,              # Factor de actualización suave
    "update_every": 4,          # Frecuencia de actualización
    "gradient_clipping_norm": 0.5,  # Norma para clipping de gradientes
    "epsilon_decay_rate_denominator": 200  # Tasa de decaimiento de epsilon
}
```

### Proceso de Entrenamiento
1. Inicialización del entorno y agente
2. Recolección de experiencias mediante interacción con el entorno
3. Actualización de la red mediante mini-batch gradient descent
4. Actualización periódica de la red objetivo
5. Registro de métricas y evaluación del rendimiento

### Métricas de Evaluación
- Puntuación por episodio
- Media móvil de puntuaciones
- Desviación estándar de puntuaciones
- Videos de evaluación del agente entrenado

## Resultados y Análisis

Los agentes se evaluaron en el entorno CartPole-v1 durante 500 episodios. Los resultados muestran:

1. **DQN Clásico**:
   - Aprendizaje estable pero relativamente lento
   - Tendencia a sobreestimar valores Q

2. **Double DQN**:
   - Mejor estabilidad en el aprendizaje
   - Estimaciones más precisas de valores Q
   - Convergencia más rápida

3. **Dueling DQN**:
   - Mejor rendimiento en estados donde las acciones tienen impacto similar
   - Aprendizaje más eficiente de la función de valor
   - Mayor estabilidad en las estimaciones

## Guía de Uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Entrenamiento mediante Script
```bash
python train_cartpole.py
```

### Entrenamiento mediante Notebook
1. Abrir `train_agents.ipynb`
2. Ejecutar las celdas en orden
3. Los resultados se guardarán en el directorio `results/`

### Visualización de Resultados
- Las curvas de aprendizaje se muestran en tiempo real
- Los modelos entrenados se guardan automáticamente
- Se generan GIFs del agente entrenado

### Personalización
- Modificar hiperparámetros en la clase `Config`
- Ajustar arquitectura de red en los agentes
- Personalizar estrategias de exploración

## Comparación con TP1 (Keras vs PyTorch)

Este trabajo práctico (TP2) utiliza el mismo entorno CartPole-v1 que el TP1, pero con una diferencia fundamental en la implementación:
- **TP1**: Implementación en Keras/TensorFlow
- **TP2**: Migración completa a PyTorch

### Mejoras de Rendimiento
La migración a PyTorch resultó en una mejora significativa en la velocidad de entrenamiento:
1. **Tiempo de Entrenamiento**:
   - TP1 (Keras): Entrenamiento notablemente más lento
   - TP2 (PyTorch): Reducción significativa en tiempo de entrenamiento usando los mismos recursos de hardware

2. **Posibles Razones de la Mejora**:
   - PyTorch tiene un overhead menor en la creación y actualización de redes neuronales
   - Mejor gestión de memoria y computación en GPU
   - Implementación más eficiente del buffer de experiencias
   - Optimizaciones en el proceso de backpropagation

3. **Ventajas Adicionales de PyTorch**:
   - API más intuitiva y pythónica
   - Mayor flexibilidad en la definición de arquitecturas
   - Mejor depuración gracias al modo eager execution
   - Comunidad activa y documentación extensa

## Conclusiones

1. **Rendimiento Comparativo**:
   - Double DQN mostró el mejor balance entre estabilidad y velocidad de aprendizaje
   - Dueling DQN fue más eficiente en el uso de experiencias
   - DQN clásico sirvió como baseline robusto

2. **Mejoras Potenciales**:
   - Implementar Prioritized Experience Replay
   - Explorar arquitecturas de red más complejas
   - Añadir técnicas de regularización

3. **Lecciones Aprendidas**:
   - Importancia del balance exploración-explotación
   - Impacto de la arquitectura de red en el aprendizaje
   - Valor de las técnicas de estabilización
