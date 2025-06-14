import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DDQN
from agents.dueling_dqn_agent import DuelingDQNAgent

@dataclass
class Config:
    """Configuración para los agentes de aprendizaje por refuerzo.
    
    Args:
        environment: Entorno de gymnasium
        hyperparameters: Diccionario con los hiperparámetros del agente
        seed: Semilla para reproducibilidad
        debug_mode: Si True, activa el modo de depuración
        use_GPU: Si True, usa GPU si está disponible
        visualise_individual_results: Si True, muestra resultados individuales
    """
    environment: gym.Env
    hyperparameters: dict
    seed: int = 42
    debug_mode: bool = False
    use_GPU: bool = torch.cuda.is_available()
    visualise_individual_results: bool = True

def create_agent(agent_type, config):
    """Crea un agente del tipo especificado.
    
    Args:
        agent_type (str): Tipo de agente ('dqn', 'double_dqn', 'dueling_dqn')
        config (Config): Configuración del agente
        
    Returns:
        Agent: Instancia del agente
    """
    agents = {
        'dqn': DQNAgent,
        'double_dqn': DDQN,
        'dueling_dqn': DuelingDQNAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f'Tipo de agente no válido: {agent_type}')
        
    return agents[agent_type](config)

def record_video(env, agent, video_path, num_episodes=3):
    """Graba un video del agente actuando en el entorno.
    
    Args:
        env (gym.Env): Entorno de gymnasium
        agent: Agente entrenado
        video_path (str): Ruta donde guardar el video
        num_episodes (int): Número de episodios a grabar
    """
    # Configurar entorno para grabar
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    frames = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            frames.append(env.render())
            action = agent.pick_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
    env.close()
    
    # Guardar video como GIF
    print('\nGuardando video...')
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Crear animación
    fig = plt.figure()
    plt.axis('off')
    
    def animate(i):
        plt.imshow(frames[i])
        return plt.gcf()
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
    anim.save(video_path, writer='pillow')
    plt.close()
    
    print(f'\nVideo guardado como {video_path}')



def get_default_config(env):
    """Obtiene la configuración por defecto para el agente.
    
    Args:
        env: Entorno de gymnasium
    
    Returns:
        Config: Configuración por defecto
    """
    hyperparameters = {
        # Parámetros del buffer de replay
        "buffer_size": 100000,
        "batch_size": 64,
        # Parámetros de la red
        "linear_hidden_units": [64, 64],
        "learning_rate": 0.001,
        # Parámetros del entrenamiento
        "gamma": 0.99,
        "tau": 0.001,
        "update_every": 4,
        "gradient_clipping_norm": 0.5,
        # Parámetros de exploración
        "epsilon_decay_rate_denominator": 200
    }
    
    config = Config(
        environment=env,
        hyperparameters=hyperparameters,
        seed=42,
        debug_mode=False
    )
    
    return config

def train_and_evaluate(agent_type, config_override=None):
    """Entrena y evalúa un agente específico.
    
    Args:
        agent_type (str): Tipo de agente ('dqn', 'double_dqn', 'dueling_dqn')
        config_override (dict, optional): Configuración personalizada
    """
    # Crear entorno
    env = gym.make('CartPole-v1')
    
    # Obtener configuración
    config = get_default_config(env)
    if config_override:
        config.hyperparameters.update(config_override)
    
    # Crear agente
    agent = create_agent(agent_type, config)
    
    # Entrenamiento
    num_episodes = 500  
    scores = []
    window_size = 2  
    moving_averages = []
    
    print(f'\nEntrenando agente {agent_type.upper()}...')
    
    # Crear directorio para resultados
    results_dir = os.path.join('results', agent_type)
    os.makedirs(results_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_score = 0
        done = False
        
        while not done:
            action = agent.pick_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward
            
        scores.append(episode_score)
        
        # Calcular media móvil
        if len(scores) >= window_size:
            window_average = np.mean(scores[-window_size:])
            moving_averages.append(window_average)
            print(f'Episodio {episode+1}/{num_episodes} | Score: {episode_score:.2f} | Media: {np.mean(scores):.2f}')
            print(f'Media móvil de últimos {window_size} episodios: {window_average:.2f}')
    
    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Puntuaciones', alpha=0.3)
    plt.plot(range(window_size-1, len(scores)), moving_averages, 
             label=f'Media móvil ({window_size} episodios)', linewidth=2)
    
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title(f'Curva de Aprendizaje - {agent_type.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico
    plot_path = os.path.join(results_dir, 'learning_curve.png')
    plt.savefig(plot_path)
    plt.tight_layout()
    plt.savefig(f'results/{agent_type}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Grabar video de evaluación
    print('\nGrabando video de evaluación...')
    video_path = os.path.join(results_dir, 'evaluation.gif')
    record_video(env, agent, video_path)
    
    # Guardar modelo entrenado
    print('\nGuardando modelo entrenado...')
    model_path = os.path.join(results_dir, f'model_{num_episodes}ep.pth')
    torch.save({
        'model_state_dict': agent.q_network_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'num_episodes': num_episodes,
        'scores': scores,
        'moving_averages': moving_averages,
        'hyperparameters': config.hyperparameters
    }, model_path)
    print(f'Modelo guardado como {model_path}')
    
    env.close()
    
    return np.mean(scores), np.std(scores)

def main():
    """Función principal que entrena los tres tipos de agentes."""
    results = {}
    agent_types = ['dqn', 'double_dqn', 'dueling_dqn']
    
    for agent_type in agent_types:
        print(f'\nEntrenando {agent_type.upper()}...')
        mean_score, std_score = train_and_evaluate(agent_type)
        results[agent_type] = (mean_score, std_score)
    
    print('\nResultados finales:')
    for agent_type, (mean, std) in results.items():
        print(f'{agent_type.upper()}: {mean:.2f} ± {std:.2f}')
        return
    # Configuración del agente
    config = Config(
        environment=env,
        hyperparameters=hyperparameters,
        use_GPU=True,
        debug_mode=False
    )

    # Seleccionar tipo de agente ('dqn', 'double_dqn', 'dueling_dqn')
    agent_type = "dqn"
    agent = create_agent(agent_type, config)

    # Entrenamiento
    num_episodes = 10  # Reducido a 10 episodios
    scores = []
    window_size = 2  # Tamaño de la ventana para la media móvil (ajustado para pocos episodios)
    moving_averages = []

    for episode in range(num_episodes):
        agent.reset_game()
        agent.step()
        score = agent.total_episode_score_so_far
        scores.append(score)
        
        # Calcular media móvil
        if len(scores) >= window_size:
            window_average = np.mean(scores[-window_size:])
            moving_averages.append(window_average)
        
        # Calcular media para criterio de terminación
        mean_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Imprimir progreso
        print(f'\rEpisodio {episode+1}/{num_episodes} | Score: {score:.2f} | Media: {mean_score:.2f}', end='')
        
        if (episode + 1) % window_size == 0:
            print(f'\nMedia móvil de últimos {window_size} episodios: {window_average:.2f}')
        
        # Verificar si el entorno está resuelto
        if mean_score >= 195.0 and len(scores) >= 100:
            print(f"\nEntorno resuelto en {episode+1} episodios!")
            break
    
    print("\n\nEntrenamiento completado!")
    
    # Graficar resultados
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Gráfico de puntuaciones
    ax1.plot(scores, label='Puntuaciones', alpha=0.3)
    ax1.plot(range(window_size-1, len(scores)), moving_averages, 
            label=f'Media móvil ({window_size} episodios)', linewidth=2)
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Curva de Aprendizaje - CartPole')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de distribución de recompensas
    ax2.hist(scores, bins=20, alpha=0.7, color='skyblue')
    ax2.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=2, label=f'Media: {np.mean(scores):.2f}')
    ax2.set_xlabel('Recompensa')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Recompensas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráficos
    plt.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
    
    # Mostrar gráficos
    plt.show()
    plt.close()
    env.close()
    
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    
    # Grabar video del agente entrenado
    print('\nGrabando video de evaluación...')
    record_video(agent)
    
    print('\nResultados guardados en:')
    print('- Gráficas: results/learning_curves.png')
    print('- Video: results/videos/cartpole.gif')

if __name__ == "__main__":
    main()
