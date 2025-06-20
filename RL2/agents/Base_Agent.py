import logging
import os
import sys
import gymnasium as gym
import random
from networks.qnetwork import QNetwork
import numpy as np
import torch
import time
# import tensorflow as tf
from networks.qnetwork import QNetwork
# from tensorboardX import SummaryWriter
from torch.optim import optimizer

class Base_Agent(object):
    """Clase base para todos los agentes de aprendizaje por refuerzo.
    
    Esta clase implementa la funcionalidad básica compartida por todos los agentes:
    - Interacción con el entorno
    - Gestión de episodios y puntuaciones
    - Configuración de semillas aleatorias
    - Creación de redes neuronales
    - Actualización de parámetros y optimización
    
    Args:
        config: Objeto de configuración con los parámetros del agente
    """

    def __init__(self, config):
        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        # if self.debug_mode: self.tensorboard = SummaryWriter()
        self.config = config
        self.hyperparameters = config.hyperparameters
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.environment_title = self.get_environment_title()
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.state_size = int(self.get_state_size())
        self.q_network_local = QNetwork(input_dim=self.state_size, output_dim=self.action_size, hidden_units=self.hyperparameters.get('linear_hidden_units', [64, 64]))
        self.config.action_size = self.action_size

        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        self.log_game_info()

    def step(self):
        """Ejecuta un paso en el entorno.
        
        Este método debe ser implementado por las clases hijas.
        """
        """Ejecuta un paso en el entorno. Este método debe ser implementado por las clases hijas"""
        raise ValueError("Step needs to be implemented by the agent")

    def get_environment_title(self):
        """Extrae el nombre del entorno"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach": return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze": return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper": return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d": return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<": name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<": name = name[1:]
                if name[-3:] == "Env": name = name[:-3]
        return name

    def get_lowest_possible_episode_score(self):
        """Retorna la puntuación más baja posible que se puede obtener en un episodio del entorno"""
        if self.environment_title == "Taxi": return -800
        return None

    def get_action_size(self):
        """Obtiene el tamaño de la acción del entorno gym en la forma correcta para una red neuronal"""
        if "overwrite_action_size" in self.config.__dict__: return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__: return self.environment.action_size
        if self.action_types == "DISCRETE": return self.environment.action_space.n
        else: return self.environment.action_space.shape[0]

    def get_state_size(self):
        """Obtiene el tamaño del estado del entorno gym en la forma correcta para una red neuronal"""
        state, _ = self.environment.reset()
        if isinstance(state, dict):
            state_size = state["observation"].shape[0] + state["desired_goal"].shape[0]
            return state_size
        else:
            return state.shape[0]

    def get_score_required_to_win(self):
        """Obtiene la puntuación promedio requerida para ganar el juego"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach": return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d"]:
            print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try: return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Obtiene el número de intentos sobre los que promediar una puntuación"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole"]: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials

    def setup_logger(self):
        """Configura el logger"""
        filename = "Training.log"
        try: 
            if os.path.isfile(filename): 
                os.remove(filename)
        except: pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def log_game_info(self):
        """Registra información relacionada con el juego"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, self.lowest_possible_episode_score,
                      self.state_size, self.hyperparameters, self.average_score_required_to_win, self.rolling_score_window,
                      self.device]):
            self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Configura todas las semillas aleatorias para reproducibilidad.
        
        Args:
            random_seed (int): Semilla para todos los generadores aleatorios
        """
        """Configura todas las semillas aleatorias posibles para que los resultados sean reproducibles"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self):
        """Reinicia el entorno para un nuevo episodio.
        
        Reinicia el estado del entorno y las variables de seguimiento del episodio.
        """
        """Reinicia la información del juego para estar listos para un nuevo episodio"""
        self.state, _ = self.environment.reset(seed=self.config.seed)
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))

    def track_episodes_data(self):
        """Guarda los datos de los episodios recientes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Ejecuta el juego hasta completarlo n veces, luego resume los resultados y guarda el modelo (si se solicita)"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal: self.show_whether_achieved_goal()
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def conduct_action(self, action):
        """Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar en el entorno
            
        La acción se ejecuta en el entorno y se actualizan:
        - next_state: Siguiente estado
        - reward: Recompensa recibida
        - done: Si el episodio ha terminado
        - total_episode_score_so_far: Puntuación acumulada
        """
        """Ejecuta una acción en el entorno"""
        self.next_state, self.reward, terminated, truncated, _ = self.environment.step(action)
        self.done = terminated or truncated
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]: self.reward = max(min(self.reward, 1.0), -1.0)


    def save_and_print_result(self):
        """Guarda e imprime los resultados del juego"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Guarda el resultado de un episodio del juego"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Actualiza el mejor resultado de episodio visto hasta ahora"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Imprime los resultados del último episodio"""
        text = """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Imprime si el agente alcanzó el objetivo del entorno"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Retorna el episodio en el que el agente alcanzó el objetivo o -1 si nunca lo alcanzó"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def update_learning_rate(self, starting_lr,  optimizer):
        """Reduce la tasa de aprendizaje según qué tan cerca estamos de la solución"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def enough_experiences_to_learn_from(self):
        """Booleano que indica si hay suficientes experiencias en el buffer de memoria para aprender"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        """Guarda la experiencia reciente en el buffer de memoria"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Toma un paso de optimización calculando los gradientes dada la pérdida y luego actualizando los parámetros"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))


    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Actualiza la red objetivo en la dirección de la red local pero tomando un tamaño de paso
        menor que uno para que los valores de los parámetros de la red objetivo sigan a los de la red local. Esto ayuda a estabilizar el entrenamiento"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Crea una red neuronal para el agente.
        
        Args:
            input_dim (int): Dimensión de entrada
            output_dim (int): Dimensión de salida
            key_to_use (str, optional): Clave para los hiperparámetros
            override_seed (int, optional): Semilla para sobreescribir la predeterminada
            hyperparameters (dict, optional): Hiperparámetros personalizados
            
        Returns:
            QNetwork: Instancia de la red neuronal creada
        """
        """Crea una red neuronal para que la use el agente"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return QNetwork(input_dim=input_dim, output_dim=output_dim, hidden_units=hyperparameters["linear_hidden_units"]).to(self.device)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def freeze_all_but_output_layers(self, network):
        """Congela todas las capas excepto la capa de salida de una red"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Descongela todas las capas de una red"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())