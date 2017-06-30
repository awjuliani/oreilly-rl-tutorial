import atexit
import io
import json
import logging
import numpy as np
import os
import scipy.misc
import socket
import subprocess

from PIL import Image
from sys import platform


class UnityEnvironment(object):
    def __init__(self, file_name, train_model=True, worker_num=0, headless=False, config={}):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Inspired by OpenAI gym: https://github.com/openai/gym/tree/master/gym/envs
        :string file_name: Name of Unity environment binary.
        :bool train_model: Whether to load the environment for training [True] or for inference [False].
        :int worker_num: Number to add to communication port [0]. Used for asynchronous agent scenarios.
        :bool headless: Whether to load the environment headless [True] or in a window [False].
        :dict config: Used to define additional environment-specific config flags.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        atexit.register(self.close)
        port = 5005 + worker_num
        self._buffer_size = 120000

        try:
            cwd = os.getcwd()
            launch_string = ""
            if platform == "linux" or platform == "linux2":
                launch_string = os.path.join(cwd, file_name + '.x86_64')
            elif platform == 'darwin':
                launch_string = os.path.join(cwd, file_name + '.app', 'Contents', 'MacOS', file_name)
            elif platform == 'win32':
                launch_string = os.path.join(cwd, file_name + '.exe')

            # Collect environment-specific config flags
            config_list = []
            for key in config:
                config_list.append(key)
                config_list.append(str(config[key]))

            # Launch Unity environment
            subprocess.Popen(
                [launch_string,
                 '--port', str(port),
                 '--train', str(train_model),
                 '{}'.format('-batchmode' if headless else '')] + config_list)

            # Establish communication socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("localhost", port))

            # Start listening on socket
            self._socket.listen(1)
            self._conn, _ = self._socket.accept()

            # Get state and action space from environment
            p = self._conn.recv(self._buffer_size).decode('utf-8')
            p = json.loads(p)
            self._state_space_size = p["state_size"]
            self._number_observations = p["observation_size"]
            self._action_space_size = p["action_size"]
            self._action_descriptions = p["action_descriptions"]
            self._environment_name = p["env_name"]
            self._action_space_type = p["action_space_type"]
            self._state_space_type = p["state_space_type"]
            self._number_agents = p["num_agents"]
            self._conn.send(b".")
            self._loaded = True
            self.logger.info("'{}' started successfully!".format(self._environment_name))

            self.resolution = 80
            self.bw_render = False

        except socket.error:
            self.close()
            raise Exception("Couldn't launch new environment because communication port {} is still in use. "
                            "You may need to manually close a previously opened environment.".format(str(port)))

    @property
    def state_space_size(self):
        return self._state_space_size

    @property
    def number_observations(self):
        return self._number_observations

    @property
    def action_space_size(self):
        return self._action_space_size

    @property
    def action_descriptions(self):
        return self._action_descriptions

    @property
    def environment_name(self):
        return self._environment_name

    @property
    def action_space_type(self):
        return self._action_space_type

    @property
    def state_space_type(self):
        return self._state_space_type

    @property
    def number_agents(self):
        return self._number_agents

    @property
    def done(self):
        return self._done

    def _process_pixels(self, image_bytes=None):
        """
        Converts bytearray observation image into numpy array, resizes it, and optionally converts it to greyscale
        :param image_bytes: input bytearray corresponding to image
        :param resolution: Desired output resolution in length and width
        :param bw: Whether to greyscale the image
        :return: processed numpy array of observation from environment
        """
        s = bytearray(image_bytes)
        image = Image.open(io.BytesIO(s))
        s = np.array(image)
        s = scipy.misc.imresize(s, [self.resolution, self.resolution]) / 255.0
        if self.bw_render:
            s = np.mean(s, axis=2)
            s = np.reshape(s, [self.resolution, self.resolution, 1])
        return s

    def __str__(self):
        message = '''Unity environment name: {0}
        Number of agents: {1}
        Number of observations (per agent): {2}
        State space type: {3}
        State space size (per agent): {4}
        Action space type: {5}
        Action space size (per agent): {6}
        Action descriptions: {7}'''.format(self._environment_name, str(self._number_agents),
                                           str(self._number_observations), self._state_space_type,
                                           str(self._state_space_size), self._action_space_type,
                                           str(self._action_space_size),
                                           ', '.join(self._action_descriptions))
        return message

    def _get_state_image(self):
        s = self._conn.recv(self._buffer_size)
        s = self._process_pixels(image_bytes=s)
        self._conn.send(b"RECEIVED")
        return s

    def _get_state_dict(self):
        state = self._conn.recv(self._buffer_size).decode('utf-8')
        state_dict = json.loads(state)
        return state_dict

    def reset(self):
        """
        Sends a signal to reset the unity environment.
        :return: A new (observations, state) tuple corresponding to the initial reset state of the environment.
        """
        if self._loaded:
            self._conn.send(b"RESET")
            observations, state, reward, done = self._get_state()
            return observations, state
        else:
            raise Exception("No Unity environment is loaded.")

    def _get_state(self):
        observations = []
        for i in range(self._number_observations):
            observations.append(self._get_state_image())
        state_dict = self._get_state_dict()
        state = np.array(state_dict["state"])
        reward = float(state_dict["reward"])
        done = str(state_dict["done"]) == "True"
        self._done = done
        return observations, state, reward, done

    def _send_action(self, action, value):
        self._conn.recv(self._buffer_size)
        action_message = {"action": action, "value": value}
        self._conn.send(json.dumps(action_message).encode('utf-8'))

    def step(self, action, value):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly, and returns
        observation, state, and reward information to the agent.
        :param action: Agent's action to send to environment. Can be a scalar or vector of int/floats.
        :param value: Value estimate to send to environment for visualization. Can be a scalar or vector of float(s).
        :return: An (observations, state, reward, done) tuple corresponding to the new state of the environment.
        """
        if self._loaded and not self._done:
            if isinstance(action, (int, np.int_)):
                action = [int(action)]
            if isinstance(value, (int, np.int_, float, np.float_)):
                value = [float(value)]
            if (self._action_space_type == "discrete" and len(action) == self._number_agents) or \
                    (self._action_space_type == "continuous" and len(
                        action) == self._action_space_size * self._number_agents):
                self._conn.send(b"STEP")
                self._send_action(action, value)
                return self._get_state()
            else:
                exception_message = '''There was a mismatch between the provided action and environment's expectation:
                The environment expected {0} {1} action(s), but was provided: {2}'''.format(
                    self._number_agents if self._action_space_type == "discrete"
                    else str(self._action_space_size * self._number_agents),
                    self._action_space_type, str(action))
                raise Exception(exception_message)
        elif not self._loaded:
            raise Exception("No Unity environment is loaded.")
        elif self._done:
            raise Exception("The episode is completed. Reset the environment with 'reset()'")

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._conn.send(b"EXIT")
            self._conn.close()
            self._socket.close()
            self._loaded = False
        else:
            raise Exception("No Unity environment is loaded.")
