# import abstract base class
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import argparse


class WrapperBase(ABC):
    @abstractmethod
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the wrapper with the given arguments.

        Args:
            args (Any): The arguments to initialize the wrapper with.
        """
        pass

    # @property
    # @abstractmethod
    # def topic_diversity(self) -> float:
    #     """
    #     Calculate and return the topic diversity.
    #     """
    #     pass

    # @property
    # @abstractmethod
    # def topic_coherence(self) -> float:
    #     """
    #     Calculate and return the topic coherence.
    #     """
    #     pass

    @abstractmethod
    def visualize_hierarchy(self) -> None:
        """
        Visualize the hierarchy of topics.
        """
        pass

    @abstractmethod
    def visualize_weights(self) -> None:
        """
        Visualize the weights of topics.
        """
        pass