"""
Custom MCP servers for Global MCP Client
"""

from .calculator_server import CalculatorServer
from .weather_server import WeatherServer

__all__ = [
    "WeatherServer",
    "CalculatorServer",
]
