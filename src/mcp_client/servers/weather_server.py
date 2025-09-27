"""
Sample weather server implementation
"""

import json
import asyncio
from typing import Dict, Any, List
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx


class WeatherServer:
    """Sample weather MCP server"""

    def __init__(self):
        self.server = Server("weather-server")
        self.setup_tools()

    def setup_tools(self):
        """Setup weather-related tools"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_weather",
                    description="Get current weather for a city",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "units": {
                                "type": "string",
                                "enum": ["metric", "imperial"],
                                "description": "Temperature units",
                                "default": "metric",
                            },
                        },
                        "required": ["city"],
                    },
                ),
                Tool(
                    name="get_forecast",
                    description="Get weather forecast for a city",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {
                                "type": "integer",
                                "description": "Number of days for forecast (1-5)",
                                "minimum": 1,
                                "maximum": 5,
                                "default": 3,
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "get_weather":
                return await self._get_weather(arguments)
            elif name == "get_forecast":
                return await self._get_forecast(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _get_weather(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get current weather"""
        city = args["city"]
        units = args.get("units", "metric")

        # Mock weather data (in production, use real weather API)
        weather_data = {
            "city": city,
            "temperature": 22 if units == "metric" else 72,
            "units": "°C" if units == "metric" else "°F",
            "description": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 8 if units == "metric" else 18,
            "wind_units": "km/h" if units == "metric" else "mph",
        }

        result = (
            f"Current weather in {weather_data['city']}:\n"
            f"Temperature: {weather_data['temperature']}{weather_data['units']}\n"
            f"Condition: {weather_data['description']}\n"
            f"Humidity: {weather_data['humidity']}%\n"
            f"Wind: {weather_data['wind_speed']} {weather_data['wind_units']}"
        )

        return [TextContent(type="text", text=result)]

    async def _get_forecast(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get weather forecast"""
        city = args["city"]
        days = args.get("days", 3)

        # Mock forecast data
        forecast_data = []
        for i in range(days):
            forecast_data.append(
                {
                    "day": f"Day {i+1}",
                    "high": 25 - i,
                    "low": 15 - i,
                    "condition": ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Windy"][
                        i % 5
                    ],
                }
            )

        result = f"{days}-day weather forecast for {city}:\n"
        for day_data in forecast_data:
            result += (
                f"{day_data['day']}: {day_data['condition']}, "
                f"High: {day_data['high']}°C, Low: {day_data['low']}°C\n"
            )

        return [TextContent(type="text", text=result)]

    def run(self, host: str = "localhost", port: int = 8001):
        """Run the weather server"""
        import uvicorn
        from mcp.server.fastapi import create_app

        app = create_app(self.server)
        uvicorn.run(app, host=host, port=port)


async def main():
    """Main function to run the weather server"""
    server = WeatherServer()
    await server.server.run()


if __name__ == "__main__":
    asyncio.run(main())
