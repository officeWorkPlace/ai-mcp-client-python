"""
Sample calculator server implementation
"""

import json
import asyncio
import math
from typing import Dict, Any, List
from mcp.server import Server
from mcp.types import Tool, TextContent


class CalculatorServer:
    """Sample calculator MCP server"""

    def __init__(self):
        self.server = Server("calculator-server")
        self.setup_tools()

    def setup_tools(self):
        """Setup calculator tools"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="add",
                    description="Add two numbers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                ),
                Tool(
                    name="subtract",
                    description="Subtract two numbers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                ),
                Tool(
                    name="multiply",
                    description="Multiply two numbers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                ),
                Tool(
                    name="divide",
                    description="Divide two numbers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "Dividend"},
                            "b": {"type": "number", "description": "Divisor"},
                        },
                        "required": ["a", "b"],
                    },
                ),
                Tool(
                    name="power",
                    description="Raise a number to a power",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "base": {"type": "number", "description": "Base number"},
                            "exponent": {"type": "number", "description": "Exponent"},
                        },
                        "required": ["base", "exponent"],
                    },
                ),
                Tool(
                    name="sqrt",
                    description="Calculate square root of a number",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "number",
                                "description": "Number to calculate square root of",
                                "minimum": 0,
                            }
                        },
                        "required": ["number"],
                    },
                ),
                Tool(
                    name="factorial",
                    description="Calculate factorial of a non-negative integer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "integer",
                                "description": "Non-negative integer",
                                "minimum": 0,
                                "maximum": 100,
                            }
                        },
                        "required": ["number"],
                    },
                ),
                Tool(
                    name="evaluate_expression",
                    description="Safely evaluate a mathematical expression",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "add":
                    result = arguments["a"] + arguments["b"]
                    return [
                        TextContent(
                            type="text",
                            text=f"{arguments['a']} + {arguments['b']} = {result}",
                        )
                    ]

                elif name == "subtract":
                    result = arguments["a"] - arguments["b"]
                    return [
                        TextContent(
                            type="text",
                            text=f"{arguments['a']} - {arguments['b']} = {result}",
                        )
                    ]

                elif name == "multiply":
                    result = arguments["a"] * arguments["b"]
                    return [
                        TextContent(
                            type="text",
                            text=f"{arguments['a']} × {arguments['b']} = {result}",
                        )
                    ]

                elif name == "divide":
                    if arguments["b"] == 0:
                        return [
                            TextContent(type="text", text="Error: Division by zero")
                        ]
                    result = arguments["a"] / arguments["b"]
                    return [
                        TextContent(
                            type="text",
                            text=f"{arguments['a']} ÷ {arguments['b']} = {result}",
                        )
                    ]

                elif name == "power":
                    result = pow(arguments["base"], arguments["exponent"])
                    return [
                        TextContent(
                            type="text",
                            text=f"{arguments['base']} ^ {arguments['exponent']} = {result}",
                        )
                    ]

                elif name == "sqrt":
                    if arguments["number"] < 0:
                        return [
                            TextContent(
                                type="text",
                                text="Error: Cannot calculate square root of negative number",
                            )
                        ]
                    result = math.sqrt(arguments["number"])
                    return [
                        TextContent(
                            type="text", text=f"√{arguments['number']} = {result}"
                        )
                    ]

                elif name == "factorial":
                    number = arguments["number"]
                    if number < 0:
                        return [
                            TextContent(
                                type="text",
                                text="Error: Factorial is not defined for negative numbers",
                            )
                        ]
                    result = math.factorial(number)
                    return [TextContent(type="text", text=f"{number}! = {result}")]

                elif name == "evaluate_expression":
                    result = self._safe_eval(arguments["expression"])
                    return [
                        TextContent(
                            type="text", text=f"{arguments['expression']} = {result}"
                        )
                    ]

                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate a mathematical expression"""
        # Allowed names for evaluation
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        # Remove any potentially dangerous characters/keywords
        dangerous = [
            "import",
            "exec",
            "eval",
            "__",
            "open",
            "file",
            "input",
            "raw_input",
        ]
        for dangerous_word in dangerous:
            if dangerous_word in expression.lower():
                raise ValueError(f"Dangerous keyword '{dangerous_word}' not allowed")

        try:
            # Compile and evaluate the expression
            code = compile(expression, "<string>", "eval")
            result = eval(code, allowed_names, {})
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    def run(self, host: str = "localhost", port: int = 8002):
        """Run the calculator server"""
        import uvicorn
        from mcp.server.fastapi import create_app

        app = create_app(self.server)
        uvicorn.run(app, host=host, port=port)


async def main():
    """Main function to run the calculator server"""
    server = CalculatorServer()
    await server.server.run()


if __name__ == "__main__":
    asyncio.run(main())
