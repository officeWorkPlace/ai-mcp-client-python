"""
Enhanced chatbot implementation with Global MCP Client integration
"""

import asyncio
import sys
import time
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from .core import GlobalMCPClient, Config, setup_logging
from .core.exceptions import GlobalMCPClientError
from .utils.validators import InputValidator
from .utils.rate_limiter import RateLimiter


class GlobalMCPChatBot:
    """
    Enhanced chatbot with rich UI and production features
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the chatbot

        Args:
            config: Configuration instance
        """
        self.config = config or Config()
        self.console = Console()
        self.client: Optional[GlobalMCPClient] = None
        self.validator = InputValidator()
        self.rate_limiter = (
            RateLimiter(max_requests=100, window_seconds=3600)
            if self.config.development_mode
            else None
        )

        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=self.config.enable_console_logging,
            development_mode=self.config.development_mode,
        )

        self.session_stats = {
            "queries_processed": 0,
            "tools_called": 0,
            "errors": 0,
            "start_time": None,
        }

    def display_banner(self) -> None:
        """Display welcome banner"""
        banner = Text()
        banner.append("[*] Global MCP Client Chatbot", style="bold blue")
        banner.append("\n")
        banner.append(
            "Production-ready MCP client with multi-server support", style="dim"
        )

        panel = Panel(
            banner,
            title="[bold green]Welcome[/bold green]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def display_server_info(self) -> None:
        """Display connected servers and available tools"""
        if not self.client:
            return

        info = self.client.get_server_info()

        # Servers table
        servers_table = Table(title="Connected Servers")
        servers_table.add_column("Server", style="cyan")
        servers_table.add_column("Status", style="green")
        servers_table.add_column("Tools", style="yellow")

        for server_name in info["connected_servers"]:
            connection = self.client.connections[server_name]
            tool_count = len(connection.tools)
            status = "[+] Healthy" if connection.is_healthy else "[!] Unhealthy"

            servers_table.add_row(server_name, status, str(tool_count))

        self.console.print(servers_table)
        self.console.print()

        # Tools table
        if info["available_tools"]:
            tools_table = Table(title="Available Tools")
            tools_table.add_column("Tool Name", style="cyan")
            tools_table.add_column("Server", style="green")
            tools_table.add_column("Description", style="dim")

            for tool in info["available_tools"][:10]:  # Show first 10 tools
                tools_table.add_row(
                    tool["name"],
                    tool["server"],
                    (
                        tool["description"][:60] + "..."
                        if len(tool["description"]) > 60
                        else tool["description"]
                    ),
                )

            if len(info["available_tools"]) > 10:
                tools_table.add_row(
                    "...", "...", f"and {len(info['available_tools']) - 10} more tools"
                )

            self.console.print(tools_table)
            self.console.print()

    def display_help(self) -> None:
        """Display help information"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[yellow]/help[/yellow] or [yellow]/h[/yellow]     - Show this help message
[yellow]/info[/yellow] or [yellow]/i[/yellow]     - Show server and tool information  
[yellow]/health[/yellow]               - Check server health status
[yellow]/stats[/yellow]                - Show session statistics
[yellow]/clear[/yellow]                - Clear the screen
[yellow]/config[/yellow]               - Show current configuration
[yellow]/quit[/yellow] or [yellow]/q[/yellow]     - Exit the chatbot

[bold cyan]Usage Tips:[/bold cyan]
• Ask natural language questions
• Request specific tool usage
• Chain multiple operations together
• Use descriptive queries for best results

[bold cyan]Examples:[/bold cyan]
• "List files in the current directory"
• "Search for information about Python"
• "Execute a database query to show all users"
• "Fetch the content of https://example.com"
        """

        panel = Panel(
            help_text, title="[bold green]Help[/bold green]", border_style="green"
        )

        self.console.print(panel)

    async def display_health_status(self) -> None:
        """Display server health status"""
        if not self.client:
            self.console.print("[red]No client connection available[/red]")
            return

        with self.console.status("[bold green]Checking server health..."):
            health = await self.client.health_check()

        health_table = Table(title="Server Health Status")
        health_table.add_column("Server", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Uptime", style="yellow")
        health_table.add_column("Tool Calls", style="blue")
        health_table.add_column("Errors", style="red")

        for server_name, status in health["servers"].items():
            status_icon = "[+]" if status["status"] == "healthy" else "[!]"
            uptime = f"{status['uptime']:.1f}s"

            health_table.add_row(
                server_name,
                f"{status_icon} {status['status'].title()}",
                uptime,
                str(status["tool_calls"]),
                str(status["errors"]),
            )

        self.console.print(health_table)

        # Summary
        summary = f"""
[bold]Summary:[/bold]
• Total Servers: {health['total_servers']}
• Healthy Servers: {health['healthy_servers']}
• Total Tools: {health['total_tools']}
        """

        self.console.print(Panel(summary, border_style="green"))

    def display_stats(self) -> None:
        """Display session statistics"""
        if self.session_stats["start_time"]:
            duration = time.time() - self.session_stats["start_time"]
            duration_str = f"{duration:.1f} seconds"
        else:
            duration_str = "N/A"

        stats_text = f"""
[bold]Session Statistics:[/bold]

• Queries Processed: {self.session_stats['queries_processed']}
• Tools Called: {self.session_stats['tools_called']}
• Errors: {self.session_stats['errors']}
• Session Duration: {duration_str}
• Success Rate: {((self.session_stats['queries_processed'] - self.session_stats['errors']) / max(1, self.session_stats['queries_processed']) * 100):.1f}%
        """

        panel = Panel(
            stats_text, title="[bold blue]Statistics[/bold blue]", border_style="blue"
        )

        self.console.print(panel)

    def display_config(self) -> None:
        """Display current configuration"""
        config_text = f"""
[bold]Configuration:[/bold]

[cyan]AI Provider:[/cyan]
• Model: {self.config.default_model}
• Max Tokens: {self.config.max_tokens}
• Temperature: {self.config.temperature}

[cyan]Logging:[/cyan]
• Level: {self.config.log_level}
• File Logging: {self.config.enable_file_logging}
• Console Logging: {self.config.enable_console_logging}

[cyan]Servers:[/cyan]
• Enabled Servers: {len(self.config.get_enabled_servers())}
• Total Configured: {len(self.config.mcp_servers)}

[cyan]Mode:[/cyan]
• Debug: {self.config.debug}
• Development: {self.config.development_mode}
        """

        panel = Panel(
            config_text,
            title="[bold yellow]Configuration[/bold yellow]",
            border_style="yellow",
        )

        self.console.print(panel)

    async def process_command(self, command: str) -> bool:
        """
        Process special commands

        Args:
            command: Command string

        Returns:
            True if should continue, False if should exit
        """
        command = command.lower().strip()

        if command in ["/quit", "/q", "/exit"]:
            return False
        elif command in ["/help", "/h"]:
            self.display_help()
        elif command in ["/info", "/i"]:
            self.display_server_info()
        elif command == "/health":
            await self.display_health_status()
        elif command == "/stats":
            self.display_stats()
        elif command == "/clear":
            self.console.clear()
        elif command == "/config":
            self.display_config()
        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type [yellow]/help[/yellow] for available commands")

        return True

    async def process_query(self, query: str) -> None:
        """
        Process a user query

        Args:
            query: User query
        """
        if not self.client:
            self.console.print("[red]No client connection available[/red]")
            return

        # Validate input
        if not self.validator.validate_query(query):
            self.console.print("[red]Invalid query. Please try again.[/red]")
            return

        # Rate limiting
        if self.rate_limiter and not self.rate_limiter.allow_request():
            self.console.print(
                "[red]Rate limit exceeded. Please wait before making another request.[/red]"
            )
            return

        self.session_stats["queries_processed"] += 1

        try:
            with self.console.status("[bold green]Processing your query..."):
                response = await self.client.process_query(query)

            # Display response
            response_panel = Panel(
                response,
                title="[bold green]Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )

            self.console.print(response_panel)

            # Update stats
            if self.client:
                total_tool_calls = sum(
                    conn.tool_call_count for conn in self.client.connections.values()
                )
                self.session_stats["tools_called"] = total_tool_calls

        except GlobalMCPClientError as e:
            self.session_stats["errors"] += 1
            self.console.print(f"[red]Error: {str(e)}[/red]")
            self.logger.error(f"Query processing error: {e}")
        except Exception as e:
            self.session_stats["errors"] += 1
            self.console.print(f"[red]Unexpected error: {str(e)}[/red]")
            self.logger.error(
                f"Unexpected error during query processing: {e}", exc_info=True
            )

    async def initialize_client(self) -> bool:
        """
        Initialize the MCP client

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.console.status("[bold yellow]Initializing MCP client..."):
                self.client = GlobalMCPClient(self.config)
                await self.client.connect_to_all_servers()

            self.console.print("[green][+] MCP client initialized successfully[/green]")
            return True

        except Exception as e:
            self.console.print(
                f"[red][!] Failed to initialize MCP client: {str(e)}[/red]"
            )
            self.logger.error(f"Client initialization failed: {e}", exc_info=True)
            return False

    async def chat_loop(self) -> None:
        """Main chat loop"""
        self.session_stats["start_time"] = time.time()

        self.console.print(
            "Type your queries or use commands (type [yellow]/help[/yellow] for help)"
        )
        self.console.print(
            "Press [yellow]Ctrl+C[/yellow] or type [yellow]/quit[/yellow] to exit"
        )
        self.console.print()

        while True:
            try:
                # Get user input
                query = Prompt.ask("[bold cyan]Query[/bold cyan]", default="").strip()

                if not query:
                    continue

                # Handle commands
                if query.startswith("/"):
                    should_continue = await self.process_command(query)
                    if not should_continue:
                        break
                else:
                    # Process regular query
                    await self.process_query(query)

                self.console.print()

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]Received interrupt signal. Exiting...[/yellow]"
                )
                break
            except EOFError:
                self.console.print("\n[yellow]EOF received. Exiting...[/yellow]")
                break
            except Exception as e:
                self.console.print(
                    f"[red]Unexpected error in chat loop: {str(e)}[/red]"
                )
                self.logger.error(f"Chat loop error: {e}", exc_info=True)

    async def run(self) -> None:
        """Main entry point for the chatbot"""
        try:
            # Display banner
            self.display_banner()

            # Initialize client
            if not await self.initialize_client():
                return

            # Display server info
            self.display_server_info()

            # Start chat loop
            await self.chat_loop()

        finally:
            # Cleanup
            if self.client:
                self.console.print("[yellow]Cleaning up connections...[/yellow]")
                await self.client.cleanup()

            # Display final stats
            self.console.print()
            self.display_stats()

            self.console.print("[green]Thank you for using Global MCP Client![/green]")


async def main():
    """Main function to run the chatbot"""
    try:
        # Load configuration
        config = Config("configs")

        # Create and run chatbot
        chatbot = GlobalMCPChatBot(config)
        await chatbot.run()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
