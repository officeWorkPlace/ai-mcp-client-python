"""
Enhanced chatbot interface for MCP Client
"""

import asyncio
import sys
import time
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Import from new structure
from ..core import GlobalMCPClient, Config, setup_logging
from ..core.exceptions import GlobalMCPClientError
from ..utils.validators import InputValidator
from ..utils.rate_limiter import RateLimiter


class MCPChatBot:
    """Modern chatbot interface for MCP Client"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.console = Console()
        self.client: Optional[GlobalMCPClient] = None
        self.validator = InputValidator()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

        # Setup logging
        self.logger = setup_logging(
            log_level="WARNING",
            log_file=self.config.log_file,
            enable_file_logging=self.config.enable_file_logging,
            enable_console_logging=False,
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
        banner.append("[*] MCP Client - Chatbot Interface", style="bold blue")
        banner.append("\n")
        banner.append("Interactive chat with multi-server MCP support", style="dim")

        panel = Panel(
            banner,
            title="[bold green]Welcome[/bold green]",
            border_style="blue",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    async def initialize_client(self) -> bool:
        """Initialize the MCP client"""
        try:
            with self.console.status("[bold yellow]Initializing MCP client..."):
                self.client = GlobalMCPClient(self.config)
                await self.client.connect_to_all_servers()

            self.console.print("[green][+] MCP client initialized successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red][!] Failed to initialize MCP client: {str(e)}[/red]")
            self.logger.error(f"Client initialization failed: {e}", exc_info=True)
            return False

    async def process_query(self, query: str) -> None:
        """Process user query"""
        if not self.client:
            self.console.print("[red]No client connection available[/red]")
            return

        # Validate and rate limit
        if not self.validator.validate_query(query):
            self.console.print("[red]Invalid query. Please try again.[/red]")
            return

        if not self.rate_limiter.allow_request():
            self.console.print("[red]Rate limit exceeded. Please wait.[/red]")
            return

        self.session_stats["queries_processed"] += 1

        try:
            with self.console.status("[bold green]Processing query..."):
                response = await self.client.process_query(query)

            response_panel = Panel(
                response,
                title="[bold green]Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )

            self.console.print(response_panel)

        except Exception as e:
            self.session_stats["errors"] += 1
            self.console.print(f"[red]Error: {str(e)}[/red]")
            self.logger.error(f"Query processing error: {e}")

    async def chat_loop(self) -> None:
        """Main chat loop"""
        self.session_stats["start_time"] = time.time()

        self.console.print("Type your queries or commands (type [yellow]/help[/yellow] for help)")
        self.console.print("Press [yellow]Ctrl+C[/yellow] or type [yellow]/quit[/yellow] to exit\n")

        while True:
            try:
                query = Prompt.ask("[bold cyan]Query[/bold cyan]", default="").strip()

                if not query:
                    continue

                if query.lower() in ["/quit", "/q", "/exit"]:
                    break
                elif query.lower() in ["/help", "/h"]:
                    self.console.print("[cyan]Available commands: /help, /quit[/cyan]")
                else:
                    await self.process_query(query)

                self.console.print()

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {str(e)}[/red]")
                self.logger.error(f"Chat loop error: {e}", exc_info=True)

    async def run(self) -> None:
        """Main entry point"""
        try:
            self.display_banner()

            if not await self.initialize_client():
                return

            await self.chat_loop()

        finally:
            if self.client:
                await self.client.cleanup()

            self.console.print("[green]Thank you for using MCP Client![/green]")


async def main():
    """Main function to run the chatbot"""
    config = Config()
    chatbot = MCPChatBot(config)
    await chatbot.run()


def cli_main():
    """Synchronous entry point for CLI"""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()