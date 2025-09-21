"""
Command-line interface for Global MCP Client
"""

import click
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .core import Config, GlobalMCPClient, setup_logging
from .chatbot import GlobalMCPChatBot
from .utils import validate_environment, get_system_info
from .dynamic_orchestrator import UniversalAutoProcessor


@click.group()
@click.version_option(version="1.0.0", prog_name="Global MCP Client")
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration directory path"
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Set log level",
)
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool, log_level: Optional[str]):
    """Global MCP Client - Connect to multiple MCP servers with AI chat interface"""

    # Ensure object exists
    ctx.ensure_object(dict)

    # Store common options
    ctx.obj["config_dir"] = config
    ctx.obj["debug"] = debug
    ctx.obj["log_level"] = log_level


@cli.command()
@click.pass_context
def chat(ctx):
    """Start the interactive chat interface"""
    try:
        # Load configuration
        config = Config(config_dir=ctx.obj.get("config_dir"))

        # Override log level if specified
        if ctx.obj.get("log_level"):
            import os

            os.environ["LOG_LEVEL"] = ctx.obj["log_level"]

        if ctx.obj.get("debug"):
            import os

            os.environ["DEBUG"] = "true"

        # Start chatbot
        chatbot = GlobalMCPChatBot(config)
        asyncio.run(chatbot.run())

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get("debug"):
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate the environment and configuration"""
    click.echo("[*] Validating Global MCP Client environment...")

    # Validate environment
    issues = validate_environment()
    if issues:
        click.echo("[!] Environment validation failed:")
        for issue in issues:
            click.echo(f"   - {issue}")
    else:
        click.echo("[+] Environment validation passed")

    # Validate configuration
    try:
        config = Config(config_dir=ctx.obj.get("config_dir"))
        config_issues = config.validate()

        if config_issues:
            click.echo("[!] Configuration validation failed:")
            for issue in config_issues:
                click.echo(f"   - {issue}")
        else:
            click.echo("[+] Configuration validation passed")

        # Show server info
        enabled_servers = config.get_enabled_servers()
        click.echo(f"[i] Found {len(enabled_servers)} enabled servers:")
        for name, server_config in enabled_servers.items():
            click.echo(f"   - {name}: {server_config.description}")

    except Exception as e:
        click.echo(f"[!] Configuration error: {e}")
        if ctx.obj.get("debug"):
            import traceback

            traceback.print_exc()


@cli.command()
@click.option("--server", "-s", help="Test specific server only")
@click.pass_context
def test(ctx, server: Optional[str]):
    """Test connections to MCP servers"""

    async def test_connections():
        try:
            config = Config(config_dir=ctx.obj.get("config_dir"))

            click.echo("[*] Testing MCP server connections...")

            client = GlobalMCPClient(config)

            if server:
                # Test specific server
                if server not in config.mcp_servers:
                    click.echo(f"[!] Server '{server}' not found in configuration")
                    return

                server_config = config.mcp_servers[server]
                if not server_config.enabled:
                    click.echo(f"[!] Server '{server}' is disabled")
                    return

                try:
                    connection = await client.connect_to_server(server, server_config)
                    click.echo(f"[+] Successfully connected to '{server}'")
                    click.echo(
                        f"   Tools: {[tool['name'] for tool in connection.tools]}"
                    )
                except Exception as e:
                    click.echo(f"[!] Failed to connect to '{server}': {e}")
            else:
                # Test all servers
                try:
                    await client.connect_to_all_servers()

                    for name, connection in client.connections.items():
                        click.echo(f"[+] {name}: {len(connection.tools)} tools")

                    # Perform health check
                    health = await client.health_check()
                    click.echo(
                        f"[i] Health check: {health['healthy_servers']}/{health['total_servers']} servers healthy"
                    )

                except Exception as e:
                    click.echo(f"[!] Connection test failed: {e}")
                    if ctx.obj.get("debug"):
                        import traceback

                        traceback.print_exc()

            await client.cleanup()

        except Exception as e:
            click.echo(f"[!] Test failed: {e}")
            if ctx.obj.get("debug"):
                import traceback

                traceback.print_exc()

    asyncio.run(test_connections())


@cli.command()
@click.pass_context
def info(ctx):
    """Show system and configuration information"""

    click.echo("[i] Global MCP Client Information")
    click.echo("=" * 40)

    # System info
    sys_info = get_system_info()
    click.echo("[+] System Information:")
    click.echo(f"   Platform: {sys_info['platform']} {sys_info['platform_release']}")
    click.echo(f"   Architecture: {sys_info['architecture']}")
    click.echo(f"   Python: {sys_info['python_version']}")
    click.echo(f"   CPU Cores: {sys_info['cpu_count']}")
    click.echo(
        f"   Memory: {sys_info['memory_available'] // (1024**3):.1f}GB available"
    )

    # Configuration info
    try:
        config = Config(config_dir=ctx.obj.get("config_dir"))

        click.echo("\n[i] Configuration:")
        click.echo(f"   Config Directory: {config.config_dir}")
        click.echo(f"   Log Level: {config.log_level}")
        click.echo(f"   Default Model: {config.default_model}")
        click.echo(f"   Debug Mode: {config.debug}")

        enabled_servers = config.get_enabled_servers()
        click.echo(f"\n[+] MCP Servers ({len(enabled_servers)} enabled):")
        for name, server_config in enabled_servers.items():
            click.echo(
                f"   - {name}: {server_config.command} {' '.join(server_config.args)}"
            )

    except Exception as e:
        click.echo(f"\n[!] Configuration error: {e}")


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for configuration")
@click.pass_context
def config(ctx, output: Optional[str]):
    """Show or export current configuration"""

    try:
        config = Config(config_dir=ctx.obj.get("config_dir"))

        config_data = {
            "mcpServers": {
                name: server_config.dict()
                for name, server_config in config.mcp_servers.items()
            },
            "global_settings": config.global_settings.dict(),
        }

        if output:
            # Export to file
            import json

            with open(output, "w") as f:
                json.dump(config_data, f, indent=2)
            click.echo(f"[+] Configuration exported to {output}")
        else:
            # Display configuration
            import json

            click.echo(json.dumps(config_data, indent=2))

    except Exception as e:
        click.echo(f"[!] Error: {e}")
        if ctx.obj.get("debug"):
            import traceback

            traceback.print_exc()


@cli.command()
@click.argument("query")
@click.option("--timeout", "-t", type=int, default=60, help="Query timeout in seconds")
@click.pass_context
def query(ctx, query: str, timeout: int):
    """Execute a single query without interactive mode"""

    async def execute_query():
        try:
            config = Config(config_dir=ctx.obj.get("config_dir"))

            async with GlobalMCPClient(config) as client:
                response = await asyncio.wait_for(
                    client.process_query(query), timeout=timeout
                )
                click.echo(response)

        except asyncio.TimeoutError:
            click.echo(f"[!] Query timed out after {timeout} seconds", err=True)
        except Exception as e:
            click.echo(f"[!] Error: {e}", err=True)
            if ctx.obj.get("debug"):
                import traceback

                traceback.print_exc()

    asyncio.run(execute_query())


@cli.command(name="analyze")
@click.argument("request")
@click.option("--schema", "-s", required=True, help="Database schema name to analyze")
@click.option("--timeout", "-t", type=int, default=120, help="Timeout in seconds")
@click.pass_context
def analyze(ctx, request: str, schema: str, timeout: int):
    """Run a dynamic, multi-query analysis against a schema and return a dashboard-style report"""

    async def run_analysis():
        try:
            config = Config(config_dir=ctx.obj.get("config_dir"))
            async with GlobalMCPClient(config) as client:
                processor = UniversalAutoProcessor(client)
                result = await asyncio.wait_for(
                    processor.process_request(request, schema_name=schema),
                    timeout=timeout,
                )
                click.echo(result)
        except asyncio.TimeoutError:
            click.echo(f"[!] Analysis timed out after {timeout} seconds", err=True)
        except Exception as e:
            click.echo(f"[!] Error: {e}", err=True)
            if ctx.obj.get("debug"):
                import traceback
                traceback.print_exc()

    asyncio.run(run_analysis())


if __name__ == "__main__":
    cli()
