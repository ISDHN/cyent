"""Slash command framework and built-in commands.

Extensible: register additional ``SlashCommand`` instances on the registry
(or pass a custom registry to ``Repl``/``launch``) without touching the
REPL loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cyent.cli.prompts import build_system_prompt
from cyent.core.context import ContextManager

if TYPE_CHECKING:  # annotations only; runtime import would be circular
    from cyent.cli.repl import Repl


@dataclass(slots=True)
class SlashCommand:
    """One registered slash command.

    ``handler(repl, arg) -> bool``: return True to exit the REPL.
    ``names``: primary name first (e.g. ("/quit", "/exit", "/q")).
    ``hidden``: exclude from /help listing (e.g. aliases).
    """

    names: tuple[str, ...]
    description: str
    usage: str = ""
    hidden: bool = False
    handler: Callable[[Repl, str], bool] = lambda repl, arg: False

    @property
    def primary(self) -> str:
        return self.names[0]

    def help_line(self) -> str:
        usage = f" {self.usage}" if self.usage else ""
        return f"  {self.primary}{usage:14s} {self.description}"


class CommandRegistry:
    """Registry of slash commands: lookup, dispatch, help rendering."""

    def __init__(self, commands: list[SlashCommand] | None = None) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._order: list[SlashCommand] = []
        for cmd in commands or []:
            self.register(cmd)

    def register(self, command: SlashCommand) -> None:
        """Register a command; later registrations override earlier aliases."""
        self._order.append(command)
        for name in command.names:
            self._commands[name.lower()] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name.lower())

    def all_commands(self) -> list[SlashCommand]:
        """Unique commands in registration order."""
        seen: set[int] = set()
        out: list[SlashCommand] = []
        for cmd in self._order:
            if id(cmd) not in seen:
                seen.add(id(cmd))
                out.append(cmd)
        return out

    def dispatch(self, repl: Repl, line: str) -> bool:
        """Execute ``line`` (a '/...' command). Returns True to exit REPL."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        command = self.get(cmd)
        if command is None:
            print(f"unknown command {cmd!r}. Try /help")
            return False
        return command.handler(repl, arg)

    def help_text(self) -> str:
        lines = ["Slash commands:"]
        for cmd in self.all_commands():
            if not cmd.hidden:
                lines.append(cmd.help_line())
        lines.append("Anything else is sent to the agent as a task.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Built-in command handlers
# --------------------------------------------------------------------------- #
def _cmd_help(repl: Repl, arg: str) -> bool:
    print(repl.commands.help_text())
    return False


def _cmd_quit(repl: Repl, arg: str) -> bool:
    return True


def _cmd_model(repl: Repl, arg: str) -> bool:
    if arg:
        repl.settings.model = arg
        print(f"model switched to {arg}")
    else:
        print(f"current model: {repl.settings.model}")
    return False


def _cmd_clear(repl: Repl, arg: str) -> bool:
    repl.context = ContextManager(system_prompt=build_system_prompt())
    repl.engine.context = repl.context
    print("context cleared.")
    return False


def _cmd_tools(repl: Repl, arg: str) -> bool:
    for name in repl.executor.tool_names:
        tool = repl.executor.get(name)
        desc = (tool.description or "").splitlines()[0] if tool else ""
        print(f"  {name:14s} {desc}")
    return False


def _cmd_stats(repl: Repl, arg: str) -> bool:
    st = repl.context.stats_snapshot()
    es = repl.engine.stats
    print(
        f"  messages: {st.messages} | ~tokens: {st.approx_tokens} "
        f"| budget: {repl.context._budget} | trims: {st.trims} | summaries: {st.summaries}\n"
        f"  last run: rounds={es.rounds}, tool_calls={es.tool_calls}, stop={es.stop_reason}"
    )
    return False


def default_commands() -> list[SlashCommand]:
    """The built-in slash commands, in help order."""
    return [
        SlashCommand(
            names=("/help", "/?"),
            description="show this help",
            handler=_cmd_help,
        ),
        SlashCommand(
            names=("/model",),
            description="show or switch the model",
            usage="[NAME]",
            handler=_cmd_model,
        ),
        SlashCommand(
            names=("/clear",),
            description="clear conversation context",
            handler=_cmd_clear,
        ),
        SlashCommand(
            names=("/tools",),
            description="list available tools",
            handler=_cmd_tools,
        ),
        SlashCommand(
            names=("/stats",),
            description="show context/token statistics",
            handler=_cmd_stats,
        ),
        SlashCommand(
            names=("/quit", "/exit", "/q"),
            description="exit Cyent (Ctrl+C twice also works)",
            hidden=True,  # aliases would duplicate the listing
            handler=_cmd_quit,
        ),
    ]
