"""Slash command framework and built-in commands.

Extensible: register additional ``SlashCommand`` instances on the registry
(or pass a custom registry to ``Repl``/``launch``) without touching the
REPL loop.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    # Clearing starts a NEW archive; the old one stays on disk (/resume).
    repl.context.restore([])
    repl.store.start(model=repl.settings.model)
    repl.session.resumed_from = None
    print(f"context cleared; new session {repl.store.current_id}.")
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


def _cmd_sessions(repl: Repl, arg: str) -> bool:
    sessions = repl.store.list_sessions()
    if not sessions:
        print("  no saved sessions.")
        return False
    current = repl.store.current_id
    print(f"  {'ID':24s} {'model':16s} {'msgs':>5s}  updated (UTC)")
    for s in sessions[:20]:
        mark = "*" if s.id == current else " "
        updated = s.updated_at[:19].replace("T", " ")
        print(f"{mark} {s.id:24s} {s.model:16s} {s.messages:5d}  {updated}")
    return False


def _cmd_resume(repl: Repl, arg: str) -> bool:
    if not arg:
        print("usage: /resume <session-id>  (see /sessions)")
        return False
    try:
        messages = repl.store.load(arg)
    except FileNotFoundError:
        print(f"no session {arg!r} — see /sessions")
        return False
    repl.context.restore(messages)
    repl.store.adopt(arg)
    repl.session.resumed_from = arg
    print(f"resumed {arg} ({len(messages)} messages). /clear starts a new session.")
    repl.show_history()
    return False


def _cmd_new(repl: Repl, arg: str) -> bool:
    repl.context.restore([])
    repl.store.start(model=repl.settings.model)
    repl.session.resumed_from = None
    print(f"new session {repl.store.current_id}.")
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
            names=("/sessions",),
            description="list saved sessions",
            handler=_cmd_sessions,
        ),
        SlashCommand(
            names=("/resume",),
            description="continue a saved session",
            usage="<ID>",
            handler=_cmd_resume,
        ),
        SlashCommand(
            names=("/new",),
            description="start a new session (archive kept)",
            handler=_cmd_new,
        ),
        SlashCommand(
            names=("/quit", "/exit", "/q"),
            description="exit Cyent (Ctrl+C twice also works)",
            hidden=True,  # aliases would duplicate the listing
            handler=_cmd_quit,
        ),
    ]
