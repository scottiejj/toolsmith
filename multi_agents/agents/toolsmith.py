import json
import logging
import os
import re
import sys
from typing import Dict, Any, List, Tuple

# Make local imports work when running as a script
sys.path.append('..')
sys.path.append('../..')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_base import Agent
from utils import read_file, PREFIX_MULTI_AGENTS
from state import State
from prompts.prompt_toolsmith import (
    PROMPT_TOOLSMITH_TASK,
    PROMPT_TOOLSMITH_MARKDOWN_FORMAT,
    PROMPT_TOOLSMITH_CONSTRAINTS,
    PROMPT_TOOLSMITH_PHASE_GUIDELINES,
    PROMPT_TOOLSMITH_CODE_MODULE,
    PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Toolsmith(Agent):
    """
    Toolsmith agent (code-first workflow):
    - Reads Reader output (competition_info.txt) as background
    - Generates a complete Python module of tools for the current phase
    - Saves module under tools/generated so Planner/Developer can import it
    - Converts the module into phase tool docs (markdown) for RAG
    - Updates config.json phase_to_ml_tools with extracted function names

    Keep it simple for easy debugging; add comments at key steps.
    """

    def __init__(self, model: str, type: str):
        super().__init__(
            role="toolsmith",
            description="You design phase-appropriate tools and their documentation for Planner/Developer to use.",
            model=model,
            type=type,
        )

    # --------------- helpers --------------- #
    def _load_background(self, state: State) -> str:
        """Read Reader's summary file produced earlier; fallback to empty text."""
        path = f"{state.competition_dir}/data_preview.txt"
        if os.path.exists(path):
            try:
                return read_file(path)
            except Exception:
                return ""
        return ""

    def _render_code_prompt(self, state: State, background: str) -> str:
        try:
            constraints = PROMPT_TOOLSMITH_CONSTRAINTS.format(phase_name=state.phase)
        except Exception:
            constraints = PROMPT_TOOLSMITH_CONSTRAINTS
        task = PROMPT_TOOLSMITH_TASK.format(phase_name=state.phase)
        guidelines = PROMPT_TOOLSMITH_PHASE_GUIDELINES
        return (
            f"# TASK\n{task}\n\n# CONSTRAINTS\n{constraints}\n\n# GUIDELINES\n{guidelines}\n\n"
            + PROMPT_TOOLSMITH_CODE_MODULE.format(
                phase_name=state.phase,
                competition=state.competition,
                background=background,
            )
        )

    def _render_markdown_from_code_prompt(self, python_source: str) -> str:
        return PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE.format(
            markdown_format_spec=PROMPT_TOOLSMITH_MARKDOWN_FORMAT,
            python_source=python_source,
        )

    def _extract_markdown_from_reply(self, raw_reply: str) -> Tuple[str, str]:
        """Extract markdown text and category from model reply.

        Supports two formats:
        - JSON wrapper: {"category": "...", "markdown": "..."}
        - Fenced markdown block: ```markdown ... ```
        Returns (markdown_text, category_or_empty).
        """
        category = ""
        text = raw_reply.strip()
        # Try JSON first
        if text.startswith("{"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "markdown" in obj:
                    category = obj.get("category", "") or ""
                    return str(obj["markdown"]), category
            except Exception:
                pass
        # Fallback to fenced markdown parsing via base helper
        return self._parse_markdown(raw_reply), category

    def _extract_python_from_reply(self, raw_reply: str) -> str:
        """Extract the python module from a reply. Prefer fenced python block; else return raw."""
        text = raw_reply.strip()
        m = re.search(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
        if m:
            return m.group(1)
        # Also try generic triple backticks
        m = re.search(r"```\n(.*?)\n```", text, flags=re.DOTALL)
        if m:
            return m.group(1)
        return text

    def _save_python_module(self, state: State, python_source: str) -> str:
        gen_dir = f"{PREFIX_MULTI_AGENTS}/tools/generated"
        os.makedirs(gen_dir, exist_ok=True)
        out_path = f"{gen_dir}/{state.dir_name}.py"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(python_source.strip() + "\n")
        return out_path

    def _extract_function_names_from_python(self, python_source: str) -> List[str]:
        try:
            import ast
            tree = ast.parse(python_source)
            names = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
            return names
        except Exception:
            # Fallback regex
            return re.findall(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", python_source, flags=re.MULTILINE)

    def _save_markdown(self, state: State, markdown_text: str) -> str:
        """Write merged tool markdown to ml_tools_doc/<phase_dir>_tools.md."""
        doc_dir = f"{PREFIX_MULTI_AGENTS}/tools/ml_tools_doc"
        os.makedirs(doc_dir, exist_ok=True)
        out_path = f"{doc_dir}/{state.dir_name}_tools.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        return out_path

    # Legacy markdown-first helpers removed in code-first flow

    def _update_config_tools(self, state: State, tool_names: List[str]) -> None:
        """Overwrite the tools list for the current phase in config.json."""
        cfg_path = f"{PREFIX_MULTI_AGENTS}/config.json"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            logger.info("Could not read config.json; skipping config update.")
            return

        phase = state.phase
        if "phase_to_ml_tools" not in cfg:
            cfg["phase_to_ml_tools"] = {}
        # Overwrite with provided tool_names (order preserved as given)
        cfg["phase_to_ml_tools"][phase] = list(tool_names)

        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4)
        except Exception:
            logger.info("Could not write config.json; skipping config update.")

    # --------------- core --------------- #
    def _execute(self, state: State, role_prompt: str) -> Dict[str, Any]:
        """
        Code-first execution:
        - Prepare system/user messages
        - Generate a full Python module of tools
        - Save module and extract function names
        - Convert the module to markdown and save
        - Update config with function names
        - Return memory payload for this agent
        """
        history: List[Dict[str, str]] = []

        # Gather context
        background = self._load_background(state)

        # System/user primer (match style of other agents)
        if self.model == 'gpt-4o':
            history.append({"role": "system", "content": f"{role_prompt}{self.description}"})
        else:
            history.append({"role": "user", "content": f"{role_prompt}{self.description}"})

        # 1) Generate Python module first
        code_prompt = self._render_code_prompt(state, background)
        raw_code_reply, history = self.llm.generate(code_prompt, history, max_completion_tokens=4096)
        python_source = self._extract_python_from_reply(raw_code_reply)
        py_path = self._save_python_module(state, python_source)

        # Extract tool names from function definitions
        tool_names = self._extract_function_names_from_python(python_source)

        # 2) Convert Python source to Markdown docs
        md_prompt = self._render_markdown_from_code_prompt(python_source)
        raw_md_reply, history = self.llm.generate(md_prompt, history, max_completion_tokens=4096)
        # Ensure fenced markdown to satisfy agent_base._parse_markdown expectations
        fenced_md = raw_md_reply.strip()
        if '```markdown' not in fenced_md:
            fenced_md = f"```markdown\n{fenced_md}\n```"
        markdown_text, _ = self._extract_markdown_from_reply(fenced_md)
        md_path = self._save_markdown(state, markdown_text)

        # Update config with discovered tool names
        if tool_names:
            self._update_config_tools(state, tool_names)

        # Persist agent history for debugging
        with open(f"{state.restore_dir}/{self.role}_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        input_used_in_review = (
            f"   <background_info>\n{background}\n    </background_info>\n"
            f"   <tools_markdown_path>\n{md_path}\n    </tools_markdown_path>"
        )

        print(f"State {state.phase} - Agent {self.role} finishes working.")

        # Minimal result payload (planner/developer don't read this directly; docs+config are primary integration)
        return {
            self.role: {
                "history": history,
                "role": self.role,
                "description": self.description,
                "task": PROMPT_TOOLSMITH_TASK.format(phase_name=state.phase),
                "input": input_used_in_review,
                "tool_names": tool_names,
                "markdown_path": md_path,
                "python_path": py_path,
                "category": state.phase,
                "result": markdown_text,
            }
        }
