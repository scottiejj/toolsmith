import json
import logging
import os
import re
import sys
from typing import Dict, Any, List, Tuple
import ast

# Make local imports work when running as a script
sys.path.append("..")
sys.path.append("../..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_base import Agent
from utils import read_file, PREFIX_MULTI_AGENTS
from state import State
from prompts.prompt_toolsmith import (
    PROMPT_TOOLSMITH_TASK,
    PROMPT_TOOLSMITH_CONSTRAINTS,
    PHASE_TO_PHASE_GUIDELINES,
    PROMPT_TOOLSMITH_CODE_MODULE,
    PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE,
    PHASE_TO_PHASE_GUIDELINES
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Toolsmith(Agent):
    """
    Toolsmith agent (code-first workflow):
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

    def _get_previous_plan_and_report(self, state: State):
        previous_plan = ""
        previous_phases = state.get_previous_phase(type="plan")
        for previous_phase in previous_phases:
            previous_dir_name = state.phase_to_directory[previous_phase]
            previous_plan += f"## {previous_phase.upper()} ##\n"
            path_to_previous_plan = (
                f"{state.competition_dir}/{previous_dir_name}/markdown_plan.txt"
            )
            if os.path.exists(path_to_previous_plan):
                with open(path_to_previous_plan, "r") as f:
                    previous_plan += f"{previous_phase} Plan\n"
                    previous_plan += f.read()
                    previous_plan += "\n"
            else:
                previous_plan += "There is no plan in this phase.\n"
        path_to_previous_report = (
            f"{state.competition_dir}/{previous_dir_name}/report.txt"
        )
        previous_report = ""
        if os.path.exists(path_to_previous_report):
            with open(path_to_previous_report, "r") as f:
                previous_report += f"{previous_phase} Report\n"
                previous_report += f.read()
        else:
            previous_report += "There is no report in the previous phase.\n"
        return previous_plan, previous_report

    def _render_code_prompt(self, state: State) -> str:
        # Round 1 only: provide CURRENT phase docs requested and ask for code module
        previous_plan, previous_report = self._get_previous_plan_and_report(state)
        available_features = self._read_data(state, num_lines=1)

        phase_specific_instructions = PHASE_TO_PHASE_GUIDELINES.get(state.phase, "")

        return PROMPT_TOOLSMITH_CODE_MODULE.format(
            phase_name=state.phase,
            competition=state.competition,
            data_preview=state.background_info,
            available_features=available_features,
            previous_plans_report=previous_plan + "\n" + previous_report,
            phase_specific_instructions=phase_specific_instructions,
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

    def _is_valid_python_module(self, source: str) -> bool:
        """Basic validation: ensure module parses and has non-trivial length."""
        if not source or len(source.strip()) < 200:
            return False
        try:
            ast.parse(source)
            return True
        except Exception:
            return False

    def _phase_output_dir(self, state: State) -> str:
        # e.g., multi_agents/competition/titanic/data_cleaning/data_cleaning_generated_tools
        phase_dir = state.phase_to_directory.get(state.phase, "")
        out_dir = os.path.join(
            PREFIX_MULTI_AGENTS,
            "competition",
            state.competition,
            phase_dir,
            f"{phase_dir}_generated_tools",
        )
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_python_module(self, state: State, python_source: str) -> str:
        out_dir = self._phase_output_dir(state)
        phase_dir = state.phase_to_directory.get(state.phase, "")
        py_path = os.path.join(out_dir, f"{phase_dir}.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(python_source)
        logger.info(f"Wrote Python tools to {py_path}")
        return py_path

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
            return re.findall(
                r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                python_source,
                flags=re.MULTILINE,
            )

    def _save_markdown(self, state: State, markdown_text: str) -> str:
        out_dir = self._phase_output_dir(state)
        phase_dir = state.phase_to_directory.get(state.phase, "")
        md_path = os.path.join(out_dir, f"{phase_dir}_tools.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info(f"Wrote tool docs to {md_path}")
        return md_path

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
        - Round 0: prime with role and context
        - Round 1: generate a full Python module of tools using background + data preview
        - Save module and extract function names
        - Round 2: convert the module to markdown and save
        - Update config with function names
        - Return memory payload for this agent
        """
        history: List[Dict[str, str]] = []
        data_preview = self._data_preview(state, num_lines=11)
        background_info = f"Data preview:\n{data_preview}"
        state.set_background_info(background_info)
        input_used_in_review = (
            f"   <background_info>\n{background_info}\n    </background_info>"
        )

        # If not the first execution, reuse previous Toolsmith output
        if len(state.memory) > 1:
            last_state = state.memory[-2]
            if "toolsmith" in last_state:
                return {"toolsmith": last_state["toolsmith"]}

        # System/user primer (match style of other agents)
        if self.model in ["gpt-4o", "gpt-4.1"]:
            history.append(
                {"role": "system", "content": f"{role_prompt}{self.description}"}
            )
        elif self.model == "o1-mini":
            history.append(
                {"role": "user", "content": f"{role_prompt}{self.description}"}
            )

        # Round 0: Prime with task, constraints, guidelines, and initial dataset background
        # Ask the model to request CURRENT phase docs (available_features, data_preview, previous_plans_report)
        task_text = PROMPT_TOOLSMITH_TASK.format(phase_name=state.phase)
        constraints_text = PROMPT_TOOLSMITH_CONSTRAINTS.format(phase_name=state.phase)

        with open(f"{state.competition_dir}/competition_info.txt", "r") as f:
            try:
                initial_bg = f.read()
            except Exception:
                initial_bg = ""

        round0_input = (
            "# INITIAL DATASET BACKGROUND\n"
            "Your ultimate goal is to design tools for the dataset described below (initial background):\n"
            f"{initial_bg}\n\n"
            f"# TASK\n{task_text}\n\n# CONSTRAINTS\n{constraints_text}\n\n"
            "#############\n# START TOOL DESIGN #\n"
            "Before you begin, please request the following information from me, which contain important information that will guide your tool design for current phase:\n"
            "0. CURRENT phase specific tool GUIDELINES\n"
            "1. CURRENT phase available features\n"
            "2. CURRENT phase data preview\n"
            "3. Previous phases' plans and reports\n"
        )
        _, history = self.llm.generate(
            round0_input, history, max_completion_tokens=8192
        )

        # Round 1: Generate Python module
        # Round 1: Provide requested CURRENT phase docs and generate the Python module
        code_prompt = self._render_code_prompt(state)
        raw_code_reply, history = self.llm.generate(
            code_prompt, history, max_completion_tokens=8192
        )
        python_source = self._extract_python_from_reply(raw_code_reply)

        # Validate extracted code and retry once if invalid/too short
        code_valid = self._is_valid_python_module(python_source)
        while not code_valid:
            logger.info("Initial tool module invalid or too short; retrying generation with clarification.")
            retry_prompt = (
                "\n\nThe previous module was invalid or incomplete due to token constraints. DECREASE the number of tools generated but ENSURE a COMPLETE, self-contained Python module with import statements, functions, and no truncation."
                +code_prompt
            )
            raw_code_reply, history = self.llm.generate(
                retry_prompt, history, max_completion_tokens=4096
            )
            python_source = self._extract_python_from_reply(raw_code_reply)
            code_valid = self._is_valid_python_module(python_source)

        py_path = self._save_python_module(state, python_source)

        # Extract tool names from function definitions
        tool_names = self._extract_function_names_from_python(python_source)

        # Round 2: Convert Python source to Markdown docs
        raw_md_reply, history = self.llm.generate(
            PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE, history, max_completion_tokens=4096
        )

        # Ensure fenced markdown to satisfy agent_base._parse_markdown expectations
        fenced_md = raw_md_reply.strip()
        if "```markdown" not in fenced_md:
            fenced_md = f"```markdown\n{fenced_md}\n```"
        markdown_text, _ = self._extract_markdown_from_reply(fenced_md)
        md_path = self._save_markdown(state, markdown_text)

        # Update config with discovered tool names
        if tool_names:
            self._update_config_tools(state, tool_names)

        # Persist agent history for debugging
        with open(
            f"{state.restore_dir}/{self.role}_history.json", "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, indent=4)

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
                "result": markdown_text,
            }
        }
