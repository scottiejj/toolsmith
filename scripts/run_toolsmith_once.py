import os
import sys
import argparse

# Ensure repo root on path
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(ROOT, '..'))
sys.path.append(ROOT)

from multi_agents.state import State
from multi_agents.agents.toolsmith import Toolsmith
from utils import PREFIX_MULTI_AGENTS


def main():
    parser = argparse.ArgumentParser(description='Run Toolsmith once for a given phase and competition.')
    parser.add_argument('--competition', type=str, default='titanic', help='Competition name (folder under multi_agents/competition)')
    parser.add_argument('--phase', type=str, default='Data Cleaning', help='Phase name as in config.json')
    args = parser.parse_args()

    # Prepare state
    state = State(phase=args.phase, competition=args.competition)
    state.make_dir()
    state.make_context()

    agent = Toolsmith('gpt-4o-mini', 'api')
    result = agent.action(state)

    doc_dir = os.path.join(PREFIX_MULTI_AGENTS, 'tools', 'ml_tools_doc')
    out_path = os.path.join(doc_dir, f"{state.dir_name}_tools.md")
    if os.path.exists(out_path):
        print(f"OK: Wrote markdown to {out_path}")
    else:
        print(f"ERROR: Expected markdown not found at {out_path}")

    # Verify python file
    py_path = os.path.join(PREFIX_MULTI_AGENTS, 'tools', 'generated', f"{state.dir_name}.py")
    if os.path.exists(py_path):
        print(f"OK: Wrote python stubs to {py_path}")
    else:
        print(f"ERROR: Expected python stubs not found at {py_path}")


if __name__ == '__main__':
    main()
