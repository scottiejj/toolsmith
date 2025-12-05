import os
import sys
import importlib.util
import sys as _sys
import subprocess


def write_smoke_runner(competition: str, phase_dir: str, restore_dir: str) -> str:
    run_path = os.path.join(restore_dir, f"{phase_dir}_smoke_run.py")
    header = (
        "import sys\n"
        "import os\n"
        "import importlib.util\n"
        "import sys as _sys\n\n"
        "sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents','multi_agents/tools', 'multi_agents/prompts'])\n"
        "sys.path.append(os.path.dirname(os.path.abspath(__file__)))\n\n"
        f"phase_module_path = os.path.join('multi_agents','competition','{competition}','{phase_dir}','{phase_dir}_generated_tools','{phase_dir}.py')\n"
        "spec = importlib.util.spec_from_file_location('phase_tools', phase_module_path)\n"
        "phase_tools = importlib.util.module_from_spec(spec)\n"
        "_sys.modules['phase_tools'] = phase_tools\n"
        "spec.loader.exec_module(phase_tools)\n"
        "from phase_tools import *\n\n"
    )
    body = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "def _tiny_df():\n"
        "    return pd.DataFrame({'num':[1,2,3],'cat':['a','b','a'],'id':[10,11,12]})\n\n"
        "def _load_phase_subset():\n"
        "    base = os.path.join('multi_agents','competition','" + competition + "')\n"
        "    try:\n"
        "        # Determine phase-dependent files\n"
        "        if 'feature_engineering' in '" + phase_dir + "':\n"
        "            train_path = os.path.join(base, 'cleaned_train.csv')\n"
        "            test_path = os.path.join(base, 'cleaned_test.csv')\n"
        "        elif 'model_building' in '" + phase_dir + "' or 'Model' in '" + phase_dir + "':\n"
        "            train_path = os.path.join(base, 'processed_train.csv')\n"
        "            test_path = os.path.join(base, 'processed_test.csv')\n"
        "        else:\n"
        "            train_path = os.path.join(base, 'train.csv')\n"
        "            test_path = os.path.join(base, 'test.csv')\n"
        "        # Prefer train subset; fallback to synthetic\n"
        "        if os.path.exists(train_path):\n"
        "            df = pd.read_csv(train_path)\n"
        "            return df.head(200).copy()\n"
        "    except Exception as e:\n"
        "        pass\n"
        "    return _tiny_df()\n\n"
        "def main():\n"
        "    df = _load_phase_subset()\n"
        "    funcs = [name for name, obj in globals().items() if callable(obj) and not name.startswith('_')]\n"
        "    for name in funcs:\n"
        "        try:\n"
        "            fn = globals()[name]\n"
        "            try:\n"
        "                _ = fn(df)\n"
        "            except TypeError:\n"
        "                _ = fn(data=df) if 'data' in fn.__code__.co_varnames else fn(df)\n"
        "            print(f'TOOL {name}: PASS')\n"
        "        except Exception as e:\n"
        "            print(f'TOOL {name}: FAIL - {e}')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    with open(run_path, 'w', encoding='utf-8') as f:
        f.write(header + body)
    return run_path


def run_smoke_runner(run_path: str, restore_dir: str) -> None:
    out_path = os.path.join(restore_dir, os.path.basename(run_path).replace('_smoke_run.py', '_smoke_output.txt'))
    err_path = os.path.join(restore_dir, os.path.basename(run_path).replace('_smoke_run.py', '_smoke_error.txt'))
    interpreter = sys.executable
    try:
        result = subprocess.run([interpreter, run_path], capture_output=True, text=True, timeout=300)
        with open(out_path, 'w') as f:
            f.write(result.stdout or '')
        if result.stderr:
            with open(err_path, 'w') as f:
                f.write(result.stderr)
    except Exception as e:
        with open(err_path, 'w') as f:
            f.write(str(e))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: toolsmith_smoke_runner.py <competition> <phase_dir> <restore_dir>')
        sys.exit(1)
    competition, phase_dir, restore_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    rp = write_smoke_runner(competition, phase_dir, restore_dir)
    run_smoke_runner(rp, restore_dir)
