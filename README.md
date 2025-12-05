# AutoKaggle-Toolsmith

This repository is an adaptation of the original AutoKaggle paper: "AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions" (original repo: https://github.com/multimodal-art-projection/AutoKaggle).

<p align="center">
    <a href="https://arxiv.org/abs/2410.20424.pdf"><img src="https://img.shields.io/badge/Paper-Arxiv-red"></a>
</p>


## Toolsmith Key Adaptations

- **New agent**: `Toolsmith` — understands dataset background and phase context to design domain-specific tools and sklearn model wrappers with a uniform interface and explicit return schemas, rather than relying on a static hand‑coded ML library.
- Robustness improvements — generated modules are validated; standardized outputs reduce schema/key mismatches across agents.
- Broader model breadth — encourages more complex, diverse model families (linear/logistic, tree/ensemble, kernel, gradient boosting, optional probabilistic/neural) via uniform wrappers.
- Model support — supports `gpt-4.1` for `Toolsmith`/`Planner`/`Developer`.
- Example output is under EXAMPLE TOOLSMITH OUTPUT


## AutoKaggle

AutoKaggle is a powerful framework that assists data scientists in completing data science pipelines through a collaborative multi-agent system. The framework combines iterative development, comprehensive testing, and a machine learning tools library to automate Kaggle competitions while maintaining high customizability. The original features of AutoKaggle include:

- **Multi-agent Collaboration**: Five specialized agents (`Reader`, `Planner`, `Developer`, `Reviewer`, and `Summarizer`) work together through six key competition phases.
- **Iterative Development and Unit Testing**: Robust code verification through debugging and comprehensive unit testing.
- **ML Tools Library**: Validated functions for data cleaning, feature engineering, and modeling.
- **Comprehensive Reporting**: Detailed documentation of workflow and decision-making processes.



### Running

To run experiments, use the following command:

```bash
bash run_multi_agents1.sh
```

#### Configuration Parameters

- **Competition Selection**
  - `competitions`: Define target competitions in the script

- **Experiment Control**
  - `start_run`, `end_run`: Define experiment iterations (default: 1-5)
  - `dest_dir_param`: Output directory specification (default: "all_tools")

- **Model Configuration**
  - Supports `gpt-4o/gpt-4.1` for `Toolsmith`/`Planner`/`Developer`; configure in the run script or `multi_agents/sop.py`.

