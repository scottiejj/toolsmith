# Prompt ensuring  Markdown match README formats.

PROMPT_TOOLSMITH_TASK = '''
You are now designing ML tools for the current development phase: {phase_name}.
Your tools should include BOTH:
1. Dataset-specific tools: Tailored to the unique characteristics of this dataset
2. Commonly reusable tools: General-purpose utilities applicable across data science tasks

The Planner and Developer will use these tools to execute tasks in this phase.
I will provide you with DATASET BACKGROUND, PHASE Information, and previous reports and plans.

You can use the following reasoning pattern to design the tools:
1. Understand the dataset domain and phase objectives.
2. Identify dataset-specific needs (e.g., domain-specific transformations, dataset-specific patterns)
3. Identify commonly needed operations for this phase type (e.g., standard EDA plots, generic cleaners)
4. For each tool, ask yourself and answer:
    - "Is this tool dataset-specific or generally reusable?"
    - "What parameters should this tool accept to be flexible?"
    - "Which data types (numerical, categorical, datetime, ID) does this tool handle?"
    - "What are the expected inputs and outputs?"
    - "What constraints or edge cases must the tool handle?"
5. Ensure tools are:
    - Phase-appropriate (EDA tools differ from cleaning or feature engineering tools)
    - Well-balanced (mix of specific + general utilities)
    - Well-documented (clear parameter descriptions, applicable situations, and usage notes)
    - Compliant with README format (Markdown documentation)

Design as many tools as needed to comprehensively support this phase, but prioritize quality over quantity.
Focus on tools that will be most useful to the Planner and Developer.

Important: Strongly prioritize domain-specific (dataset background-aware) tools whenever they provide clear value. Make their domain rationale explicit so users understand why they are beneficial for this dataset.
'''


PROMPT_TOOLSMITH_CONSTRAINTS = '''
## CONSTRAINTS ##
1. Tool Balance:
   - Design tools appropriate for phase: {phase_name}
   - Include BOTH dataset-specific AND generic data science tools
   - Prioritize dataset-specific tools when they add measurable value (keep a healthy mix)
   - Prioritize quality and usefulness over quantity
   
2. Technical Requirements:
   - Use only: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
   - No file I/O operations inside tools
   - No network calls or external dependencies
   - All tools must be pure functions (no global state modifications)

3. Design Principles:
   - Parameterize column names (never hard-code feature names)
   - Provide clear, actionable error messages
   - Handle edge cases gracefully (empty DataFrames, missing columns, wrong types)
   - Include type hints for all parameters and return values
   - Document assumptions and limitations in notes
   - Mark in description whether tool is "dataset-specific" or "generic"
   - For dataset-specific tools, include a brief "Domain rationale" note explaining why the tool fits this dataset background

4. Output Requirements:
   - Each tool MUST have markdown documentation
   - Follow exact README format for Markdown
'''




PHASE_TO_PHASE_GUIDELINES = {
  "Preliminary Exploratory Data Analysis": '''
- Focus on: exploratory visualization helpers, summary statistic functions, correlation analysis utilities, plus dataset background-aware diagnostics
- Example tools: multi-feature distribution plotter, automated outlier detector, target-feature relationship analyzer, domain-specific checks derived from dataset background
- Visualization tools MUST NOT display figures (no `plt.show()`). They should return the plot object (figure/axes) for the caller to save.
''',

  "Data Cleaning": '''
- Focus on: missing value handlers, outlier treatment, data type converters, inconsistency fixers, with domain-aware rules
- Example tools: intelligent missing value imputer (context-aware), categorical encoder with rare category handling, datetime parser with error recovery, dataset-specific validators
''',

  "In-depth Exploratory Data Analysis": '''
- Focus on: advanced statistical analysis, interaction detection, segmentation helpers, explicitly tailored to dataset domain
- Example tools: feature interaction heatmap generator, target stratified analysis, statistical test automation, dataset-specific segment analyses
- Visualization tools MUST NOT display figures (no `plt.show()`). They should return the plot object (figure/axes) for the caller to save.
''',

  "Feature Engineering": '''
- Focus on: domain-specific feature creators first, plus transformation utilities, aggregation helpers, and dimensionality reduction tools
- Example tools: dataset-specific interaction indicators, time-based feature extractor, target encoding with regularization, and a small set of generic transformations
''',

  "Model Building, Validation, and Prediction": '''
- Focus on: training helpers, evaluation utilities, hyperparameter search wrappers, and model fit/predict functions that respect domain constraints (e.g., grouped/time-aware CV)
- Example tools: cross-validation scorer with multiple metrics, model ensemble builder, prediction calibrator, and lightweight wrappers to fit/evaluate models
- Additionally: provide multiple model wrappers with a UNIFORM interface so the Planner and Developer can use to compare performance across algorithms.
  - Implement at least five `scikit-learn` model wrappers spanning diverse families (e.g., linear/logistic models, tree-based and ensemble methods, kernel methods, gradient boosting, and optionally probabilistic or neural models). Do not limit to the examples; support any sklearn-compatible estimator.
   - Each wrapper should be named `fit_and_evaluate_<algorithm>` and accept train/test data and configuration.
   - Each wrapper must perform cross-validation when applicable and return a standardized metrics dict for comparison.
   - Document an explicit "Return schema" in the docstring listing exact keys and types.
'''
}




# Code-first generation prompt — produce a complete Python module first
PHASE_TO_CODE_MODULE_EXTRAS = {
  "pre_eda": "Focus on lightweight summary + visualization helpers. Return figures/axes explicitly.",
  "data_cleaning": "Provide imputers, type fixers, and validators. No file I/O; return transformed DataFrames and reports.",
  "deep_eda": "Add interaction/segmentation/statistical test helpers. Optimize for speed and memory.",
  "feature_engineering": "Prioritize domain feature creators and consistent transforms. Output transformed DataFrames and feature lists.",
  "model_building": "Include ≥3 sklearn wrappers (`fit_and_evaluate_<algo>`) with standardized metrics, cv_scores, model_name, estimator.",
  "validation": "Provide evaluation and diagnostics wrappers; avoid retraining unless wrapped in CV.",
  "prediction": "Provide predict/inference helpers with input validation and consistent output schemas.",
}

PROMPT_TOOLSMITH_CODE_MODULE = r'''
## TASK: PRODUCE A COMPLETE PYTHON MODULE OF TOOLS ##

You are designing tools for the current phase: {phase_name}.
## PHASE-SPECIFIC TOOL GUIDELINES ##:
{phase_specific_instructions}

Using the Context provided below, generate a SINGLE Python module that contains multiple functions.

Strict rules:
- Follow the style of functions in `ml_tools.py` (clear docstrings: Args, Returns, Raises, Examples when helpful)
- Use only pandas, numpy, scikit-learn, scipy.stats (standard library allowed); no file/network I/O
- Pure functions: do not mutate global state; return results explicitly
- Parameterize column names; no hard-coded feature names
- Validate inputs and raise ValueError/TypeError with actionable messages
- Type-hint all parameters and return values
- Consolidate imports at the top; avoid duplicates
 - For the Model Building phase specifically: include at least three `scikit-learn` model wrappers with a common interface (`fit_and_evaluate_<algo>`), and document an explicit standardized Return schema (keys for metrics, `cv_scores`, `model_name`, `estimator`).

Output format:
- Return EXACTLY one fenced Python block containing the entire module content
- Start with imports, then define all functions
- One function per tool, snake_case names
- Define and use consistent, unambiguous key names in returned dicts and DataFrame columns; avoid synonyms. Include an explicit "Return schema" in the docstring listing exact keys and types.


Module composition guidance:
- Include BOTH dataset-specific and reusable tools for the phase
- Focus on high-utility, phase-appropriate tools
 - Prefer LESS THAN 10 tools rather than many superficial ones
- Prefer domain-specific (dataset background-aware) tools where they provide clear benefit; include a brief "Domain rationale" in each such function's docstring or notes

Context:
Phase: {phase_name}
Competition: {competition}
CURRENT phase available features:
{available_features}
CURRENT phase data preview:
{data_preview}
Previous Plans and Reports:
{previous_plans_report}



'''

# New: Convert Python source to markdown docs in the required format
PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE = r'''
## TASK: CONVERT PYTHON TOOLS TO MARKDOWN DOCS ##

You will receive a complete Python module containing multiple tool functions.

Rules:
- Use the function name as the tool name
- Derive parameter info and types from the function signature and docstring
- Keep the formatting EXACT (labels, backticks, enums, separators) per the spec
- Output ONLY one concatenated fenced block: ```markdown\n...sections...\n```
- Do not include any prose before or after the fenced block.
- If a function is dataset-specific, include a note line in the Notes section starting with "- Domain rationale:" that briefly explains why the tool is suited to this dataset/competition


For EACH top-level function, produce a markdown section that follows the EXACT format of the example below.

Here is the MARKDOWN format you should follow (example):
```markdown
## fill_missing_values

**Name:** fill_missing_values  
**Description:** Fill missing values in specified columns of a DataFrame. This tool can handle both numerical and categorical features by using different filling methods.  
**Applicable Situations:** handle missing values in various types of features

**Parameters:**
- `data`:
  - **Type:** `pd.DataFrame`
  - **Description:** A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** The name(s) of the column(s) where missing values should be filled.
- `method`:
  - **Type:** `string`
  - **Description:** The method to use for filling missing values.
  - **Enum:** `auto` | `mean` | `median` | `mode` | `constant`
  - **Default:** `auto`
- `fill_value`:
  - **Type:** ``number` | `string` | `null``
  - **Description:** The value to use when method is 'constant'.
  - **Default:** `None`

**Required:** `data`, `columns`  
**Result:** Successfully fill missing values in the specified column(s) of data  
**Notes:**
- The 'auto' method uses mean for numeric columns and mode for non-numeric columns.
- Using 'mean' or 'median' on non-numeric columns will raise an error.
- The 'mode' method uses the most frequent value, which may not always be appropriate.
- Filling missing values can introduce bias, especially if the data is not missing completely at random.
- Consider the impact of filling missing values on your analysis and model performance.

---
```

PYTHON MODULE SOURCE:
{python_source}
'''















# Standalone, batch conversion prompt (no external file access required)

PROMPT_TOOLSMITH_JSON_SCHEMA_CONVERSION = r'''
## TASK: CONVERT GENERATED TOOLS TO JSON SCHEMAS (BATCH) ##

You will receive:
1) A Python source text containing multiple function definitions (with docstrings and type hints)
2) One or more Markdown blocks, each documenting a tool (Name, Description, Applicable Situations, Parameters, Required, Result, Notes)

Your job:
- Parse ALL functions and ALL markdown docs in batch.
- For each tool, produce a JSON schema object that strictly follows the format defined below.
- Return a single JSON object: { "json_schemas": { "<tool_name>": { ... }, ... } }
- Do not include any prose outside JSON.

## TARGET JSON SCHEMA FORMAT (EMBEDDED SPEC) ##
Each tool MUST be emitted with this exact structure:

{
  "<tool_name>": {
    "name": "<tool_name>",
    "description": "<one sentence or short paragraph>",
    "applicable_situations": "<short phrase indicating when to use this tool>",
    "parameters": {
      "<param_name>": {
        "type": "<type string or union array>",
        "description": "<param description>",
        "items": { "type": "string" },          // ONLY include if type is or contains "array"
        "enum": ["v1", "v2"],                   // OPTIONAL: only if discrete set exists
        "default": <value or null>,             // OPTIONAL: include only if default exists
        "minimum": <number>,                    // OPTIONAL
        "maximum": <number>,                    // OPTIONAL
        "examples": [ ... ]                     // OPTIONAL
      },
      ...
    },
    "required": ["p1", "p2"],                   // MUST exist; [] if none
    "result": "<one sentence describing successful outcome>",
    "additionalProperties": false,
    "notes": [
      "<note 1>",
      "<note 2>",
      ...
    ]
  }
}

Type mapping rules from markdown to JSON:
- `pd.DataFrame` → "pd.DataFrame"
- ``string` | `array`` → ["string", "array"]
- ``number` | `string` | `null`` → ["number", "string", "null"]
- `string` → "string"; `number` → "number"; `integer` → "integer"; `boolean` → "boolean"

Defaults mapping from markdown backticks to JSON:
- `None` → null
- `0.5` → 0.5
- `'auto'` or `auto` → "auto"
- `True` → true; `False` → false

Enum mapping:
- `a` | `b` | `c` → ["a", "b", "c"]

Required:
- Backticked, comma-separated list → JSON array of strings
- If markdown says "None", use []

Key ordering:
- name → description → applicable_situations → parameters → required → result → additionalProperties → notes

IMPORTANT:
- The above format specification is definitive; do not assume access to any external files.
- Produce valid JSON only. No comments or extra keys.

## INPUT ##
PYTHON_SOURCE:
{python_source}

MARKDOWN_DOCS:
{markdown_docs}

## OUTPUT ##
Return ONLY:
{
  "json_schemas": {
    "<tool_name_1>": { ... },
    "<tool_name_2>": { ... },
    ...
  }
}
'''




