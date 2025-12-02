# Prompt ensuring JSON and Markdown match README formats.

PROMPT_TOOLSMITH_TASK = '''
Please design ML tools for the current development phase: {phase_name}.
Your tools should include BOTH:
1. Dataset-specific tools: Tailored to the unique characteristics of this competition's data
2. Commonly reusable tools: General-purpose utilities applicable across data science tasks

The Planner and Developer will use these tools to execute tasks in this phase.
I will provide you with DATASET BACKGROUND, PHASE CONTEXT, and previous reports and plans.

You can use the following reasoning pattern to design the tools:
1. Understand the dataset domain and phase objectives.
2. Identify dataset-specific needs (e.g., domain-specific transformations, competition-specific patterns)
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
    - Compliant with README format (JSON schema + Markdown documentation)

Design as many tools as needed to comprehensively support this phase, but prioritize quality over quantity.
Focus on tools that will be most useful to the Planner and Developer.
'''


PROMPT_TOOLSMITH_CONSTRAINTS = '''
## CONSTRAINTS ##
1. Tool Balance:
   - Design tools appropriate for phase: {phase_name}
   - Include BOTH dataset-specific AND commonly reusable tools
   - Avoid duplicating existing tools (list provided)
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
   - Mark in description whether tool is "dataset-specific" or "reusable"

4. Output Requirements:
   - Each tool MUST have corresponding JSON schema entry
   - Each tool MUST have markdown documentation
   - Follow exact README format for both JSON and Markdown
   - Assign correct category: "data_cleaning" | "feature_engineering" | "model_build_predict"
'''




PROMPT_TOOLSMITH_PHASE_GUIDELINES = '''
## PHASE-SPECIFIC TOOL GUIDELINES ##

### Preliminary EDA Phase
- Focus on: exploratory visualization helpers, summary statistic functions, correlation analysis utilities
- Example tools: multi-feature distribution plotter, automated outlier detector, target-feature relationship analyzer

### Data Cleaning Phase
- Focus on: missing value handlers, outlier treatment, data type converters, inconsistency fixers
- Example tools: intelligent missing value imputer (context-aware), categorical encoder with rare category handling, datetime parser with error recovery

### In-depth EDA Phase
- Focus on: advanced statistical analysis, interaction detection, segmentation helpers
- Example tools: feature interaction heatmap generator, target stratified analysis, statistical test automation

### Feature Engineering Phase
- Focus on: transformation utilities, aggregation helpers, domain-specific feature creators
- Example tools: polynomial/interaction feature generator with auto-selection, time-based feature extractor, target encoding with regularization

### Model Building Phase
- Focus on: training helpers, evaluation utilities, hyperparameter search wrappers
- Example tools: cross-validation scorer with multiple metrics, model ensemble builder, prediction calibrator
'''



PROMPT_TOOLSMITH_MARKDOWN_FORMAT = r'''
## REQUIRED MARKDOWN FORMAT ##
For each tool, produce EXACT markdown following this structure:

## <tool_name>

**Name:** <tool_name>  
**Description:** <one-sentence or short paragraph description>  
**Applicable Situations:** <short phrase describing when to use this tool>

**Parameters:**
- `<param_name>`:
  - **Type:** `<type>` (e.g., `pd.DataFrame`, ``string` | `array``, `number`, `boolean`)
  - **Description:** <clear parameter description>
  - **Enum:** `value1` | `value2` | `value3`  (ONLY include this line if enum exists)
  - **Default:** `<value>` or `None`  (ONLY include this line if default exists)

(Repeat the above parameter block for each parameter)

**Required:** `param1`, `param2`  (or write `None` if no required parameters)
**Result:** <one-sentence description of the successful outcome>  
**Notes:**
- <explanatory note 1>
- <explanatory note 2>
- <explanatory note 3>
(Add as many notes as needed to explain assumptions, limitations, edge cases, or best practices)

---

IMPORTANT FORMATTING RULES:
1. Use exactly two spaces after each colon in bold labels (e.g., "**Name:**  ")
2. Parameter type uses backticks with pipes for unions: ``string` | `array``
3. Enum values separated by ` | ` (space-pipe-space) with backticks around each value
4. Default values use backticks: `None`, `0.5`, `'auto'`
5. End each tool section with "---" separator (three dashes)
6. Required parameters listed as comma-separated with backticks: `param1`, `param2`
7. Notes are bullet points, each starting with "- "

Example of correct formatting:

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

'''


# Additional prompt: require Python implementations alongside markdown docs.
# Do NOT change the markdown format spec; this augments the output with code blocks.
PROMPT_TOOLSMITH_CODE_IMPLEMENTATION = r'''
## REQUIRED CODE OUTPUT (IN ADDITION TO MARKDOWN) ##

For EACH tool documented in markdown, also emit a fenced Python code block that implements the tool.

Rules:
- Use only pandas, numpy, scikit-learn (and standard library) — no file/network I/O.
- Pure functions: no global state; return results explicitly.
- Match names/parameters/types exactly as documented in the markdown.
- Include type hints on parameters and return values.
- Raise clear ValueError/TypeError when inputs are invalid (missing columns, wrong dtypes).
- Keep implementations concise and practical; prefer readability over micro-optimizations.

Formatting:
- Place the code block immediately after the tool's markdown section.
- Use fenced block with language hint: ```python
- Define ONE function per tool, named exactly as the tool name (snake_case).
- Function must be standalone (no external helpers) unless trivially short.

Example structure per tool (abbreviated):
## fill_missing_values
... (markdown as specified) ...

```python
from typing import List, Union
import pandas as pd
import numpy as np

def fill_missing_values(
  data: pd.DataFrame,
  columns: Union[str, List[str]],
  method: str = "auto",
  fill_value: Union[float, int, str, None] = None,
) -> pd.DataFrame:
  # validate inputs
  if isinstance(columns, str):
    columns = [columns]
  missing = [c for c in columns if c not in data.columns]
  if missing:
    raise ValueError(f"Columns not found: {missing}")

  df = data.copy()
  for c in columns:
    if method == "auto":
      if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].fillna(df[c].mean())
      else:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else df[c])
    elif method == "mean":
      if not pd.api.types.is_numeric_dtype(df[c]):
        raise TypeError("mean requires numeric column")
      df[c] = df[c].fillna(df[c].mean())
    elif method == "median":
      if not pd.api.types.is_numeric_dtype(df[c]):
        raise TypeError("median requires numeric column")
      df[c] = df[c].fillna(df[c].median())
    elif method == "mode":
      df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else df[c])
    elif method == "constant":
      df[c] = df[c].fillna(fill_value)
    else:
      raise ValueError(f"Unknown method: {method}")
  return df
```
'''

# New: Code-first generation prompt — produce a complete Python module first
PROMPT_TOOLSMITH_CODE_MODULE = r'''
## TASK: PRODUCE A COMPLETE PYTHON MODULE OF TOOLS ##

You are designing tools for the current phase: {phase_name}.
Using the dataset background provided, generate a SINGLE Python module that contains multiple functions.

Strict rules:
- Follow the style of functions in `ml_tools.py` (clear docstrings: Args, Returns, Raises, Examples when helpful)
- Use only pandas, numpy, scikit-learn, scipy.stats (standard library allowed); no file/network I/O
- Pure functions: do not mutate global state; return results explicitly
- Parameterize column names; no hard-coded feature names
- Validate inputs and raise ValueError/TypeError with actionable messages
- Type-hint all parameters and return values
- Consolidate imports at the top; avoid duplicates

Output format:
- Return EXACTLY one fenced Python block containing the entire module content
- Start with imports, then define all functions
- One function per tool, snake_case names

Module composition guidance:
- Include BOTH dataset-specific and reusable tools for the phase
- Focus on high-utility, phase-appropriate tools
- Prefer 6–12 tools rather than many superficial ones

Context:
Phase: {phase_name}
Competition: {competition}
Dataset Background:
{background}
'''

# New: Convert Python source to markdown docs in the required format
PROMPT_TOOLSMITH_MARKDOWN_FROM_CODE = r'''
## TASK: CONVERT PYTHON TOOLS TO MARKDOWN DOCS ##

You will receive a complete Python module containing multiple tool functions.
For EACH top-level function, produce a markdown section that follows the EXACT format of PROMPT_TOOLSMITH_MARKDOWN_FORMAT.

Rules:
- Use the function name as the tool name
- Derive parameter info and types from the function signature and docstring
- Keep the formatting EXACT (labels, backticks, enums, separators) per the spec
- Output ONLY one concatenated fenced block: ```markdown\n...sections...\n```
- Do not include any prose before or after the fenced block.

REFERENCE FORMAT (do not reproduce here, just follow it):
{markdown_format_spec}

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

# Helper wrapper to inject inputs (batch-mode)
PROMPT_TOOLSMITH_JSON_CONVERSION_WITH_CONTEXT = '''
{conversion_spec}

PYTHON_SOURCE:
{python_source}

MARKDOWN_DOCS:
{markdown_docs}

Convert ALL tools above to JSON schemas following the spec exactly. Output only the JSON object.
'''.format(conversion_spec=PROMPT_TOOLSMITH_JSON_SCHEMA_CONVERSION, python_source="{python_source}", markdown_docs="{markdown_docs}")
