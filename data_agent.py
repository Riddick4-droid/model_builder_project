"""
Data Agent: inspects the raw data file and returns metadata.
No actual file reading yet – the LLM infers from the path and context.
In production, add a tool to compute real statistics.
"""

from model_builder_project.base import LLMAgent

DATA_SYSTEM_PROMPT = """
You are a Data Agent. Your job is to inspect a dataset given by a file path.
You will receive the `data_path` in the state. Based on the file name and extension,
you should infer or simulate basic metadata.

Return a JSON object with the following keys:
- "data_loaded": boolean (true if path looks valid)
- "shape": [number_of_rows, number_of_columns] (estimate)
- "columns": list of column names (generic if unknown)
- "missing_percentage": float (0-100)
- "imputation_suggestion": string (e.g., "mean", "median", "drop rows")
- "target_column_guess": string (most likely target column name, e.g., "target", "label", "class")

Be realistic. If the file name contains "iris", assume 150 rows, 4 features, target "species".
If it's "titanic", assume 891 rows, 12 columns, target "survived".
If the extension is .csv, it's likely tabular. For .parquet, same.
If the path seems invalid, set data_loaded to false and explain in a new key "error".
"""

class DataAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            name="DataAgent",
            system_prompt=DATA_SYSTEM_PROMPT,
            model="gpt-4o-mini",
            temperature=0
        )