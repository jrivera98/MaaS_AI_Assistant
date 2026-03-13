import pandas as pd
from langchain_ollama import OllamaLLM


# -------------------------
# Load Excel
# -------------------------

df = pd.read_excel("data/metro_rikshaw_dummy_demand.xlsx")

# Convert wide → long format
df_long = df.melt(
    id_vars=["time"],
    var_name="station",
    value_name="riksha_passengers"
)

# Remove rows where no riksha required
df_long = df_long[df_long["riksha_passengers"] > 0]

# print(df_long.head())
# print(df["time"].dtype)
# df["time"] = pd.to_datetime(df["time"], unit="d", origin="1899-12-30").dt.strftime("%I:%M %p")
# -------------------------
# LLM
# -------------------------

llm = OllamaLLM(
    model="qwen2.5:3b"
)


# -------------------------
# Agent Function
# -------------------------

def ask_agent(question: str):
    question=question.lower()
    prompt = f"""
You are a Python data analyst, who analyzes the data and produce answers in natural language.

You are working with a pandas dataframe called df_long.

DataFrame columns:
- time
- station
- riksha_passengers

Example rows:

{df_long.head(8).to_string()}

Important rules:
- Only generate valid Python pandas code
- Do not explain anything
- Store the final answer in a variable called result
- don't give one word answers like only result, form a sentence
-Always compute the answer in a variable called value first, then create result using f"{{value}} ...".
Never put pandas calculations directly inside the f-string.
- Use the dataframe df_long

Examples:

Question: Which station has highest riksha demand?
Code:
station = df_long.groupby("station")["riksha_passengers"].sum().idxmax()
value = df_long.groupby("station")["riksha_passengers"].sum().max()
result = f"{{station}} station has the highest riksha demand with {{value}} passengers."

Question: Total riksha passengers
Code:
value = df_long["riksha_passengers"].sum()
result = f"Total passengers needing riksha are {{value}}."

Question: How many passengers need rickshaws at 6:27 AM in the deccan gymkhana station?

Code:
value = df_long[(df_long["time"] == "06:27:00") & (df_long["station"] == "deccan gymkhana")]["riksha_passengers"].sum()
result = f"{{value}} passengers need rickshaws at 6:27 AM in the Deccan Gymkhana station."

Now answer this question.

Question:
{question}
"""

    # Generate pandas code
    code = llm.invoke(prompt)

    try:

        local_vars = {"df_long": df_long, "pd": pd}

        exec(code, {}, local_vars)

        result = local_vars.get("result")

        if result is None:
            if "value" in local_vars:
             result = f"{local_vars['value']} passengers need rikshaws."

        return str(result)

    except Exception as e:

        return f"""
Error executing generated code.

Generated code:
{code}

Error:
{e}
"""