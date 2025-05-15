import pandas as pd

capitale_aziendale = [
    "Net Value Per Share (A)",
    # "Working Capital to Total Assets",
    "Current Assets/Total Assets",
    "Fixed Assets to Assets",
    "Cash/Total Assets",
]

solidita_finanziaria = [
    "Debt ratio %",
    "Total debt/Total net worth",
    "Quick Assets/Current Liability",
    "Interest-bearing debt interest rate",
    # "Equity to Liability",
]

conto_economico = [
    "Operating Profit Rate",
    # "After-tax Net Profit Growth Rate",
    "Cash Flow to Total Assets",
    "Net Income to Stockholder's Equity",
    "Operating Profit Growth Rate",
]

indici_operativi = [
    "Total Asset Turnover",
    "Inventory Turnover Rate (times)",
    "Accounts Receivable Turnover",
    "Current Ratio",
    "Cash Flow to Sales",
]

variabile_outcome = [
    "Current Asset Turnover Rate",
    # "Liability-Assets Flag",
    "Continuous Net Profit Growth Rate",
    # "Regular Net Profit Growth Rate",
    "Cash Turnover Rate",
    "Bankrupt?",
]

# List of all features
sel_vars = (
    capitale_aziendale
    + solidita_finanziaria
    + conto_economico
    + indici_operativi
    + variabile_outcome
)


def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    return df[sel_vars]


def partially_clean_dataframe(df):
    df.columns = df.columns.str.strip()
    return df
