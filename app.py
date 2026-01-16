from dotenv import load_dotenv
import streamlit as st
import sqlite3
import pandas as pd
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Optional
import operator
import plotly.express as px
import plotly.graph_objects as go


# === CONFIGURATION ===
load_dotenv()
DB_PATH = "my_data 1.db"
JARGON_PATH = "Jargons.xlsx"
LOGIC_PATH = "Logics.xlsx"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TABLE_PNL = "p_and_l"
TABLE_UT = "utt"

if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit Cloud secrets or .env file.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)



# === SHARED STATE ===
class GraphState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    blueprint: str
    sql_query: str 
    data_frame: Optional[pd.DataFrame]
    final_insights: str


# === LOGIC LOADING FUNCTIONS ===
def load_excel_logic():
    try:
        jargon_df = pd.read_excel(JARGON_PATH)
        logic_df = pd.read_excel(LOGIC_PATH)
        return jargon_df.to_string(index=False), logic_df.to_string(index=False)
    except Exception as e:
        return f"Error loading files: {e}", ""


# === DATABASE FUNCTIONS ===
def execute_sql_query(query: str, db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        return df, "Success"
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()


# === AGENT 1: THE ARCHITECT (IMPROVED GROUPING & JARGON AWARENESS) ===
def architect_node(state: GraphState):
    latest_user_message = state["messages"][-1].content
    recent_history = "\n".join(
        f"{type(m).__name__}: {m.content[:120]}..."
        for m in state["messages"][-4:-1]
    )
    jargon, logic = load_excel_logic()
    
    prompt = f"""
You are the Architect. **CRITICAL: IGNORE ALL PREVIOUS BLUEPRINTS. Analyze ONLY THE LATEST USER MESSAGE.**

LATEST USER QUESTION (this is the ONLY source for requirements):  
{latest_user_message}

**MANDATORY REFERENCES — READ CAREFULLY:**
EXCEL LOGICS REFERENCE (find metric / formula here first):  
{logic}

JARGONS REFERENCE (use this to understand synonyms and column mappings):  
{jargon}

PREVIOUS CONVERSATION (business context only — ignore dates & numbers):  
{recent_history if recent_history.strip() else "(none)"}

--- VERY STRICT ANALYSIS RULES ---

1. METRIC LOOKUP
   - Search the "Prompt" column in logics.xlsx for the best match (exact or strong keyword match, case insensitive)
   - If multiple close matches → choose the most specific one

2. TABLE MAPPING
   - Use the "Table" value from the matched logics row
   - P&L  → p_and_l
   - UT   → utt
   - P&L, UT → both (only if formula clearly needs fields from both)

3. FORMULA
   - Copy the "Formula" from logics.xlsx exactly when available
   - Only derive when no formula exists and it's very obvious

4. GROUPING — IMPORTANT IMPROVEMENT
   - Look at the user question for words indicating the desired grouping dimension:
     - "account", "accounts", "customer", "customers", "by account" → `FinalCustomerName`
     - "DU", "DUs", "Delivery Unit", "Delivery Units", "Exec DU", "Executing DU" → `Exec DU` (P&L) or `Delivery_Unit` (UT)
     - "BU", "BUs", "Exec DG", "DG", "Delivery Group" → `Exec DG` (P&L) or `DeliveryGroup` (UT)
     - "Segment", "Participating Vertical", "PVG", "PDG", "Vertical" → `Segment` or `ParticipatingVerticalDeliveryGroup`
     - "Region", "Sales Region" → `Sales Region`
     - "Month", "MoM", "trend", "over months" → `Month`
   - ALWAYS scan jargons.xlsx "Jargon" column for any terms in the question — if a match, use the corresponding "P&L Report Column name" or "UT Report Column name" based on the table.
   - If the question is clearly asking to group or list by a dimension other than account/customer → **do NOT default to FinalCustomerName**.
   - Only use `FinalCustomerName` when the question contains words like account/customer/final customer.
   - If no grouping dimension is mentioned → no GROUP BY (overall total)

5. DATE FILTER
   - ONLY use dates explicitly mentioned in the latest question
   - Format: 'YYYY-MM-01' for Month-level filters
   - Trends/MoM/multiple months → multiple values or no single filter

6. CONDITIONS
   - Parse filters like "<30%", "greater than 5", "negative", etc.

7. PERCENTAGES
   - Always protect division: / NULLIF(denominator, 0)

8. COMPARISON — ENHANCED FOR VARIATION QUERIES
   - If question asks for variation/change/difference/comparison between specific months (e.g. "varied from July 2025 to August 2025"):
     - Set Grouping_Column: `Month`
     - Date_Filter: IN ('YYYY-MM-01' for each mentioned month)
     - In Formula: calculate separate values for each month (e.g. via CTE or subquery), then add diff = value_month2 - value_month1, pct_change = (value_month2 - value_month1)/NULLIF(value_month1,0) * 100
     - Output as single row with columns like Month1_Value, Month2_Value, Variation, Pct_Change

9. C&B SPECIFIC
   - For C&B: SUM(`Amount in USD`) WHERE `Group Description` IN ('C&B Cost Onsite', 'C&B Cost Offshore')

10. REVENUE DEFAULT (if needed)
    - SUM WHERE `Group1` IN ('ONSITE', 'OFFSHORE', 'INDIRECT REVENUE')

OUTPUT FORMAT — STRICTLY THIS STRUCTURE (one value per line):

**Metric**: ...
**Table**: p_and_l / utt / both
**Column**: ... (main column used)
**Formula**: ... (exact from logics "Formula" column — do NOT rephrase or simplify)
**Grouping_Column**: ... (exact database column name — very important!)
**Grouping_Name**: ... (human-friendly name, e.g. Delivery Unit, Segment, Account)
**Date_Filter**: YYYY-MM-01 or NONE or IN ('2025-07-01', ...)
**Conditions**: ... (e.g. CM_Percentage < 30)
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"blueprint": response.content.strip()}


# === AGENT 2: THE SQL ANALYST (ENHANCED GROUPING & COMPARISON AWARENESS) ===
def sql_analyst_node(state: GraphState):
    blueprint = state["blueprint"]
    user_query = state["messages"][-1].content 
    jargon, logic = load_excel_logic()
    
    prompt = f"""You are an expert SQL Analyst for SQLite. Generate **only** a correct SQL query.

**BLUEPRINT FROM ARCHITECT**:
{blueprint}

**USER QUESTION**:
{user_query}

**EXCEL LOGICS REFERENCE (MANDATORY — FOLLOW EXACTLY)**:
{logic}

**JARGONS REFERENCE (use for column names when needed)**:
{jargon}

MANDATORY RULES:

1. Use the **Grouping_Column** from the blueprint as the GROUP BY column.
   - If blueprint says **Grouping_Column**: `Exec DU` → GROUP BY `Exec DU`
   - If blueprint says **Grouping_Column**: `Delivery_Unit` → GROUP BY `Delivery_Unit`
   - If blueprint says **Grouping_Column**: `Exec DG` → GROUP BY `Exec DG`
   - If blueprint says **Grouping_Column**: `Segment` → GROUP BY `Segment`
   - Do NOT fall back to `FinalCustomerName` unless the blueprint explicitly says so.

2. Respect the **Table** from blueprint:
   - p_and_l → FROM `p_and_l`
   - utt    → FROM `utt`
   - both   → use appropriate JOIN (usually ON `Month` or customer key)

3. AGGREGATION PATTERN — MUST FOLLOW THIS EXACTLY
   - NEVER sum columns directly like SUM(`Direct Expense` + `OWN OVERHEADS`)
   - ALWAYS use SUM(CASE WHEN `Group1` IN (list of values) THEN `Amount in USD` ELSE 0 END)
   - For revenue: usually `Group1` IN ('ONSITE', 'OFFSHORE', 'INDIRECT REVENUE')
   - For costs / direct expenses / overheads: use the exact list of Group1 values from logics.xlsx or blueprint
   - Example for costs (adapt the list from your logics file):
     SUM(CASE WHEN `Group1` IN ('Direct Expense', 'OWN OVERHEADS', 'Indirect Expense', 'Project Level Depreciation', 'Direct Expense - DU Block Seats Allocation', 'Direct Expense - DU Pool Allocation', 'Establishment Expenses') THEN `Amount in USD` ELSE 0 END) AS `Cost`
   - Do NOT invent column names that do not exist in the table
   - If you see words like "Direct Expense", "OWN OVERHEADS", etc. in the formula or logics — they are ALWAYS values inside the `Group1` column, NEVER separate columns.

4. For percentages: always use NULLIF(denominator, 0)

5. Always quote column names with backticks: `Group1`, `Group Description`, `Month`, `Amount in USD`, `FinalCustomerName`, etc.

6. Output style — REQUIRED:
   - SELECT grouping column first (aliased nicely, e.g. `Exec DU` AS `DU`)
   - Then intermediate aggregations with clear aliases (Revenue, Cost, CB_Cost, etc.)
   - Final metric LAST with alias like `CM_Percentage`, `CB_Percentage`, etc.
   - IMPORTANT: In the final metric expression, REPEAT the full SUM(CASE ...) expressions — DO NOT reference the intermediate aliases in the final calculation.
   - Do NOT collapse everything into one column. Show the building blocks separately.
   - WHERE clause from Date_Filter and any Conditions
   - GROUP BY the column named in Grouping_Column
   - If Conditions contain the metric (e.g. CM < 30) → use HAVING

7. Do NOT add ORDER BY unless MoM/trend (then ORDER BY `Month`). Under no circumstances turn Group1 filter values into column names.

8. **Formula handling**
   - Copy the "Formula" column **verbatim** when it exists
   - If the formula uses descriptions like "Direct Expense, OWN OVERHEADS, …" → interpret them as `Group1` values, NOT as column names
   - Never rewrite CASE logic into direct column sums

9. COMPARISON / VARIATION HANDLING
   - If blueprint indicates a comparison/variation (e.g. diff between months):
     - Use GROUP BY `Month` (even if Grouping_Column is Month)
     - Calculate separate values per month using CTE or subqueries
     - Output columns like: `Month`, `Metric_Value`, or pivoted: Month1_Value, Month2_Value, Variation, Pct_Change
     - Variation = value_month2 - value_month1
     - Pct_Change = (value_month2 - value_month1) / NULLIF(value_month1, 0) * 100
     - Do NOT sum across months — always separate rows or pivoted columns

10. TWO-MONTH COMPARISON SPECIAL CASE
   - When the blueprint Date_Filter is IN (exactly two months) and the question asks for variation/change/difference between them:
     - Produce a SINGLE-ROW result (pivoted)
     - Use month-specific column names in the format: "{{Month_Year}} {{Metric}}" (e.g. `July 2025 CB Cost`, `August 2025 CB Cost`)
     - Order the columns so the later month comes first (most recent on the left)
     - Then add `Variation` and `Pct_Change`
     - Example desired output columns for July → August:
       `August 2025 CB Cost`, `July 2025 CB Cost`, `Variation`, `Pct_Change`
     - Use CTE + CASE or subqueries to pivot (do NOT use dynamic SQL or PIVOT function — SQLite doesn't have it)
     - Calculate:
       Variation = later_month_value - earlier_month_value
       Pct_Change = (later - earlier) / NULLIF(earlier, 0) * 100
   - Only pivot when exactly two months are compared — for more months keep row-per-month output

    Return **ONLY** the raw SQL query.
    No explanation, no markdown, no ```sql fences, no comments.

Return **ONLY** the raw SQL query.
No explanation, no markdown, no ```sql fences, no comments.
"""
    raw_sql = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    sql_query = raw_sql.replace("```sql", "").replace("```", "").replace("\\n", " ").strip()
    
    df, status = execute_sql_query(sql_query, DB_PATH)
    return {"sql_query": sql_query, "data_frame": df}

# === AGENT 3: INSIGHTS (MINOR UPDATE FOR BETTER TREND ANALYSIS) ===
def insights_node(state: GraphState):
    df = state["data_frame"]
    user_query = state["messages"][-1].content
    
    if df is None or df.empty:
        return {"final_insights": "Query executed but no data matched filters."}

    data_sample = df.head(15).to_string()
    
    prompt = f"""
Insights Agent. Analyze for: '{user_query}'

DATA:
{data_sample}

TASK: 3-sentence summary. For trends: highlight growth/decline. For lists: top/bottom performers by % metric.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_insights": response.content}


# === GRAPH CONSTRUCTION ===
if "graph" not in st.session_state:
    workflow = StateGraph(GraphState)
    
    workflow.add_node("architect", architect_node)
    workflow.add_node("analyst", sql_analyst_node)
    workflow.add_node("insights", insights_node)

    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "analyst")
    workflow.add_edge("analyst", "insights")
    workflow.add_edge("insights", END)
    
    st.session_state.graph = workflow.compile()


# === STREAMLIT UI ===
st.set_page_config(page_title="L&T Business Intelligence", layout="wide")
st.title("📊 Multi-Agent Business Intelligence (Logics.xlsx Enabled)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize render counter for unique plotly chart keys
if "render_counter" not in st.session_state:
    st.session_state.render_counter = 0


# ────────────────────────────────────────────────────────────────
#   1. Render ALL PREVIOUS assistant responses with full content
# ────────────────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
        continue

    # Assistant message
    with st.chat_message("assistant"):
        st.markdown(message["content"])

        if "sql" in message and message["sql"]:
            with st.expander("🛠️ Executed SQL Query", expanded=False):
                st.code(message["sql"], language="sql")

        if "df" in message and message["df"] is not None and not message["df"].empty:
            df = message["df"]

            # KPIs
            st.markdown("### 🎯 Key Performance Indicators")
            kpi_cols = st.columns(4)
            numeric_cols = df.select_dtypes(include=['number']).columns

            if len(numeric_cols) >= 1:
                for i, col in enumerate(numeric_cols[:4]):
                    val = df[col].sum() if col.lower() not in ['cm_percentage', 'cb_percentage', 'percentage'] else df[col].mean()
                    kpi_cols[i % 4].metric(
                        label=col.replace("_", " ").title(),
                        value=f"{val:,.2f}" if isinstance(val, (int, float)) else val,
                        delta=None
                    )

            # Visualization
            st.markdown("### 📈 Visualization")
            df_cols_lower = {col.lower(): col for col in df.columns}

            group_candidates = ['finalcustomername', 'customer', 'month', 'account', 'name', 'group1']
            group_col = next((df_cols_lower.get(c) for c in group_candidates if c in df_cols_lower), None)

            value_candidates = ['revenue', 'cost', 'cb_cost', 'cb_percentage', 'cm_percentage', 'cm %', 'amount']
            value_cols = [df_cols_lower[cand] for cand in value_candidates if cand in df_cols_lower]

            if group_col and value_cols:
                current_key = f"visualization_{st.session_state.render_counter}"
                st.session_state.render_counter += 1

                if any('percentage' in c.lower() or 'cm' in c.lower() for c in value_cols) and len(df) <= 15:
                    perc_col = next(c for c in value_cols if any(x in c.lower() for x in ['percentage', 'cm']))
                    fig = px.pie(df, names=group_col, values=perc_col, title=f"% by {group_col}", hole=0.35)
                    st.plotly_chart(fig, use_container_width=True, key=current_key)
                else:
                    main_value = value_cols
                    fig = px.bar(
                        df.sort_values(main_value, ascending=False).head(15),
                        x=group_col, y=main_value,
                        title=f"{', '.join(main_value)} by {group_col}",
                        text_auto=True
                    )
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    st.plotly_chart(fig, use_container_width=True, key=current_key)

            # Results Table
            st.markdown("### 📋 Results Table")
            st.dataframe(df, use_container_width=True)


# ────────────────────────────────────────────────────────────────
#   2. New user input + current assistant response
# ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a business question (e.g. 'M-o-M C&B cost % trend')..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents collaborating (using logics.xlsx)..."):
            input_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.chat_history
            ]

            fresh_state = {
                "messages": input_messages,
                "blueprint": "",
                "sql_query": "",
                "data_frame": None,
                "final_insights": ""
            }

            result = st.session_state.graph.invoke(fresh_state)

            insights = result["final_insights"]
            sql      = result["sql_query"]
            df       = result["data_frame"]

            # Blueprint (debug)
            with st.expander("🔍 Architect Blueprint", expanded=False):
                st.info(result["blueprint"])

            # Insights
            st.markdown("### 📋 Insights")
            st.markdown(insights)

            # ── Immediately render KPIs + chart + table + SQL for this new response ──
            if df is not None and not df.empty:
                st.markdown("### 🎯 Key Performance Indicators")
                kpi_cols = st.columns(4)
                numeric_cols = df.select_dtypes(include=['number']).columns

                if len(numeric_cols) >= 1:
                    for i, col in enumerate(numeric_cols[:4]):
                        val = df[col].sum() if col.lower() not in ['cm_percentage', 'cb_percentage', 'percentage'] else df[col].mean()
                        kpi_cols[i % 4].metric(
                            label=col.replace("_", " ").title(),
                            value=f"{val:,.2f}" if isinstance(val, (int, float)) else val,
                            delta=None
                        )

                # Visualization
                st.markdown("### 📈 Visualization")
                df_cols_lower = {col.lower(): col for col in df.columns}

                group_candidates = ['finalcustomername', 'customer', 'month', 'account', 'name', 'group1']
                group_col = next((df_cols_lower.get(c) for c in group_candidates if c in df_cols_lower), None)

                value_candidates = ['revenue', 'cost', 'cb_cost', 'cb_percentage', 'cm_percentage', 'cm %', 'amount']
                value_cols = [df_cols_lower[cand] for cand in value_candidates if cand in df_cols_lower]

                if group_col and value_cols:
                    current_key = f"visualization_{st.session_state.render_counter}"
                    st.session_state.render_counter += 1

                    if any('percentage' in c.lower() or 'cm' in c.lower() for c in value_cols) and len(df) <= 15:
                        perc_col = next(c for c in value_cols if any(x in c.lower() for x in ['percentage', 'cm']))
                        fig = px.pie(df, names=group_col, values=perc_col, title=f"% by {group_col}", hole=0.35)
                        st.plotly_chart(fig, use_container_width=True, key=current_key)
                    else:
                        main_value = value_cols
                        fig = px.bar(
                            df.sort_values(main_value, ascending=False).head(15),
                            x=group_col, y=main_value,
                            title=f"{', '.join(main_value)} by {group_col}",
                            text_auto=True
                        )
                        fig.update_layout(xaxis={'categoryorder':'total descending'})
                        st.plotly_chart(fig, use_container_width=True, key=current_key)

                # Table
                st.markdown("### 📋 Results Table")
                st.dataframe(df, use_container_width=True)

            # SQL
            with st.expander("🛠️ Executed SQL Query", expanded=False):
                st.code(sql, language="sql")

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": insights,
                "sql": sql,
                "df": df
            })