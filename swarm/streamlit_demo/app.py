import re
import pandas as pd
import streamlit as st
import json
import logging
import csv
import io
from string import Template
from streamlit_agraph import agraph, Node, Edge, Config
from swarm.base.kg_provider import KGProvider, KGConfig
from swarm.providers.llms.langchain.langchain import LangchainLLM
from swarm.base.llm_provider import LLMConfig
import fitz
import os
import plotly.express as px

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Dynamic Knowledge Graph and Chat with Data")
st.write("Upload a document (text or PDF) or chat with a preloaded dataset to analyze entities, rules, and relationships.")

# Initialize session state variables
if 'graph_ready' not in st.session_state:
    st.session_state.graph_ready = False  # Indicates if the knowledge graph is ready

if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = {"nodes": [], "edges": []}

if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

# Load CSV data from the project directory
@st.cache_data
def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset into a DataFrame."""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return pd.DataFrame()

# Load the dataset
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), '100_Sales.csv')
sales_data = load_csv_data(CSV_FILE_PATH)

# Check if dataset is large and warn user
if len(sales_data) > 100:
    st.warning("The dataset is large. Only the first 50 rows will be used for analysis.")

# Helper function for reading PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
    return text

# Prompts for knowledge graph generation and querying CSV data
QUESTION_ANSWERING_PROMPT = Template("""
You are an expert assistant tasked with answering the following question using **only** the information provided in the **Context**. Do not use any prior knowledge or external information.

---

**Question**: $query

---

**Context**:
Entities:
$entities

Rules:
$rules

Relationships:
$relationships

---

**Instructions**:
- Answer the question strictly based on the given context.
- Answer questions about KPIs also based on the given context.
- If the answer is not present in the context, respond exactly with "⚠ ***Not Enough Information*** ⚠".

**Answer**:
""")

DYNAMIC_KNOWLEDGE_GRAPH_PROMPT = Template("""
You are an expert assistant responsible for building a knowledge graph that illustrates relationships and processes for compliance in document data. Based on the following answer, extend the existing knowledge graph by:

1. Reusing nodes if they already exist (e.g., "Account Management System" should use the same ID and label if it was previously created).
2. Adding new unique nodes and relationships.
3. Creating relationships between existing nodes and any newly added nodes that are contextually related.
4. Ensuring directional relationships (arrows) where appropriate, based on dependencies, requirements, or flows.
5. Return only a json structure and nothing else.

### Existing Knowledge Graph:
- **Nodes**: ${existing_nodes}
- **Edges**: ${existing_edges}

---

**Answer**: $answer

**Extended Knowledge Graph** (in JSON format):
- **Nodes**: Each entity with attributes like ID, label, and type (e.g., "System," "Process," "Requirement").
- **Edges**: Each relationship between nodes, with properties like source, target, and label (e.g., "undergoes," "required by").
""")

CHAT_WITH_DATA_PROMPT = Template("""
You are an expert assistant tasked with analyzing the following data and answering user queries. Use only the data and extracted insights provided below.

---

**Extracted Insights**:
Entities: $entities
Rules: $rules
Relationships: $relationships

**Sales Data**:
$sales_data

---

**User Query**:
$query

---

**Instructions**:
- Answer the question using the provided data and extracted insights.
- If the answer cannot be derived from the context, respond with "⚠ ***Not Enough Information*** ⚠".
- Provide a concise and direct answer based strictly on the context.
- If calculations are required, provide only the final result unless intermediate steps are explicitly requested.

**Answer**:
""")

VISUALIZATION_AGENT_PROMPT = Template("""
You are an expert assistant tasked with dynamically generating a dashboard based on user queries about the provided dataset.

---

**Dataset Summary**:
$sales_data_preview

**User Query**:
$query

---

**Instructions**:
1. Use the dataset summary to identify key KPIs or metrics relevant to the query.
2. Determine the most appropriate visualizations to display these metrics, using charts like bar, line, scatter, or pie.
3. For any time-based visualizations (e.g., involving dates), ensure the data is sorted chronologically by the time column before generating the chart. For example a visualization like Profit Over Time you would sort Y- Axis Ship_Date dates to be in chronological order from 2010 to 2017.
4. Specify the layout, including:
   - KPIs as metric cards.
   - Charts in a grid layout.
5. If additional filters or transformations are needed, specify them clearly.
6. Output the response in the following JSON format:

```json
{
    "kpis": [
        {"name": "Total Revenue", "value": "123456", "unit": "USD"},
        {"name": "Total Profit", "value": "65432", "unit": "USD"}
    ],
    "charts": [
        {
            "chart_type": "Line Chart",
            "title": "Profit Over Time",
            "x_axis": "Ship_Date",
            "y_axis": "Total_Profit",
            "color": null,
            "filters": {}
        }
    ]
}
Ensure the response is concise and strictly adheres to the JSON format. Use the dataset summary effectively to create visualizations that answer the query.
""")

# Helper function for querying CSV data with LLM
def query_csv_with_llm(dataframe, query, llm_agent, entities=None, rules=None, relationships=None):
    # Define a map for query keywords and corresponding columns
    query_map = {
        "region": "Region",
        "country": "Country",
        "product": "Item_Type",
        "sales channel": "Sales_Channel",
        "revenue": "Total_Revenue",
        "profit": "Total_Profit",
    }

    # Find relevant columns based on query keywords
    relevant_columns = [col for key, col in query_map.items() if key in query.lower()]

    # Default to all columns if no specific match
    if not relevant_columns:
        relevant_columns = ["Region", "Country", "Item_Type", "Total_Revenue", "Total_Profit", "Sales_Channel"]

    # Filter DataFrame by relevant columns
    filtered_dataframe = dataframe[relevant_columns]

    # Limit the number of rows if the dataset is large
    filtered_dataframe = filtered_dataframe.head(50)

    # Convert the filtered data to CSV format
    sales_data_str = filtered_dataframe.to_csv(index=False)

    # Prepare document extractions as context
    entities_str = "\n".join([f"- {e['name']} ({e['type']})" for e in (entities or [])])
    rules_str = "\n".join([f"- {r['entity']}: {r['rule']}" for r in (rules or [])])
    relationships_str = "\n".join(
        [f"- {rel['subject']} {rel['predicate']} {rel['object']}" for rel in (relationships or [])])

    # Create the prompt
    prompt_content = CHAT_WITH_DATA_PROMPT.substitute(
        sales_data=sales_data_str,
        query=query,
        entities=entities_str,
        rules=rules_str,
        relationships=relationships_str
    )

    # Log the prompt for debugging
    logger.debug(f"Generated prompt:\n{prompt_content}")

    # Send the prompt to the LLM and return the response
    try:
        response = llm_agent.extract(prompt_content)
        return response
    except Exception as e:
        logger.error(f"Error querying data with LLM: {e}")
        return "⚠ ***An error occurred while processing your request. Please try again.*** ⚠"

def summarize_dataframe(dataframe):
    """
    Summarize the dataframe for inclusion in the LLM prompt.
    - Categorical columns: Unique values
    - Numeric columns: Min, Max, Mean
    """
    summary = {}
    for col in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            summary[col] = {
                "min": dataframe[col].min(),
                "max": dataframe[col].max(),
                "mean": dataframe[col].mean()
            }
        else:
            summary[col] = {"unique_values": dataframe[col].unique().tolist()[:5]}  # Limit to 5 unique values
    return summary

def generate_visualization(dataframe, query, llm_agent):
    # Summarize the full dataset
    sales_data_summary = summarize_dataframe(dataframe)
    sales_data_summary_str = json.dumps(sales_data_summary, indent=2)

    logger.debug(f"sales_data_summary:\n{sales_data_summary_str}")

    # Create the prompt for the visualization agent
    prompt_content = VISUALIZATION_AGENT_PROMPT.substitute(
        sales_data_preview=sales_data_summary_str,
        query=query
    )

    # Log the prompt
    logger.debug(f"Visualization Prompt:\n{prompt_content}")

    # Send the prompt to the LLM
    try:
        response = llm_agent.extract(prompt_content)
        logger.debug(f"Visualization Agent Response: {response}")

        # Parse response and validate
        if response.startswith("```json"):
            response = response.strip("```json").strip("```").strip()

        visualization_params = json.loads(response)

        if "kpis" not in visualization_params or "charts" not in visualization_params:
            raise ValueError("Missing 'kpis' or 'charts' in the response.")

        # Validate charts
        for chart in visualization_params.get("charts", []):
            required_keys = {"chart_type", "x_axis", "y_axis"}
            if not all(key in chart for key in required_keys):
                raise ValueError(f"Missing required keys in chart: {chart}")

        return visualization_params

    except json.JSONDecodeError as json_err:
        logger.error(f"JSON decoding error: {json_err}")
        return {"error": "The visualization agent returned an invalid response format. Please check your query."}
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return {"error": "Unable to process the query. Please try again."}

# Helper function
def get_edge_attribute(edge, primary, fallback):
    return getattr(edge, primary, getattr(edge, fallback, None))

# Step 1: Upload Document or Query Data
uploaded_file = st.file_uploader("Upload your document (text or PDF format) or interact with the dataset.", type=["txt", "pdf"])

# Step 2: Initialize Configurations
llm_config = LLMConfig(provider="openai")
llm_agent = LangchainLLM(config=llm_config)

# Initialize KG Provider with specific KGConfig for prompt file paths
kg_config = KGConfig(provider=None, kg_extraction_prompt="kg_extraction_prompt.jinja2")
kg_provider = KGProvider(config=kg_config, llm_agent=llm_agent)

# Step 3: Process the Document or Handle Queries
if uploaded_file and not st.session_state.extracted_data:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        text = None

    if text:
        st.write("### Uploaded Document Content:")
        st.write(text)

        with st.spinner("Extracting information..."):
            extraction_result = kg_provider.extract_and_store_relations(text)
            st.session_state.extracted_data = extraction_result

        st.success("Extraction complete!")

elif st.session_state.extracted_data:
    entities = st.session_state.extracted_data.get("entities", [])
    rules = st.session_state.extracted_data.get("rules", [])
    relationships = st.session_state.extracted_data.get("relationships", [])

    st.write("### Extracted Entities:")
    st.dataframe(pd.DataFrame(entities) if entities else pd.DataFrame({"No entities found": []}))

    st.write("### Extracted Rules:")
    st.dataframe(pd.DataFrame(rules) if rules else pd.DataFrame({"No rules found": []}))

    st.write("### Extracted Relationships:")
    st.dataframe(pd.DataFrame(relationships) if relationships else pd.DataFrame({"No relationships found": []}))

# Chat with Data Section
if st.checkbox("Chat with Data"):
    user_query = st.text_input("Enter your question about the dataset:")
    if user_query:
        with st.spinner("Analyzing data..."):
            response = query_csv_with_llm(
                sales_data,
                user_query,
                llm_agent,
                entities=st.session_state.extracted_data.get("entities"),
                rules=st.session_state.extracted_data.get("rules"),
                relationships=st.session_state.extracted_data.get("relationships")
            )
        st.write("### Answer:")
        st.write(response)


def validate_agent_response(response, dataframe):
    """
    Validate the response from the visualization agent.
    - Ensure required keys ('kpis', 'charts') are present.
    - Verify all columns and filters in the charts exist in the dataframe.
    """
    errors = []

    if "kpis" not in response:
        errors.append("Missing 'kpis' in the response.")
    elif not isinstance(response["kpis"], list):
        errors.append("'kpis' should be a list.")

    if "charts" not in response:
        errors.append("Missing 'charts' in the response.")
    elif not isinstance(response["charts"], list):
        errors.append("'charts' should be a list.")
    else:
        for chart in response["charts"]:
            required_keys = {"chart_type", "x_axis", "y_axis"}
            missing_keys = required_keys - chart.keys()
            if missing_keys:
                errors.append(f"Chart is missing required keys: {missing_keys}")
            else:
                # Verify columns exist in the dataframe
                for col in [chart["x_axis"], chart["y_axis"], chart.get("color")]:
                    if col and col not in dataframe.columns:
                        errors.append(f"Column '{col}' does not exist in the dataset.")

    return errors

def apply_filters(dataframe, filters):
    """
    Apply filters to the dataframe based on the agent's response.
    - Skip invalid filters or columns not present in the dataframe.
    """
    filtered_data = dataframe.copy()

    for col, values in filters.items():
        if col not in dataframe.columns:
            logger.warning(f"Filter column '{col}' does not exist. Skipping filter.")
            continue
        filtered_data = filtered_data[filtered_data[col].isin(values)]

    return filtered_data

def generate_charts(dataframe, charts):
    """
    Generate charts dynamically based on the agent's response.
    """
    for chart in charts:
        chart_type = chart.get("chart_type")
        title = chart.get("title", "Untitled Chart")
        x_axis = chart.get("x_axis")
        y_axis = chart.get("y_axis")
        color = chart.get("color")
        filters = chart.get("filters", {})

        filtered_data = apply_filters(dataframe, filters)

        # Sort data for time-based charts
        if chart_type in ["Line Chart", "Scatter Plot"] and pd.api.types.is_datetime64_any_dtype(filtered_data[x_axis]):
            filtered_data = filtered_data.sort_values(by=x_axis)

        # Generate the appropriate chart
        fig = None
        if chart_type == "Bar Chart":
            fig = px.bar(filtered_data, x=x_axis, y=y_axis, color=color, title=title)
        elif chart_type == "Line Chart":
            fig = px.line(filtered_data, x=x_axis, y=y_axis, color=color, title=title)
        elif chart_type == "Pie Chart":
            fig = px.pie(filtered_data, names=x_axis, values=y_axis, title=title)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(filtered_data, x=x_axis, y=y_axis, color=color, title=title)
        else:
            st.warning(f"Unsupported chart type: {chart_type}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

# Dynamic Dashboard Section
if st.checkbox("Enable Dashboard Agent"):
    user_query = st.text_input("Enter your query to generate a dashboard:")

    if user_query:
        with st.spinner("Generating dashboard..."):
            vis_response = generate_visualization(sales_data, user_query, llm_agent)

            if "error" in vis_response:
                st.error(vis_response["error"])
            else:
                validation_errors = validate_agent_response(vis_response, sales_data)
                if validation_errors:
                    st.error("Validation Errors in Agent Response:")
                    for error in validation_errors:
                        st.write(f"- {error}")
                else:
                    kpis = vis_response.get("kpis", [])
                    if kpis:
                        st.markdown("### Key Metrics")
                        kpi_cols = st.columns(len(kpis))
                        for i, kpi in enumerate(kpis):
                            kpi_cols[i].metric(kpi["name"], f"{kpi['value']} {kpi.get('unit', '')}")

                    charts = vis_response.get("charts", [])
                    if charts:
                        st.markdown("### Visualizations")
                        generate_charts(sales_data, charts)

# Visualize the Knowledge Graph
if st.session_state.graph_ready:
    st.write("### Knowledge Graph")
    config = Config(
        width=800, height=600, directed=True, nodeHighlightBehavior=True,
        highlightColor="#E0EFF2", link={'renderLabel': True}, staticGraph=False,
        collapsible=True, physics={"enabled": True}
    )
    agraph(
        nodes=st.session_state.knowledge_graph["nodes"],
        edges=st.session_state.knowledge_graph["edges"],
        config=config
    )