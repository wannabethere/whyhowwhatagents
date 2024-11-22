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

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Dynamic Knowledge Graph from Document Analysis")
st.write("Upload a document to extract entities, rules, and relationships, and dynamically build a knowledge graph.")

# Initialize session state variables
if 'graph_ready' not in st.session_state:
    st.session_state.graph_ready = False  # Indicates if the knowledge graph is ready

if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = {"nodes": [], "edges": []}

if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

# Prompts for knowledge graph generation and prediction
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

PREDICT_FUTURE_PATHS_PROMPT = Template("""
You are an expert in dynamic knowledge graph construction. Based on the current state of the knowledge graph and the user's recent interactions, predict the next logical questions and potential graph extensions. Use the following format:

### Existing Knowledge Graph:
- Nodes: ${existing_nodes}
- Edges: ${existing_edges}

### User Context:
- Recent Question: "$recent_question"
- Recent Answer: "$recent_answer"

### Instructions:
1. Suggest 3-5 logical follow-up questions the user might ask to expand or refine the knowledge graph.
2. Predict the potential additions to the graph (nodes, edges) if each follow-up question were asked and answered.

Provide your response in JSON format with the following structure:
{
    "predicted_questions": [
        {"question": "Question 1", "potential_extensions": [{"nodes": [...], "edges": [...]}]},
        {"question": "Question 2", "potential_extensions": [{"nodes": [...], "edges": [...]}]},
        ...
    ]
}
""")

# Helper function
def get_edge_attribute(edge, primary, fallback):
    """
    Safely retrieves an attribute from the edge object.
    If the primary attribute doesn't exist, the fallback is used.

    :param edge: Edge object or dictionary
    :param primary: Primary attribute to check
    :param fallback: Fallback attribute to use if primary is missing
    :return: Attribute value or None
    """
    return getattr(edge, primary, getattr(edge, fallback, None))

# Step 1: Upload Document
uploaded_file = st.file_uploader("Upload your document (text format)", type=["txt"])

# Step 2: Initialize Configurations
llm_config = LLMConfig(provider="openai")
llm_agent = LangchainLLM(config=llm_config)

# Initialize KG Provider with specific KGConfig for prompt file paths
kg_config = KGConfig(provider=None, kg_extraction_prompt="kg_extraction_prompt.jinja2")
kg_provider = KGProvider(config=kg_config, llm_agent=llm_agent)

# Step 3: Process the Document
if uploaded_file and not st.session_state.extracted_data:
    text = uploaded_file.read().decode("utf-8")
    st.write("### Uploaded Document Content:")
    st.write(text)

    with st.spinner("Extracting information..."):
        extraction_result = kg_provider.extract_and_store_relations(text)
        st.session_state.extracted_data = extraction_result

    st.success("Extraction complete!")

# If the document is already extracted, get data from session state
if st.session_state.extracted_data:
    entities = st.session_state.extracted_data.get("entities", [])
    rules = st.session_state.extracted_data.get("rules", [])
    relationships = st.session_state.extracted_data.get("relationships", [])

    # Display extracted data
    st.write("### Extracted Entities:")
    st.dataframe(pd.DataFrame(entities) if entities else pd.DataFrame({"No entities found": []}))

    st.write("### Extracted Rules:")
    st.dataframe(pd.DataFrame(rules) if rules else pd.DataFrame({"No rules found": []}))

    st.write("### Extracted Relationships:")
    st.dataframe(pd.DataFrame(relationships) if relationships else pd.DataFrame({"No relationships found": []}))

    # Query User for a Question
    question = st.text_input("Enter your question:")
    if question:
        # Prepare context for the LLM
        entities_context = "\n".join([f"- {entity['name']} ({entity['type']})" for entity in entities])
        rules_context = "\n".join([f"- {rule['entity']}: {rule['rule']}" for rule in rules])
        relationships_context = "\n".join([f"- {relation['subject']} {relation['predicate']} {relation['object']}" for relation in relationships])

        query_input = QUESTION_ANSWERING_PROMPT.substitute(
            query=question,
            entities=entities_context,
            rules=rules_context,
            relationships=relationships_context
        )

        with st.spinner("Getting the answer..."):
            response = llm_agent.extract(query_input)

        st.write("### Answer:")
        st.write(response)

        # Extend Knowledge Graph
        graph_response = None
        try:
            existing_nodes = [
                {"id": node.id, "label": node.label, "type": getattr(node, "type", "Entity")}
                for node in st.session_state.knowledge_graph["nodes"]
            ]
            existing_edges = [
                {
                    "source": get_edge_attribute(edge, 'source', 'node_from'),
                    "target": get_edge_attribute(edge, 'target', 'node_to'),
                    "label": edge.label
                }
                for edge in st.session_state.knowledge_graph["edges"]
            ]

            graph_input = DYNAMIC_KNOWLEDGE_GRAPH_PROMPT.substitute(
                answer=response,
                existing_nodes=json.dumps(existing_nodes, indent=4),
                existing_edges=json.dumps(existing_edges, indent=4)
            )

            with st.spinner("Building knowledge graph..."):
                graph_response = llm_agent.extract(graph_input)

            if graph_response:
                graph_data = json.loads(re.search(r'```json\n(\{.*?\})\n```', graph_response, re.DOTALL).group(1))
                for node in graph_data.get("nodes", []):
                    if node["id"] not in {n.id for n in st.session_state.knowledge_graph["nodes"]}:
                        st.session_state.knowledge_graph["nodes"].append(
                            Node(id=node["id"], label=node["label"], shape="ellipse")
                        )
                for edge in graph_data.get("edges", []):
                    edge_tuple = (
                        edge.get("source"),
                        edge.get("target"),
                        edge.get("label", "related to")
                    )
                    if all(edge_tuple) and edge_tuple not in {
                        (get_edge_attribute(e, 'source', 'node_from'), get_edge_attribute(e, 'target', 'node_to'), e.label)
                        for e in st.session_state.knowledge_graph["edges"]
                    }:
                        st.session_state.knowledge_graph["edges"].append(
                            Edge(
                                source=edge.get("source"),
                                target=edge.get("target"),
                                label=edge.get("label", "related to")
                            )
                        )
                st.session_state.graph_ready = True  # Mark graph as ready
        except Exception as e:
            st.error("Error building knowledge graph.")
            logger.error(f"Error: {e}")

    # Visualize the Knowledge Graph
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

    # Predict Future Paths (only if the knowledge graph is ready)
    if st.session_state.graph_ready:
        st.write("### Predict Future Questions and Paths")
        prediction_input = PREDICT_FUTURE_PATHS_PROMPT.substitute(
            existing_nodes=json.dumps(existing_nodes, indent=4),
            existing_edges=json.dumps(existing_edges, indent=4),
            recent_question=question,
            recent_answer=response
        )

        with st.spinner("Predicting future paths..."):
            prediction_response = llm_agent.extract(prediction_input)

        try:
            # Log the raw response for debugging purposes
            logger.debug(f"Raw prediction response: {prediction_response}")

            # Strip any whitespace and validate response content
            if not prediction_response.strip():
                raise ValueError("Prediction response is empty.")

            # Attempt to extract JSON from response
            # Match JSON structure within triple backticks
            json_match = re.search(r'```json\n(.*?)\n```', prediction_response, re.DOTALL)

            if not json_match:
                raise ValueError("No valid JSON block found in the response.")

            # Parse the JSON content
            prediction_data = json.loads(json_match.group(1))

            # Validate the structure of the JSON response
            if "predicted_questions" not in prediction_data:
                raise ValueError("Missing 'predicted_questions' in the response JSON.")

            # Display predicted questions and paths
            st.write("### Suggested Questions and Paths")
            for prediction in prediction_data["predicted_questions"]:
                st.write(f"- **{prediction['question']}**")
                st.write(f"  - **Potential Extensions**: {prediction['potential_extensions']}")

        except json.JSONDecodeError as e:
            st.error("Failed to parse prediction data. Check logs for details.")
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Raw prediction response: {prediction_response}")

        except Exception as e:
            st.error("Error predicting future paths.")
            logger.error(f"Error: {e}")
            logger.debug(f"Raw prediction response: {prediction_response}")

    # Export Results as CSV
    if st.button("Download Results as CSV"):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["subject", "predicate", "object", "properties"])
        writer.writeheader()
        for edge in st.session_state.knowledge_graph["edges"]:
            writer.writerow({
                "subject": get_edge_attribute(edge, 'source', 'node_from'),
                "predicate": edge.label,
                "object": get_edge_attribute(edge, 'target', 'node_to'),
                "properties": json.dumps({}, indent=2)  # Add any additional properties if required
            })

        st.download_button(
            label="Download CSV",
            data=output.getvalue().encode("utf-8"),
            file_name="knowledge_graph.csv",
            mime="text/csv"
        )

else:
    st.write("Please upload a text file to begin extraction.")
