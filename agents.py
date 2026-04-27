import time
import networkx as nx
import google.generativeai as genai
from mostly_junk.graph_utils import load_primekg, get_item_mapping

def setup_gemini(api_key):
    """Initializes the Gemini model using the provided API key."""
    genai.configure(api_key=api_key)
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model = 'models/gemini-2.5-flash' if 'models/gemini-2.5-flash' in available_models else available_models[0]
    return genai.GenerativeModel(target_model)

def tier1_gp_agent(llm, items):
    """Tier 1: Uses Gemini, with automatic rate-limit handling."""
    prompt = f"""
    You are a clinical triage AI. The patient is exhibiting abnormalities related to these items: {items}.
    Identify the single most likely broad clinical context or disease category (e.g., 'sepsis', 'cardiac failure', 'acute kidney injury').
    Respond with ONLY the disease name in lowercase. Do not include any other text.
    """
    for attempt in range(3):
        try:
            response = llm.generate_content(prompt)
            return response.text.strip().lower()
        except Exception as e:
            if '429' in str(e) or 'Quota' in str(e) or 'TooManyRequests' in str(e):
                print(f"⚠️ API busy. Automatically waiting 60 seconds... (Attempt {attempt + 1}/3)")
                time.sleep(60)
            else:
                raise e
    return "unknown_context"

def tier2_specialized_agent(graph, items, target_disease):
    """Tier 2: Traverses PrimeKG using ONLY clinically relevant pathways."""
    subgraph_data = []
    valid_relations = [
        'indication', 'off-label use', 'disease_phenotype_positive', 
        'disease_disease', 'phenotype_phenotype'
    ]
    
    valid_edges = [(u, v) for u, v, data in graph.edges(data=True) if data['relation'] in valid_relations]
    clinical_graph = graph.edge_subgraph(valid_edges)
    
    for item in items:
        try:
            path = nx.shortest_path(clinical_graph, source=item, target=target_disease)
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i+1]
                relation = clinical_graph[source_node][target_node]['relation']
                subgraph_data.append({
                    "source": source_node, "relation": relation, "target": target_node
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass # Fails silently so it doesn't spam Nishal's console
            
    return subgraph_data

def run_graphrag_pipeline(api_key, flagged_ids, graph, item_mapping):
    """
    THE MAIN WRAPPER: Nishal imports and calls this function directly.
    """
    llm = setup_gemini(api_key)
    
    # Translate numeric IDs to medical strings
    patient_items = [item_mapping[i] for i in flagged_ids if i in item_mapping]
    if not patient_items:
        return {"error": "No valid mappings found for the provided IDs."}

    # Run the agents
    disease_context = tier1_gp_agent(llm, patient_items)
    medical_subgraph = tier2_specialized_agent(graph, patient_items, disease_context)
    
    return {
        "identified_context": disease_context,
        "input_metadata": patient_items,
        "subgraph": medical_subgraph
    }
