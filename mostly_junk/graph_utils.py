import pandas as pd
import networkx as nx

def load_primekg(csv_path='data/kg.csv'):
    """Loads and lowercases the PrimeKG dataset."""
    print("Loading PrimeKG into memory. This will take a moment...")
    try:
        primekg_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {csv_path}. Please ensure it is in the data folder.")
    
    # Force everything to lowercase to match inputs perfectly
    primekg_df['x_name'] = primekg_df['x_name'].astype(str).str.lower()
    primekg_df['y_name'] = primekg_df['y_name'].astype(str).str.lower()
    
    # Build the graph
    G = nx.from_pandas_edgelist(
        primekg_df, 
        source='x_name', 
        target='y_name', 
        edge_attr=['relation'], 
        create_using=nx.DiGraph()
    )
    print(f"Loaded PrimeKG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def get_item_mapping(icu_path='data/icu_items.csv', lab_path='data/lab_items.csv'):
    """Creates a dictionary to translate BigQuery IDs into medical strings."""
    item_mapping = {}
    try:
        icu_df = pd.read_csv(icu_path)
        lab_df = pd.read_csv(lab_path)
        
        for index, row in icu_df.iterrows():
            item_mapping[row['itemid']] = str(row['label']).lower()
        for index, row in lab_df.iterrows():
            item_mapping[row['itemid']] = str(row['label']).lower()
    except FileNotFoundError:
        print("Warning: Item mapping CSVs not found in data folder. Mapping might fail.")
        
    return item_mapping
