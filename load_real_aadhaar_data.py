import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import os
import pickle
import time

def build_knowledge_graph():
    """
    Build heterogeneous knowledge graph - OPTIMIZED VERSION
    """
    
    print("=" * 70)
    print("STEP 2: BUILDING KNOWLEDGE GRAPH (OPTIMIZED)")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    start_total = time.time()
    
    # =====================================
    # LOAD DATA
    # =====================================
    print("\n📂 Loading processed enrollment data...")
    start = time.time()
    
    df = pd.read_csv('data/processed_real_aadhaar_data.csv')
    print(f"   ✅ Loaded {len(df):,} records")
    
    df['enrollment_timestamp'] = pd.to_datetime(df['enrollment_timestamp'])
    print(f"   ⏱️  Load time: {time.time() - start:.2f}s")
    
    # =====================================
    # INITIALIZE GRAPH
    # =====================================
    print("\n🔗 Initializing graph...")
    G = nx.Graph()
    
    # =====================================
    # INDEX BIOMETRIC HASHES (PRE-COMPUTATION)
    # =====================================
    print("\n🔐 Indexing biometric hashes...")
    start = time.time()
    
    biometric_groups = df.groupby('biometric_hash')['anonymized_uid'].apply(list)
    shared_biometrics = {
        h: uids for h, uids in biometric_groups.items() if len(uids) > 1
    }
    print(f"   Shared biometric hashes: {len(shared_biometrics)}")
    print(f"   ⏱️  Indexing time: {time.time() - start:.2f}s")
    
    # =====================================
    # ADD NODES & STATIC EDGES (FAST)
    # =====================================
    print("\n🏗️  Adding nodes and static edges...")
    start = time.time()
    
    # Convert to dict for faster lookups
    node_attrs = {}
    edges_to_add = []
    shared_bio_edges_count = 0
    
    for idx, row in df.iterrows():
        uid = int(row['anonymized_uid'])
        op_id = int(row['operator_id'])
        center_id = int(row['center_id'])
        
        p = f"P_{uid}"
        o = f"O_{op_id}"
        c = f"C_{center_id}"
        
        # Store node attributes
        node_attrs[p] = {
            'node_type': 'person',
            'risk_score': float(row.get('final_risk_score', row.get('risk_score', 0))),
            'doc_quality': float(row.get('document_quality_normalized', 0.5))
        }
        
        if o not in node_attrs:
            node_attrs[o] = {'node_type': 'operator'}
        
        if c not in node_attrs:
            node_attrs[c] = {'node_type': 'center'}
        
        # Collect edges to add (batch mode is faster)
        edges_to_add.append((p, o, {'relation': 'enrolled_by', 'timestamp': row['enrollment_timestamp'], 'weight': 1.0}))
        edges_to_add.append((p, c, {'relation': 'located_at', 'weight': 1.0}))
        
        # SHARED_BIOMETRIC edges
        bio = row['biometric_hash']
        if bio in shared_biometrics:
            for other_uid in shared_biometrics[bio]:
                if other_uid != uid:
                    other_p = f"P_{other_uid}"
                    edges_to_add.append((p, other_p, {'relation': 'shared_biometric', 'weight': 0.9}))
                    shared_bio_edges_count += 1
        
        if (idx + 1) % 100000 == 0:
            print(f"   Processed {idx + 1:,}/{len(df):,} records...")
    
    # Add all nodes at once (BATCH MODE - MUCH FASTER)
    for node_id, attrs in node_attrs.items():
        G.add_node(node_id, **attrs)
    
    # Add edges in batch (remove duplicates first)
    unique_edges = {}
    for u, v, attr in edges_to_add:
        key = tuple(sorted([u, v]))
        if key not in unique_edges:
            unique_edges[key] = (u, v, attr)
    
    for u, v, attr in unique_edges.values():
        G.add_edge(u, v, **attr)
    
    print(f"   ✅ Added {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    print(f"   ⏱️  Node/Edge creation time: {time.time() - start:.2f}s")
    
    # =====================================
    # TEMPORAL PROXIMITY (OPTIMIZED)
    # =====================================
    print("\n⏱️  Adding temporal proximity edges (FAST MODE)...")
    start = time.time()
    
    # Sort by timestamp for efficient processing
    df_sorted = df.sort_values('enrollment_timestamp').reset_index(drop=True)
    
    temporal_edges_count = 0
    TIME_WINDOW = 3600  # 1 hour in seconds
    MAX_NEIGHBORS_PER_NODE = 5  # Hard cap to prevent explosion
    
    # Group by hour for efficient processing
    df_sorted['hour'] = df_sorted['enrollment_timestamp'].dt.floor('H')
    
    for hour_group, group_df in df_sorted.groupby('hour'):
        
        # Within each hour, find temporal neighbors
        times = group_df['enrollment_timestamp'].values
        uids = group_df['anonymized_uid'].values
        
        for i in range(len(uids)):
            p_i = f"P_{int(uids[i])}"
            neighbor_count = 0
            
            # Only look at next records (within same hour)
            for j in range(i + 1, min(i + 50, len(uids))):  # Limit comparisons
                if neighbor_count >= MAX_NEIGHBORS_PER_NODE:
                    break
                
                # Calculate time difference
                time_diff_seconds = int((times[j] - times[i]) / np.timedelta64(1, 's'))
                
                # Skip if outside time window or zero diff
                if time_diff_seconds <= 0 or time_diff_seconds > TIME_WINDOW:
                    continue
                
                p_j = f"P_{int(uids[j])}"
                
                # Add edge only if doesn't exist
                if not G.has_edge(p_i, p_j):
                    weight = min(1.0 / max(time_diff_seconds, 1), 1.0)
                    G.add_edge(p_i, p_j,
                              relation='temporal_proximity',
                              time_diff=time_diff_seconds,
                              weight=weight)
                    temporal_edges_count += 1
                    neighbor_count += 1
    
    print(f"   ✅ Added {temporal_edges_count:,} temporal proximity edges")
    print(f"   ⏱️  Temporal edge time: {time.time() - start:.2f}s")
    
    # =====================================
    # GRAPH STATISTICS
    # =====================================
    print("\n" + "=" * 70)
    print("📊 GRAPH STATISTICS")
    print("=" * 70)
    
    person_count = sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'person')
    operator_count = sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'operator')
    center_count = sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'center')
    
    enrolled_by = sum(1 for u, v, d in G.edges(data=True) if d.get('relation') == 'enrolled_by')
    located_at = sum(1 for u, v, d in G.edges(data=True) if d.get('relation') == 'located_at')
    shared_bio = sum(1 for u, v, d in G.edges(data=True) if d.get('relation') == 'shared_biometric')
    temporal = sum(1 for u, v, d in G.edges(data=True) if d.get('relation') == 'temporal_proximity')
    
    print(f"\n🔵 Node Counts:")
    print(f"   Person Nodes:          {person_count:,}")
    print(f"   Operator Nodes:        {operator_count:,}")
    print(f"   Center Nodes:          {center_count:,}")
    print(f"   Total Nodes:           {G.number_of_nodes():,}")
    
    print(f"\n🔗 Edge Counts:")
    print(f"   ENROLLED_BY edges:     {enrolled_by:,}")
    print(f"   LOCATED_AT edges:      {located_at:,}")
    print(f"   SHARED_BIOMETRIC edges:{shared_bio:,}")
    print(f"   TEMPORAL_PROXIMITY edges: {temporal:,}")
    print(f"   Total Edges:           {G.number_of_edges():,}")
    
    avg_degree = (2 * G.number_of_edges()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"\n📈 Graph Metrics:")
    print(f"   Average Degree:        {avg_degree:.2f}")
    print(f"   Density:               {nx.density(G):.6f}")
    
    # =====================================
    # SAVE GRAPH (FIXED - Use pickle instead)
    # =====================================
    print(f"\n💾 Saving knowledge graph...")
    start = time.time()
    
    output_path = 'models/aadhaar_knowledge_graph.pkl'
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✅ Saved: {output_path}")
    except Exception as e:
        print(f"   ❌ Error saving graph: {str(e)}")
        print(f"   Trying alternative method...")
        # Fallback: save as JSON
        try:
            import json
            from networkx.readwrite import json_graph
            with open('models/aadhaar_knowledge_graph.json', 'w') as f:
                json.dump(json_graph.node_link_data(G), f)
            print(f"   ✅ Saved as JSON instead: models/aadhaar_knowledge_graph.json")
            output_path = 'models/aadhaar_knowledge_graph.json'
        except Exception as e2:
            print(f"   ❌ Error with JSON save: {str(e2)}")
            return None
    
    # Save graph info
    graph_info = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'person_nodes': person_count,
        'operator_nodes': operator_count,
        'center_nodes': center_count,
        'enrolled_by_edges': enrolled_by,
        'located_at_edges': located_at,
        'shared_biometric_edges': shared_bio,
        'temporal_proximity_edges': temporal,
        'graph_file': output_path
    }
    
    with open('models/graph_info.pkl', 'wb') as f:
        pickle.dump(graph_info, f)
    
    print(f"   ✅ Saved: models/graph_info.pkl")
    print(f"   ⏱️  Save time: {time.time() - start:.2f}s")
    
    # =====================================
    # FINAL SUMMARY
    # =====================================
    total_time = time.time() - start_total
    
    print("\n" + "=" * 70)
    print("✅ STEP 2 COMPLETE")
    print("=" * 70)
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\n📁 Files saved:")
    print(f"   • models/aadhaar_knowledge_graph.pkl")
    print(f"   • models/graph_info.pkl")
    print(f"\n🔄 Next step:")
    print(f"   Run: python code3_rgcn_training.py")
    
    return G


if __name__ == "__main__":
    G = build_knowledge_graph()