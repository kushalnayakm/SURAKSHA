import pandas as pd
import networkx as nx
from collections import defaultdict
import os
import pickle

# ==============================
# CONFIGURATION
# ==============================
TIME_WINDOW_SECONDS = 3600        # 1 hour
MAX_TEMPORAL_NEIGHBORS = 5        # HARD CAP (prevents explosion)

# ==============================
# MAIN FUNCTION
# ==============================
def build_knowledge_graph():

    print("=" * 60)
    print("STEP 2: BUILDING KNOWLEDGE GRAPH (STABLE & OPTIMIZED)")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    print("\n📂 Loading processed enrollment data...")
    df = pd.read_csv('data/processed_real_aadhaar_data.csv')
    print(f"   ✅ Loaded {len(df):,} records")

    df['enrollment_timestamp'] = pd.to_datetime(df['enrollment_timestamp'])

    # ------------------------------
    # INITIALIZE GRAPH
    # ------------------------------
    print("\n🔗 Initializing graph...")
    G = nx.Graph()

    # ------------------------------
    # SHARED BIOMETRIC INDEX
    # ------------------------------
    print("\n🔐 Indexing biometric hashes...")
    biometric_groups = df.groupby('biometric_hash')['anonymized_uid'].apply(list)
    shared_biometrics = {
        h: uids for h, uids in biometric_groups.items() if len(uids) > 1
    }
    print(f"   Shared biometric hashes: {len(shared_biometrics)}")

    # ------------------------------
    # ADD NODES & STATIC EDGES
    # ------------------------------
    print("\n🏗️  Adding nodes and static edges...")

    for idx, row in df.iterrows():

        uid = int(row['anonymized_uid'])
        op_id = int(row['operator_id'])
        center_id = int(row['center_id'])

        p = f"P_{uid}"
        o = f"O_{op_id}"
        c = f"C_{center_id}"

        # Nodes
        G.add_node(p,
                   node_type='person',
                   risk_score=float(row.get('risk_score', 0)),
                   doc_quality=float(row.get('document_quality_normalized', 0.5)))

        G.add_node(o, node_type='operator')
        G.add_node(c, node_type='center')

        # ENROLLED_BY
        G.add_edge(p, o,
                   relation='enrolled_by',
                   timestamp=row['enrollment_timestamp'],
                   weight=1.0)

        # LOCATED_AT
        G.add_edge(p, c,
                   relation='located_at',
                   weight=1.0)

        # SHARED_BIOMETRIC
        bio = row['biometric_hash']
        if bio in shared_biometrics:
            for other_uid in shared_biometrics[bio]:
                if other_uid != uid:
                    other_p = f"P_{other_uid}"
                    if not G.has_edge(p, other_p):
                        G.add_edge(p, other_p,
                                   relation='shared_biometric',
                                   weight=0.9)

        if (idx + 1) % 100000 == 0:
            print(f"   Processed {idx + 1:,}/{len(df):,}")

    # ------------------------------
    # TEMPORAL PROXIMITY (SAFE)
    # ------------------------------
    print("\n⏱️  Adding temporal proximity edges (SAFE MODE)...")

    df['time_bucket'] = df['enrollment_timestamp'].dt.floor('H')
    temporal_edges = 0

    for bucket, group in df.groupby('time_bucket'):

        uids = group['anonymized_uid'].values
        times = group['enrollment_timestamp'].values

        for i in range(len(uids)):
            p_i = f"P_{uids[i]}"

            neighbors = 0
            for j in range(i + 1, len(uids)):
                if neighbors >= MAX_TEMPORAL_NEIGHBORS:
                    break

                time_diff = abs((times[j] - times[i]).astype('timedelta64[s]').astype(int))
                if time_diff == 0 or time_diff > TIME_WINDOW_SECONDS:
                    continue

                p_j = f"P_{uids[j]}"
                weight = min(1.0 / max(time_diff, 1), 1.0)

                G.add_edge(p_i, p_j,
                           relation='temporal_proximity',
                           time_diff=time_diff,
                           weight=weight)

                temporal_edges += 1
                neighbors += 1

    print(f"   Temporal proximity edges added: {temporal_edges:,}")

    # ------------------------------
    # GRAPH STATS
    # ------------------------------
    print("\n📊 GRAPH STATISTICS")
    print("=" * 60)

    person_nodes = sum(1 for n in G.nodes if G.nodes[n]['node_type'] == 'person')
    operator_nodes = sum(1 for n in G.nodes if G.nodes[n]['node_type'] == 'operator')
    center_nodes = sum(1 for n in G.nodes if G.nodes[n]['node_type'] == 'center')

    print(f"Person Nodes:   {person_nodes:,}")
    print(f"Operator Nodes: {operator_nodes:,}")
    print(f"Center Nodes:   {center_nodes:,}")
    print(f"Total Nodes:    {G.number_of_nodes():,}")
    print(f"Total Edges:    {G.number_of_edges():,}")

    # ------------------------------
    # SAVE GRAPH
    # ------------------------------
    print("\n💾 Saving graph...")

    graph_path = 'models/aadhaar_knowledge_graph.gpickle'
    nx.write_gpickle(G, graph_path)

    graph_info = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'persons': person_nodes,
        'operators': operator_nodes,
        'centers': center_nodes,
        'temporal_edges': temporal_edges
    }

    with open('models/graph_info.pkl', 'wb') as f:
        pickle.dump(graph_info, f)

    print(f"   ✅ Saved graph: {graph_path}")
    print("✅ STEP 2 COMPLETE")

    return G


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    build_knowledge_graph()
