import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
import pickle
import time
import os
import numpy as np

print("=" * 70)
print("CHECKING PYTORCH INSTALLATION")
print("=" * 70)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

class SurakshaGNN(torch.nn.Module):
    """
    Relational Graph Convolutional Network for fraud detection
    """
    
    def __init__(self, num_relations, in_channels=16, hidden_channels=32):
        super(SurakshaGNN, self).__init__()
        
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, 2, num_relations)
        self.dropout_rate = 0.2
        
    def forward(self, x, edge_index, edge_type):
        """Forward pass through R-GCN"""
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


def load_graph():
    """Load knowledge graph from pickle file"""
    print("\n📂 Loading knowledge graph...")
    
    try:
        with open('models/aadhaar_knowledge_graph.pkl', 'rb') as f:
            G = pickle.load(f)
        print(f"   ✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    except FileNotFoundError:
        print("   ❌ Error: Graph file not found!")
        print("   Please run: python code2_graph_construction.py first")
        return None
    except Exception as e:
        print(f"   ❌ Error loading graph: {str(e)}")
        return None


def convert_graph_to_pyg(G, df):
    """
    Convert NetworkX graph to PyTorch Geometric format
    """
    print("\n🔄 Converting graph to PyTorch Geometric format...")
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    print(f"   Created mapping for {len(node_to_idx)} nodes")
    
    # Create edge index and edge types
    edge_index = []
    edge_type = []
    
    relation_map = {
        'enrolled_by': 0,
        'located_at': 1,
        'shared_biometric': 2,
        'temporal_proximity': 3
    }
    
    print(f"   Processing {G.number_of_edges()} edges...")
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        relation = data.get('relation', 'enrolled_by')
        edge_type.append(relation_map.get(relation, 0))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    print(f"   Created tensor with {len(edge_index[0])} edges")
    
    # Create node features (16-dimensional random features)
    num_nodes = len(G.nodes())
    x = torch.randn(num_nodes, 16)
    print(f"   Created feature matrix: {x.shape}")
    
    # Create labels based on risk scores
    y = torch.zeros(num_nodes, dtype=torch.long)
    fraud_count = 0
    
    # Map UIDs to node indices
    uid_to_node = {}
    for node, idx in node_to_idx.items():
        if node.startswith('P_'):
            uid = int(node.replace('P_', ''))
            uid_to_node[uid] = idx
    
    # Assign fraud labels based on risk scores
    for _, row in df.iterrows():
        uid = int(row['anonymized_uid'])
        if uid in uid_to_node:
            node_idx = uid_to_node[uid]
            risk_score = row.get('final_risk_score', row.get('risk_score', 0))
            
            # High risk = fraud (1), Low risk = legitimate (0)
            if risk_score > 0.5:
                y[node_idx] = 1
                fraud_count += 1
    
    print(f"   Fraud labels assigned: {fraud_count}")
    
    # Create train/val masks (80/20 split)
    num_train = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask = ~train_mask
    
    print(f"   Train set: {train_mask.sum().item()} nodes")
    print(f"   Validation set: {val_mask.sum().item()} nodes")
    print(f"   Fraud rate: {(fraud_count / num_nodes) * 100:.2f}%")
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type,
                y=y, train_mask=train_mask, val_mask=val_mask)


def train_model(model, data, num_epochs=50, device='cpu'):
    """Training loop for R-GCN"""
    
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING R-GCN MODEL")
    print("=" * 70)
    
    print(f"\n🖥️  Using device: {device}")
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: 0.001")
    print(f"   Early Stopping Patience: {patience}")
    print(f"   Optimizer: Adam")
    print(f"   Device: {device}")
    
    print(f"\n▶️  Starting training...\n")
    
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # TRAINING PHASE
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_type)
        
        # Compute loss on training nodes
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # VALIDATION PHASE (every 5 epochs)
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=1)
                
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                
                val_accuracies.append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), 'models/best_suraksha_model.pt')
                    status = " ✨ (Best)"
                else:
                    patience_counter += 1
                    status = ""
                
                print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}{status}')
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Training Complete!")
    print(f"   Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Final Epoch: {epoch + 1}")
    
    # Save training metrics
    metrics = {
        'best_val_acc': float(best_val_acc),
        'training_time': float(training_time),
        'epochs_trained': int(epoch + 1),
        'final_epoch': int(epoch + 1),
        'train_losses': [float(x) for x in train_losses],
        'val_accuracies': [float(x) for x in val_accuracies]
    }
    
    with open('models/training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\n📁 Files saved:")
    print(f"   ✅ models/training_metrics.pkl")
    print(f"   ✅ models/best_suraksha_model.pt")
    
    return model, metrics


def main():
    print("=" * 70)
    print("SURAKSHA - R-GCN MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load graph
    G = load_graph()
    if G is None:
        return
    
    # Load processed data
    print("\n📂 Loading processed data...")
    try:
        df = pd.read_csv('data/processed_real_aadhaar_data.csv')
        print(f"   ✅ Loaded {len(df):,} records")
    except FileNotFoundError:
        print("   ❌ Error: Data file not found!")
        print("   Please run: python code1_preprocessing_real.py first")
        return
    
    # Convert graph to PyTorch Geometric format
    data = convert_graph_to_pyg(G, df)
    
    # Initialize model
    print("\n🤖 Initializing R-GCN Model...")
    model = SurakshaGNN(num_relations=4, in_channels=16, hidden_channels=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model Parameters: {total_params:,}")
    print(f"   Model Architecture:")
    print(f"   - Conv1: 16 → 32 (channels)")
    print(f"   - Dropout: 20%")
    print(f"   - Conv2: 32 → 2 (classes)")
    
    # Train model
    trained_model, metrics = train_model(model, data, num_epochs=50, device=device)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ STEP 3 COMPLETE - Ready for Step 4 (Fraud Detection)")
    print("=" * 70)
    print(f"\n📊 Training Summary:")
    print(f"   Best Validation Accuracy: {metrics['best_val_acc']:.1%}")
    print(f"   Epochs Trained: {metrics['epochs_trained']}")
    print(f"   Training Time: {metrics['training_time']/60:.2f} minutes")
    print(f"\n🔄 Next step:")
    print(f"   Run: python code4_fraud_detection.py")


if __name__ == "__main__":
    main()