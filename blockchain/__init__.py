# blockchain/__init__.py
# This file makes blockchain a Python package

import json
import hashlib
import time
from datetime import datetime
import os

class BlockchainAudit:
    """Simple blockchain-based audit ledger for SURAKSHA"""
    
    def __init__(self, ledger_file='blockchain/ledger.json'):
        """Initialize blockchain"""
        self.ledger_file = ledger_file
        self.chain = []
        self.pending_transactions = []
        
        # Create blockchain directory
        ledger_dir = os.path.dirname(ledger_file)
        if ledger_dir:
            os.makedirs(ledger_dir, exist_ok=True)
        
        # Load or create
        self.load_or_create_chain()
    
    def create_genesis_block(self):
        """Create first block"""
        genesis = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'transactions': [
                {
                    'type': 'GENESIS',
                    'description': 'SURAKSHA Blockchain Audit System Initialized',
                    'data': {'system': 'SURAKSHA v1.0', 'purpose': 'Fraud Detection'}
                }
            ],
            'previous_hash': '0',
            'nonce': 0
        }
        genesis['hash'] = self.calculate_hash(genesis)
        self.chain.append(genesis)
        print("✅ Genesis block created")
    
    def load_or_create_chain(self):
        """Load existing or create new blockchain"""
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chain = data.get('chain', [])
                    self.pending_transactions = data.get('pending', [])
                if not self.chain:
                    self.create_genesis_block()
                print(f"✅ Loaded blockchain with {len(self.chain)} blocks")
            except Exception as e:
                print(f"⚠️  Error loading blockchain: {str(e)}")
                self.chain = []
                self.create_genesis_block()
        else:
            self.create_genesis_block()
    
    @staticmethod
    def calculate_hash(block):
        """Calculate SHA-256 hash"""
        block_copy = block.copy()
        block_copy.pop('hash', None)
        block_string = json.dumps(block_copy, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def add_transaction(self, tx_type, description, data=None):
        """Add transaction to pending pool"""
        transaction = {
            'type': tx_type,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        self.pending_transactions.append(transaction)
        print(f"   📝 Logged: {tx_type}")
    
    def mine_block(self):
        """Mine block with pending transactions"""
        if not self.pending_transactions:
            return False
        
        new_block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'transactions': self.pending_transactions.copy(),
            'previous_hash': self.chain[-1]['hash'] if self.chain else '0',
            'nonce': 0
        }
        
        new_block['hash'] = self.calculate_hash(new_block)
        self.chain.append(new_block)
        self.pending_transactions = []
        
        print(f"   ⛏️  Block #{new_block['index']} mined")
        return True
    
    def save_ledger(self):
        """Save blockchain to file"""
        try:
            ledger_data = {
                'chain': self.chain,
                'pending': self.pending_transactions,
                'last_updated': datetime.now().isoformat(),
                'total_blocks': len(self.chain),
                'total_transactions': sum(len(block['transactions']) for block in self.chain)
            }
            
            with open(self.ledger_file, 'w', encoding='utf-8') as f:
                json.dump(ledger_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Blockchain saved: {self.ledger_file}")
            return True
        except Exception as e:
            print(f"   ❌ Error saving blockchain: {str(e)}")
            return False
    
    def log_data_load(self, file_path, record_count):
        """Log data loading"""
        self.add_transaction('DATA_LOAD', 
                           f'Loaded {record_count:,} records from {file_path}',
                           {'file': file_path, 'records': record_count})
    
    def log_preprocessing(self, total_records, cleaned_records, anomalies_detected):
        """Log preprocessing"""
        self.add_transaction('PREPROCESSING',
                           f'Cleaned {cleaned_records:,} records, {anomalies_detected:,} anomalies',
                           {'total': total_records, 'cleaned': cleaned_records, 'anomalies': anomalies_detected})
    
    def log_graph_creation(self, num_nodes, num_edges, edge_types):
        """Log graph creation"""
        self.add_transaction('GRAPH_CREATION',
                           f'Built graph: {num_nodes:,} nodes, {num_edges:,} edges',
                           {'nodes': num_nodes, 'edges': num_edges, 'edge_types': edge_types})
    
    def log_model_training(self, best_accuracy, epochs_trained, training_time):
        """Log model training"""
        self.add_transaction('MODEL_TRAINING',
                           f'Trained R-GCN: {best_accuracy:.1%} accuracy in {epochs_trained} epochs',
                           {'accuracy': float(best_accuracy), 'epochs': epochs_trained, 'time': float(training_time)})
    
    def log_fraud_detection(self, fraud_rings_detected, nodes_flagged, detection_time, confidence_scores=None):
        """Log fraud detection"""
        data = {
            'fraud_rings': fraud_rings_detected,
            'nodes_flagged': nodes_flagged,
            'detection_time': float(detection_time)
        }
        if confidence_scores:
            data['avg_confidence'] = float(sum(confidence_scores) / len(confidence_scores))
        
        self.add_transaction('FRAUD_DETECTION',
                           f'Detected {fraud_rings_detected} fraud rings, {nodes_flagged:,} nodes flagged',
                           data)
    
    def log_report_generation(self, report_path):
        """Log report generation"""
        self.add_transaction('REPORT_GENERATION',
                           f'Generated fraud detection report: {report_path}',
                           {'report_path': report_path})
    
    def verify_chain_integrity(self):
        """Verify blockchain integrity"""
        print("\n   🔐 Verifying blockchain integrity...")
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Verify hash
            if current_block['hash'] != self.calculate_hash(current_block):
                print(f"   ❌ Block #{i}: Hash mismatch!")
                return False
            
            # Verify chain continuity
            if current_block['previous_hash'] != previous_block['hash']:
                print(f"   ❌ Block #{i}: Previous hash mismatch!")
                return False
        
        print(f"   ✅ Blockchain valid ({len(self.chain)} blocks)")
        return True
    
    def print_audit_log(self):
        """Print blockchain audit log"""
        print("\n" + "=" * 70)
        print("BLOCKCHAIN AUDIT LOG")
        print("=" * 70)
        
        total_transactions = 0
        for block in self.chain:
            print(f"\nBlock #{block['index']}")
            print(f"   Hash: {block['hash'][:32]}...")
            print(f"   Timestamp: {block['timestamp']}")
            print(f"   Transactions: {len(block['transactions'])}")
            
            for tx in block['transactions']:
                total_transactions += 1
                print(f"      - {tx['type']}: {tx['description']}")
        
        print(f"\n" + "=" * 70)
        print(f"Total Blocks: {len(self.chain)}")
        print(f"Total Transactions: {total_transactions}")
        print("=" * 70)


# Export for easy import
__all__ = ['BlockchainAudit']