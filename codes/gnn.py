# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split


def load_dataset(dataset_filepath, stats_filepath=None):
    """
    Load the dataset and optionally the stats from files.

    Args:
        dataset_filepath (str): Path to the saved dataset file
        stats_filepath (str, optional): Path to the saved stats file
        
    Returns:
        tuple: (dataset, stats) if stats_filepath is provided, otherwise just dataset
    """
    # Load dataset
    dataset = torch.load(dataset_filepath)
    print(f"Dataset loaded from: {dataset_filepath}")

    if stats_filepath:
        # Load stats
        with open(stats_filepath, 'r') as f:
            stats = json.load(f)
        print(f"Stats loaded from: {stats_filepath}")
        return dataset, stats

    return dataset

def create_pyg_data(simulation_data):
    """
    Convert a simulation dictionary to a PyTorch Geometric Data object.
    """
    return Data(
        x=simulation_data['x'],
        edge_index=simulation_data['edge_index'],
        edge_attr=simulation_data['edge_attr'],
        y=simulation_data['y'],
        edge_type=simulation_data['edge_type'],
        mask=simulation_data['mask']
    )

class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        # Define dimensions for GCN layers
        self.dim_hidden = dim_h * 4  # Hidden dimension for GCN layers
        
        # Add batch normalization
        self.batch_norm1 = torch.nn.BatchNorm1d(self.dim_hidden)
        self.batch_norm2 = torch.nn.BatchNorm1d(self.dim_hidden)
        self.batch_norm3 = torch.nn.BatchNorm1d(self.dim_hidden)
        
        # GCN layers with matching dimensions for residual connections
        self.gcn1 = GCNConv(dim_in, self.dim_hidden, improved=False, cached=False)
        self.gcn2 = GCNConv(self.dim_hidden, self.dim_hidden, improved=False, cached=False)
        self.gcn3 = GCNConv(self.dim_hidden, self.dim_hidden, improved=False, cached=False)
        
        # Final layers
        self.linear1 = torch.nn.Linear(self.dim_hidden, dim_h)
        self.linear2 = torch.nn.Linear(dim_h, dim_out)
        
        # Add dropout
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x, edge_index, edge_attr=None):
        # First GCN layer
        h = self.gcn1(x, edge_index, edge_attr)
        h = self.batch_norm1(h)
        h = torch.relu(h)
        h = self.dropout(h)
        
        # Second GCN layer with residual
        h2 = self.gcn2(h, edge_index, edge_attr)
        h2 = self.batch_norm2(h2)
        h2 = torch.relu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h  # Residual connection
        
        # Third GCN layer with residual
        h3 = self.gcn3(h2, edge_index, edge_attr)
        h3 = self.batch_norm3(h3)
        h3 = torch.relu(h3)
        h3 = self.dropout(h3)
        h3 = h3 + h2  # Residual connection
        
        # Final layers
        h = self.linear1(h3)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return h

def custom_loss(predictions, targets):
    """
    Combined loss function using MSE and Huber Loss.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
    
    Returns:
        Combined loss value
    """
    # MSE for overall accuracy
    mse_loss = F.mse_loss(predictions, targets)
    
    # Huber loss for robustness to outliers
    huber_loss = F.smooth_l1_loss(predictions, targets, beta=0.1)
    
    # Combine losses (adjust weights as needed)
    return 0.7 * mse_loss + 0.3 * huber_loss

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr)
        # Apply mask to predictions and targets
        masked_out = out[~data.mask]
        masked_targets = data.y[~data.mask]
        
        # Use custom loss
        loss = custom_loss(masked_out, masked_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            # Apply mask to predictions and targets
            masked_out = out[~data.mask]
            masked_targets = data.y[~data.mask]
            
            # Use custom loss
            loss = custom_loss(masked_out, masked_targets)
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Training setup
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss = train(model, train_loader, optimizer, device)
        
        # Validation
        val_loss = evaluate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return best_val_loss

def evaluate_predictions(model, loader, device, stats):
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            #out = data.x[:,1].unsqueeze(1)
            # Apply mask to predictions and targets
            masked_out = out[~data.mask]
            masked_targets = data.y[~data.mask]
            predictions.append(masked_out.cpu())
            actual.append(masked_targets.cpu())
    
    predictions = torch.cat(predictions, dim=0)
    actual = torch.cat(actual, dim=0)
    
    # Denormalize predictions and actual values
    #reservoir_head = stats['reservoir_head']
    #predictions = predictions * reservoir_head
    #actual = actual * reservoir_head
    
    # Calculate metrics using our custom loss
    combined_loss = custom_loss(predictions, actual).item()
    
    # Calculate individual metrics for detailed analysis
    mse = F.mse_loss(predictions, actual).item()
    mae = F.l1_loss(predictions, actual).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Calculate relative error
    max_p = predictions.max()
    min_p = predictions.min()
    # relative_error = torch.mean(torch.abs((predictions - actual) / (actual + 1e-6))).item()
    NMAE = torch.mean(torch.abs((predictions - actual) / (max_p - min_p + 1e-6))).item()
    MAPE = torch.mean((torch.abs(predictions - actual) / torch.abs(actual + 1e-6))).item()
    MaxPE = torch.max((torch.abs(predictions - actual) / torch.abs(actual + 1e-6))).item()

    accuracy10 = torch.mean(((torch.abs(predictions - actual) / torch.abs(actual + 1e-6))<0.1).float()).item()
    accuracy5 = torch.mean(((torch.abs(predictions - actual) / torch.abs(actual + 1e-6))<0.05).float()).item()



    
    print(f"\nTest Set Metrics:")
    print(f"Combined Loss: {combined_loss:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NMAE: {NMAE:.4f}")
    print(f"MAPE: {MAPE:.4f}")
    print(f"MaxPE: {MaxPE:.4f}")
    print(f"accuracy10: {accuracy10:.4f}")
    print(f"accuracy5: {accuracy5:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 10))
    plt.scatter(actual.numpy(), predictions.numpy(), alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel('Actual Pressure')
    plt.ylabel('Predicted Pressure')
    plt.title('Predicted vs Actual Pressures (Junction Nodes Only)')
    
    # Add R² score
    r2 = 1 - np.sum((predictions.numpy() - actual.numpy()) ** 2) / np.sum((actual.numpy() - np.mean(actual.numpy())) ** 2)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Print pressure range statistics
    print(f"\nPressure Range Statistics:")
    print(f"Actual Range: [{actual.min():.2f}, {actual.max():.2f}]")
    print(f"Predicted Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions, actual

# Load best model and evaluate






if __name__ == '__main__':
    # Load your dataset
    dataset_filepath = r"data/hanoi_ph_1_dataset.pt"
    stats_filepath = r"data/hanoi_ph_1_stats.json"
    dataset, stats = load_dataset(dataset_filepath, stats_filepath)
    # Print basic information about the loaded dataset
    print(f"\nDataset Information:")
    print(f"Number of simulations: {len(dataset)}")
    print(f"Number of nodes: {dataset[0]['x'].shape[0]}")
    print(f"Feature vector length: {dataset[0]['x'].shape[1]}")
    print(f"Number of edges: {dataset[0]['edge_index'].shape[1]}")
    # Convert dataset to PyG Data objects
    pyg_dataset = [create_pyg_data(sim_data) for sim_data in dataset]

    # Split dataset into train, validation, and test sets
    train_data, test_data = train_test_split(pyg_dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    print(f"\nData Split Information:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = GCN(dim_in=dataset[0]['x'].shape[1],
                dim_h=64,
                dim_out=1).to(device)
    
    # Create optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=0.001,
                               weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    best_val_loss = train_model(model, train_loader, val_loader, 
                               optimizer, scheduler, device, epochs=100)
    
    print(f'Best validation loss: {best_val_loss:.5f}')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")
    model.load_state_dict(torch.load('best_model.pt'))
    predictions, actual = evaluate_predictions(model, test_loader, device, stats) # Use test_loader instead of train_loader

