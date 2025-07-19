import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last output
        out = self.fc(out)
        return out

class StockPredictor:
    def __init__(self, symbol='AAPL', period='2y', lookback=60):
        self.symbol = symbol
        self.period = period
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        print(f"Fetching {self.symbol} data for {self.period}...")
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(period=self.period)
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self):
        """Preprocess data for LSTM"""
        # Use Close price for prediction
        prices = np.array(self.data['Close']).reshape(-1, 1)
        
        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(scaled_prices)):
            X.append(scaled_prices[i-self.lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split data (80% train, 20% test)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
        self.X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
        self.y_train = torch.FloatTensor(y_train).to(device)
        self.y_test = torch.FloatTensor(y_test).to(device)
        
        print(f"Training data: {self.X_train.shape}")
        print(f"Testing data: {self.X_test.shape}")
        
    def build_model(self, hidden_size=50, num_layers=2, dropout=0.2):
        """Build LSTM model"""
        self.model = LSTMModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        ).to(device)
        
        print(f"Model architecture:\n{self.model}")
        return self.model
        
    def train_model(self, epochs=100, batch_size=32, lr=0.001):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()
        print("Training model...")
        
        # Create data loaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        train_losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        self.train_losses = train_losses
        print("Training completed!")
        
    def predict(self):
        """Make predictions on test data"""
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.X_train)
            test_pred = self.model(self.X_test)
        
        # Convert back to original scale
        train_pred = self.scaler.inverse_transform(train_pred.cpu().numpy())
        test_pred = self.scaler.inverse_transform(test_pred.cpu().numpy())
        y_train_actual = self.scaler.inverse_transform(self.y_train.cpu().numpy().reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1))
        
        return train_pred, test_pred, y_train_actual, y_test_actual
        
    def evaluate_model(self):
        """Evaluate model performance"""
        train_pred, test_pred, y_train_actual, y_test_actual = self.predict()
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        
        print(f"\n--- Model Performance ---")
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Train MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
    def plot_results(self):
        """Plot training loss and predictions"""
        train_pred, test_pred, y_train_actual, y_test_actual = self.predict()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot training loss
        axes[0].plot(self.train_losses)
        axes[0].set_title('Training Loss Over Time')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Plot predictions
        train_dates = self.data.index[self.lookback:self.lookback+len(y_train_actual)]
        test_dates = self.data.index[self.lookback+len(y_train_actual):]
        
        axes[1].plot(train_dates, y_train_actual, label='Actual (Train)', alpha=0.7)
        axes[1].plot(train_dates, train_pred, label='Predicted (Train)', alpha=0.7)
        axes[1].plot(test_dates, y_test_actual, label='Actual (Test)', alpha=0.7)
        axes[1].plot(test_dates, test_pred, label='Predicted (Test)', alpha=0.7)
        
        axes[1].set_title(f'{self.symbol} Stock Price Prediction')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Price ($)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = f'{self.symbol}_lstm_model.pth'
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

def main():
    try:
        # Initialize predictor
        predictor = StockPredictor(symbol='AAPL', period='2y', lookback=60)
        
        # Fetch and preprocess data
        data = predictor.fetch_data()
        if len(data) < 100:  # Check if enough data
            print("Not enough data fetched!")
            return None, None
            
        predictor.preprocess_data()
        
        # Build and train model
        predictor.build_model(hidden_size=50, num_layers=2)
        predictor.train_model(epochs=100, batch_size=32, lr=0.001)
        
        # Evaluate and visualize results
        metrics = predictor.evaluate_model()
        predictor.plot_results()
        
        # Save model
        predictor.save_model()
        
        return predictor, metrics
    except Exception as e:
        print(f"Error: {e}")
        return None, None
if __name__ == "__main__":
    print("Starting main function...")
    try:
        predictor, metrics = main()
        print("Main function completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()  