#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Client for Stock AI Predictor

This script provides a client for connecting to the Stock AI Predictor API
hosted on Hugging Face Spaces. It can be used to get trading signals and
execute trades on various platforms like MT5 or Binance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
import requests
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_client")

# API Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://huggingface.co/spaces/username/stock-ai-predictor-api")
API_KEY = os.environ.get("API_KEY")

class StockAIClient:
    """Client for the Stock AI Predictor API"""
    
    def __init__(self, api_base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Stock AI client
        
        Args:
            api_base_url: Base URL for the API
            api_key: API key for authentication
        """
        self.api_base_url = api_base_url or API_BASE_URL
        self.api_key = api_key or API_KEY
        self.headers = {"X-API-Key": self.api_key}
        
        # Validate configuration
        if not self.api_base_url or not self.api_key:
            raise ValueError("API base URL and API key must be provided")
            
        logger.info(f"Initialized Stock AI client with API base URL: {self.api_base_url}")
    
    def get_parameter_optimization(self, symbol: str, timeframe: str, 
                                  lookback_period: int = 30) -> Dict[str, Any]:
        """
        Get parameter optimization results
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1D', '4H', '1H')
            lookback_period: Number of days to look back
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        endpoint = f"{self.api_base_url}/api/parameters/optimize"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback_period": lookback_period
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting parameter optimization: {e}")
            raise
    
    def get_trading_signal(self, symbol: str, timeframe: str, 
                          use_rl_model: bool = True) -> Dict[str, Any]:
        """
        Get trading signal for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'EURUSD')
            timeframe: Timeframe for analysis (e.g., '1D', '4H', '1H')
            use_rl_model: Whether to use the RL model for signal generation
            
        Returns:
            Dict[str, Any]: Trading signal data
        """
        endpoint = f"{self.api_base_url}/api/trading/signal"
        
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "use_rl_model": use_rl_model
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting trading signal: {e}")
            raise

# Trading platform connectors
class MT5Connector:
    """MetaTrader 5 connector for executing trades"""
    
    def __init__(self):
        """Initialize MT5 connector"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            if not self.mt5.initialize():
                raise ValueError(f"MT5 initialization failed: {self.mt5.last_error()}")
            logger.info("MT5 initialized successfully")
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            raise
    
    def execute_trade(self, symbol: str, order_type: str, volume: float, 
                     price: Optional[float] = None, sl: Optional[float] = None, 
                     tp: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trade on MT5
        
        Args:
            symbol: Trading symbol
            order_type: Order type ('BUY', 'SELL')
            volume: Trade volume in lots
            price: Optional price for limit orders
            sl: Optional stop loss price
            tp: Optional take profit price
            
        Returns:
            Dict[str, Any]: Order result
        """
        # Implementation for MT5 trade execution
        # This is a placeholder - you'll need to implement the actual MT5 trading logic
        logger.info(f"Executing {order_type} order for {symbol} with volume {volume}")
        # Actual implementation would use self.mt5.order_send()
        return {"success": True, "order_id": 12345}

class BinanceConnector:
    """Binance connector for executing trades"""
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize Binance connector
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        try:
            from binance.client import Client
            self.client = Client(api_key, api_secret)
            logger.info("Binance client initialized successfully")
        except ImportError:
            logger.error("Binance package not installed")
            raise
    
    def execute_trade(self, symbol: str, order_type: str, quantity: float, 
                     price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trade on Binance
        
        Args:
            symbol: Trading symbol
            order_type: Order type ('BUY', 'SELL')
            quantity: Trade quantity
            price: Optional price for limit orders
            
        Returns:
            Dict[str, Any]: Order result
        """
        # Implementation for Binance trade execution
        # This is a placeholder - you'll need to implement the actual Binance trading logic
        logger.info(f"Executing {order_type} order for {symbol} with quantity {quantity}")
        # Actual implementation would use self.client.create_order()
        return {"success": True, "orderId": 12345}

def main():
    """Main function for the trading client"""
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock AI Predictor Trading Client")
    parser.add_argument("--symbol", type=str, required=True, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="1H", help="Trading timeframe")
    parser.add_argument("--platform", type=str, choices=["mt5", "binance"], required=True, 
                      help="Trading platform to use")
    parser.add_argument("--volume", type=float, default=0.01, help="Trading volume/quantity")
    
    args = parser.parse_args()
    
    # Initialize AI client
    client = StockAIClient()
    
    # Get trading signal
    logger.info(f"Getting trading signal for {args.symbol} on {args.timeframe} timeframe")
    signal = client.get_trading_signal(args.symbol, args.timeframe)
    
    if not signal.get("success", False):
        logger.error(f"Failed to get trading signal: {signal.get('error')}")
        return
    
    # Extract trading decision
    action = signal.get("action")
    confidence = signal.get("confidence", 0.0)
    
    logger.info(f"Received trading signal: {action} with confidence {confidence}")
    
    # Execute trade based on platform
    if args.platform == "mt5":
        connector = MT5Connector()
        result = connector.execute_trade(args.symbol, action, args.volume)
    elif args.platform == "binance":
        # You would need to store Binance API credentials in env vars
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        
        connector = BinanceConnector(api_key, api_secret)
        result = connector.execute_trade(args.symbol, action, args.volume)
    
    logger.info(f"Trade execution result: {result}")

if __name__ == "__main__":
    main()
