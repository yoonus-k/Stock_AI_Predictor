#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the connection between the trading client and Hugging Face API.
This script verifies that the trading client can successfully connect to and
interact with the Stock AI Predictor API hosted on Hugging Face Spaces.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import trading client
from Deployment.trading_client import StockAIClient

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_connection_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_connection_test")

def test_connection(api_base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
    """
    Test the connection to the Hugging Face API
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key for authentication
    """
    api_base_url = api_base_url or os.environ.get("API_BASE_URL")
    api_key = api_key or os.environ.get("API_KEY")
    
    if not api_base_url or not api_key:
        logger.error("API base URL and API key must be provided")
        print("Error: API base URL and API key must be provided")
        print("Make sure to set them in the .env file or provide them as arguments")
        return
    
    print(f"Testing connection to {api_base_url}...")
    
    # Test direct HTTP connection
    try:
        headers = {"X-API-Key": api_key}
        response = requests.get(f"{api_base_url}/api/health", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ Direct HTTP connection successful")
            print(f"Server response: {response.json()}")
        else:
            print(f"❌ Direct HTTP connection failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Direct HTTP connection failed: {e}")
    
    # Test through client
    try:
        client = StockAIClient(api_base_url=api_base_url, api_key=api_key)
        print("✅ Client initialization successful")
        
        # Test parameter optimization
        print("\nTesting parameter optimization...")
        try:
            result = client.get_parameter_optimization("AAPL", "1D", lookback_period=30)
            print("✅ Parameter optimization request successful")
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Parameter optimization request failed: {e}")
        
        # Test trading signal
        print("\nTesting trading signal...")
        try:
            result = client.get_trading_signal("AAPL", "1D", use_rl_model=True)
            print("✅ Trading signal request successful")
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"❌ Trading signal request failed: {e}")
            
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
    
    print("\nConnection test completed")

def main():
    """Main function for the connection test"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the connection to the Hugging Face API")
    parser.add_argument("--api-url", type=str, help="Base URL for the API")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    
    args = parser.parse_args()
    
    test_connection(api_base_url=args.api_url, api_key=args.api_key)

if __name__ == "__main__":
    main()
