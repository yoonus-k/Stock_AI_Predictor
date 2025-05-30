#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Server for Stock AI Predictor

This script provides a FastAPI server for the Stock AI Predictor models.
It loads the models from Hugging Face Hub and provides endpoints for:
1. Parameter testing and optimization
2. RL model trading signals
3. Model metadata and information
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import FastAPI
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import model classes
from Deployment.parameter_tester_model import ParameterTesterModel
from Deployment.trading_rl_model import TradingRLModel

# Load environment variables
load_dotenv()

# Models repository IDs
PARAM_TESTER_REPO_ID = os.environ.get("PARAM_TESTER_REPO_ID", "username/stock-ai-parameter-tester")
RL_MODEL_REPO_ID = os.environ.get("RL_MODEL_REPO_ID", "username/stock-ai-rl-trader")

# API security
API_KEY = os.environ.get("API_KEY", "default_api_key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Create FastAPI app
app = FastAPI(
    title="Stock AI Predictor API",
    description="API for Stock AI Predictor models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
param_tester_model = None
rl_model = None

# Request and response models
class PriceData(BaseModel):
    timestamp: List[str] = Field(..., description="List of timestamps")
    open: List[float] = Field(..., description="List of opening prices")
    high: List[float] = Field(..., description="List of high prices")
    low: List[float] = Field(..., description="List of low prices")
    close: List[float] = Field(..., description="List of closing prices")
    volume: Optional[List[float]] = Field(None, description="List of volumes")

class SentimentData(BaseModel):
    timestamp: List[str] = Field(..., description="List of timestamps")
    sentiment_score: List[float] = Field(..., description="List of sentiment scores")
    source: Optional[str] = Field(None, description="Source of sentiment data")

class PortfolioState(BaseModel):
    balance: float = Field(..., description="Current balance")
    position: float = Field(..., description="Current position size")
    position_value: float = Field(..., description="Current position value")
    entry_price: Optional[float] = Field(None, description="Entry price")

class ParamTesterRequest(BaseModel):
    price_data: PriceData = Field(..., description="Price data")
    stock_id: int = Field(..., description="Stock ID")
    timeframe_id: int = Field(..., description="Timeframe ID")

class ParamTesterResponse(BaseModel):
    best_params: Dict[str, Any] = Field(..., description="Best parameters")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")

class TradingSignalRequest(BaseModel):
    price_data: PriceData = Field(..., description="Recent price data")
    sentiment_data: Optional[SentimentData] = Field(None, description="Recent sentiment data")
    portfolio_state: Optional[PortfolioState] = Field(None, description="Current portfolio state")

class TradingSignalResponse(BaseModel):
    action: str = Field(..., description="Trading action (BUY, SELL, HOLD)")
    position_size: float = Field(..., description="Recommended position size as fraction")
    confidence: float = Field(..., description="Confidence score")
    timestamp: str = Field(..., description="Timestamp of prediction")


# Authentication dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# Helper functions
def load_models():
    """Load models from Hugging Face Hub"""
    global param_tester_model, rl_model
    
    try:
        # Load parameter tester model
        param_tester_model = ParameterTesterModel.from_pretrained(PARAM_TESTER_REPO_ID)
        print(f"Parameter tester model loaded from {PARAM_TESTER_REPO_ID}")
    except Exception as e:
        print(f"Error loading parameter tester model: {e}")
        param_tester_model = ParameterTesterModel()  # Fallback to empty model
    
    try:
        # Load RL model
        rl_model = TradingRLModel.from_pretrained(RL_MODEL_REPO_ID)
        print(f"RL model loaded from {RL_MODEL_REPO_ID}")
    except Exception as e:
        print(f"Error loading RL model: {e}")
        rl_model = TradingRLModel()  # Fallback to empty model


def convert_to_dataframe(price_data: PriceData) -> pd.DataFrame:
    """Convert price data to pandas DataFrame"""
    data = {
        "timestamp": price_data.timestamp,
        "open": price_data.open,
        "high": price_data.high,
        "low": price_data.low,
        "close": price_data.close,
    }
    
    if price_data.volume:
        data["volume"] = price_data.volume
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    return df


# API routes
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {"message": "Stock AI Predictor API", "version": "1.0.0"}


@app.get("/health", response_class=JSONResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "models": {
            "parameter_tester": param_tester_model is not None,
            "rl_model": rl_model is not None
        }
    }


@app.post("/optimize-parameters", response_model=ParamTesterResponse, dependencies=[Depends(verify_api_key)])
async def optimize_parameters(request: ParamTesterRequest):
    """
    Optimize parameters for a given stock and timeframe
    """
    if param_tester_model is None:
        raise HTTPException(status_code=503, detail="Parameter tester model not available")
    
    try:
        # Convert price data to DataFrame
        df = convert_to_dataframe(request.price_data)
        
        # Train parameter tester model
        result = param_tester_model.train(df, request.stock_id, request.timeframe_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing parameters: {str(e)}")


@app.post("/trading-signal", response_model=TradingSignalResponse, dependencies=[Depends(verify_api_key)])
async def get_trading_signal(request: TradingSignalRequest):
    """
    Get trading signal based on price and sentiment data
    """
    if rl_model is None:
        raise HTTPException(status_code=503, detail="RL model not available")
    
    try:
        # Convert price data to DataFrame
        price_df = convert_to_dataframe(request.price_data)
        
        # Prepare state for RL model
        state = {}
        
        # Extract pattern features from price data (last N candles)
        # This is a simplified example, you would need to implement the actual feature extraction
        pattern_features = [
            price_df["close"].pct_change().iloc[-5:].mean(),
            price_df["close"].pct_change().iloc[-5:].std(),
            price_df["high"].max() / price_df["close"].iloc[-1] - 1,
            price_df["close"].iloc[-1] / price_df["low"].min() - 1,
            price_df["volume"].iloc[-5:].mean() if "volume" in price_df else 0.0
        ]
        state["pattern_features"] = pattern_features
        
        # Extract sentiment features if available
        if request.sentiment_data:
            sentiment_df = pd.DataFrame({
                "timestamp": request.sentiment_data.timestamp,
                "sentiment_score": request.sentiment_data.sentiment_score
            })
            sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
            sentiment_df = sentiment_df.set_index("timestamp")
            
            sentiment_features = [
                sentiment_df["sentiment_score"].mean(),
                sentiment_df["sentiment_score"].std(),
                sentiment_df["sentiment_score"].iloc[-1]
            ]
            state["sentiment_features"] = sentiment_features
        else:
            state["sentiment_features"] = [0.0, 0.0, 0.0]
        
        # Add portfolio state if available
        if request.portfolio_state:
            state["balance"] = request.portfolio_state.balance
            state["position"] = request.portfolio_state.position
            state["position_value"] = request.portfolio_state.position_value
        else:
            state["balance"] = 100000.0
            state["position"] = 0.0
            state["position_value"] = 0.0
        
        # Get prediction from RL model
        prediction = rl_model.predict(state)
        
        # Return response
        return {
            "action": prediction["action"],
            "position_size": prediction["position_size"],
            "confidence": prediction["confidence"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trading signal: {str(e)}")


@app.get("/model-info", response_class=JSONResponse)
async def get_model_info():
    """Get model information"""
    return {
        "parameter_tester": {
            "repo_id": PARAM_TESTER_REPO_ID,
            "config": param_tester_model.config if param_tester_model else {},
            "best_params": param_tester_model.best_params if param_tester_model else {}
        },
        "rl_model": {
            "repo_id": RL_MODEL_REPO_ID,
            "config": rl_model.config if rl_model else {}
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 7860))
    
    # Run server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
