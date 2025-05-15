import sqlite3
import logging
import json
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
DB_PATH = 'C:/Users/yoonus/Documents/GitHub/Stock_AI_Predictor/Data/Storage/data.db'

def connect_to_db():
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def get_pattern_prediction(conn, stock_id, pattern_id):
    """
    Get pattern-based prediction details
    Returns a dict with prediction outcome and confidence
    """
    try:
        cursor = conn.cursor()
        # Get historical performance of this pattern for this stock
        cursor.execute("""
            SELECT 
                p.pattern_id,
                AVG(CASE WHEN pm.actual_outcome = 'up' THEN 1 ELSE 0 END) as avg_success,
                COUNT(pm.metric_id) as total_occurrences
            FROM 
                patterns p
            JOIN 
                performance_metrics pm ON p.pattern_id = pm.pattern_id
            WHERE 
                p.pattern_id = ? AND pm.stock_id = ?
            GROUP BY 
                p.pattern_id
        """, (pattern_id, stock_id))
        
        result = cursor.fetchone()
        if not result:
            return {"outcome": "neutral", "confidence": 0.5}
        
        avg_success = result['avg_success']
        occurrences = result['total_occurrences']
        
        # Calculate confidence based on historical performance and occurrences
        base_confidence = abs(avg_success - 0.5) * 2  # Scale to 0-1
        
        # Adjust confidence based on number of occurrences
        occurrence_weight = min(occurrences / 10, 1.0)  # Cap at 10 occurrences for full weight
        pattern_confidence = base_confidence * occurrence_weight
        
        # Determine outcome
        if avg_success > 0.55:  # Slightly above random chance
            outcome = "up"
        elif avg_success < 0.45:  # Slightly below random chance
            outcome = "down"
        else:
            outcome = "neutral"
            
        return {
            "outcome": outcome,
            "confidence": pattern_confidence,
            "raw_probability": avg_success,
            "occurrences": occurrences
        }
    
    except sqlite3.Error as e:
        logger.error(f"Database error in pattern prediction: {e}")
        return {"outcome": "neutral", "confidence": 0.5}
    except Exception as e:
        logger.error(f"Unexpected error in pattern prediction: {e}")
        return {"outcome": "neutral", "confidence": 0.5}

def get_sentiment_prediction(conn, stock_id, days=7):
    """
    Get sentiment-based prediction using recent sentiment data
    Returns a dict with prediction outcome and confidence
    """
    try:
        cursor = conn.cursor()
        
        # Get recent sentiment data for this stock
        current_date = datetime.now().date()
        start_date = (current_date - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT 
                AVG(combined_sentiment_score) as avg_sentiment,
                COUNT(sentiment_id) as sentiment_count
            FROM 
                stock_sentiment
            WHERE 
                stock_id = ? AND date >= ?
        """, (stock_id, start_date))
        
        result = cursor.fetchone()
        if not result or result['sentiment_count'] == 0:
            return {"outcome": "neutral", "confidence": 0.5}
        
        avg_sentiment = result['avg_sentiment'] if result['avg_sentiment'] is not None else 0
        sentiment_count = result['sentiment_count']
        
        # Calculate confidence based on sentiment strength and data points
        sentiment_strength = min(abs(avg_sentiment), 1.0)  # Cap at 1.0
        
        # Adjust confidence based on number of data points
        data_weight = min(sentiment_count / 5, 1.0)  # Cap at 5 days for full weight
        sentiment_confidence = sentiment_strength * data_weight
        
        # Determine outcome
        if avg_sentiment > 0.1:  # Positive sentiment
            outcome = "up"
        elif avg_sentiment < -0.1:  # Negative sentiment
            outcome = "down"
        else:
            outcome = "neutral"
            
        return {
            "outcome": outcome,
            "confidence": sentiment_confidence,
            "raw_sentiment": avg_sentiment,
            "data_points": sentiment_count
        }
    
    except sqlite3.Error as e:
        logger.error(f"Database error in sentiment prediction: {e}")
        return {"outcome": "neutral", "confidence": 0.5}
    except Exception as e:
        logger.error(f"Unexpected error in sentiment prediction: {e}")
        return {"outcome": "neutral", "confidence": 0.5}

def generate_combined_prediction(conn, stock_id, pattern_id, timeframe_id, config_id=1, 
                                pattern_weight=0.7, sentiment_weight=0.3):
    """
    Generate a combined prediction using pattern analysis and sentiment data
    
    Args:
        conn: Database connection
        stock_id: ID of the stock
        pattern_id: ID of the pattern identified
        timeframe_id: ID of the timeframe (daily, weekly, etc.)
        config_id: ID of the experiment configuration
        pattern_weight: Weight given to pattern-based prediction (0-1)
        sentiment_weight: Weight given to sentiment-based prediction (0-1)
    
    Returns:
        prediction_id: ID of the inserted prediction
    """
    try:
        # Ensure weights sum to 1
        total_weight = pattern_weight + sentiment_weight
        if total_weight != 1.0:
            pattern_weight = pattern_weight / total_weight
            sentiment_weight = sentiment_weight / total_weight
        
        # Get predictions from both sources
        pattern_pred = get_pattern_prediction(conn, stock_id, pattern_id)
        sentiment_pred = get_sentiment_prediction(conn, stock_id)
        
        # Combine outcomes by converting to numeric and weighing
        outcome_map = {"up": 1, "neutral": 0, "down": -1}
        
        pattern_numeric = outcome_map[pattern_pred["outcome"]]
        sentiment_numeric = outcome_map[sentiment_pred["outcome"]]
        
        # Calculate weighted outcome
        weighted_outcome = (
            pattern_numeric * pattern_weight * pattern_pred["confidence"] +
            sentiment_numeric * sentiment_weight * sentiment_pred["confidence"]
        )
        
        # Convert back to categorical outcome
        if weighted_outcome > 0.2:
            final_outcome = "up"
        elif weighted_outcome < -0.2:
            final_outcome = "down"
        else:
            final_outcome = "neutral"
        
        # Calculate combined confidence
        combined_confidence = (
            pattern_pred["confidence"] * pattern_weight +
            sentiment_pred["confidence"] * sentiment_weight
        )
        
        # Store metrics for future analysis
        prediction_metrics = {
            "pattern": pattern_pred,
            "sentiment": sentiment_pred,
            "weights": {
                "pattern": pattern_weight,
                "sentiment": sentiment_weight
            }
        }
        
        # Insert the prediction into the database with sentiment info
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (
                stock_id, pattern_id, timeframe_id, config_id, 
                prediction_date, predicted_outcome, confidence_level,
                sentiment_data_id, prediction_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock_id, pattern_id, timeframe_id, config_id,
            datetime.now(), final_outcome, combined_confidence,
            get_latest_sentiment_id(conn, stock_id),
            json.dumps(prediction_metrics)
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        
        logger.info(f"Generated combined prediction for stock_id={stock_id}, pattern_id={pattern_id}: {final_outcome} with {combined_confidence:.2f} confidence")
        
        return prediction_id
        
    except sqlite3.Error as e:
        logger.error(f"Database error in combined prediction: {e}")
        conn.rollback()
        return None
    except Exception as e:
        logger.error(f"Unexpected error in combined prediction: {e}")
        conn.rollback()
        return None

def get_latest_sentiment_id(conn, stock_id):
    """Get the most recent sentiment ID for a stock"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sentiment_id FROM stock_sentiment
            WHERE stock_id = ?
            ORDER BY date DESC
            LIMIT 1
        """, (stock_id,))
        
        result = cursor.fetchone()
        return result['sentiment_id'] if result else None
    except Exception as e:
        logger.error(f"Error getting latest sentiment ID: {e}")
        return None

def update_predictions_schema(conn):
    """Update the predictions table to include sentiment data"""
    try:
        cursor = conn.cursor()
        
        # Check if the columns already exist
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [column['name'] for column in cursor.fetchall()]
        
        # Add sentiment_data_id column if it doesn't exist
        if 'sentiment_data_id' not in columns:
            cursor.execute("""
                ALTER TABLE predictions 
                ADD COLUMN sentiment_data_id INTEGER 
                REFERENCES stock_sentiment(sentiment_id)
            """)
            logger.info("Added sentiment_data_id column to predictions table")
        
        # Add prediction_metrics column if it doesn't exist
        if 'prediction_metrics' not in columns:
            cursor.execute("""
                ALTER TABLE predictions 
                ADD COLUMN prediction_metrics TEXT
            """)
            logger.info("Added prediction_metrics column to predictions table")
            
        conn.commit()
        logger.info("Predictions schema updated successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Error updating predictions schema: {e}")
        conn.rollback()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        conn.rollback()

def main():
    """Main execution function"""
    try:
        conn = connect_to_db()
        
        # Update schema first
        update_predictions_schema(conn)
        
        # Example: Generate a prediction for a specific stock and pattern
        # In a real application, you would scan for patterns and generate predictions as needed
        stock_id = 3  # AAPL
        pattern_id = 1  # Assuming pattern 1 exists
        timeframe_id = 1  # Daily
        
        prediction_id = generate_combined_prediction(
            conn, stock_id, pattern_id, timeframe_id,
            pattern_weight=0.7, sentiment_weight=0.3
        )
        
        if prediction_id:
            logger.info(f"Successfully generated prediction with ID: {prediction_id}")
        else:
            logger.error("Failed to generate prediction")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()
