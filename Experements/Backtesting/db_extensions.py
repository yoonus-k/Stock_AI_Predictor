"""
Database Extensions for Multi-Config Pattern Recognition
"""
# Add parent directory to path for imports
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.dirname(parent_dir))
from Data.Database.db import Database as BaseDatabase
import pandas as pd
import logging

logger = logging.getLogger("DB.MultiConfig")

class Database(BaseDatabase):
    """Extended Database class with multi-config support for pattern recognition."""
    
    def get_clusters_by_config(self, stock_id, timeframe_id, config_id=None):
        """
        Get clusters filtered by stock_id, timeframe_id, and optionally config_id.
        
        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            config_id: Configuration ID (optional)
            
        Returns:
            DataFrame with clusters
        """
        cursor = self.connection.cursor()
        params = []
        query = "SELECT * FROM clusters"
        conditions = []
        
        # Ensure parameters are properly converted to native Python types
        if config_id is not None:
            conditions.append("config_id = ?")
            params.append(int(config_id) if hasattr(config_id, '__int__') else config_id)
        
        if stock_id is not None:
            conditions.append("stock_id = ?")
            params.append(int(stock_id) if hasattr(stock_id, '__int__') else stock_id)
        
        if timeframe_id is not None:
            conditions.append("timeframe_id = ?")
            params.append(int(timeframe_id) if hasattr(timeframe_id, '__int__') else timeframe_id)
        
        # Debug logging
        #logger.debug(f"Query params: {params}, types: {[type(p) for p in params]}")
        
        if conditions:
            # Join conditions with AND
            query += " WHERE " + " AND ".join(conditions)
            # Get clusters for specific config
            cursor.execute(query, tuple(params) if params else None)
        else:
            # Get all clusters for this stock and timeframe
            query = """
                SELECT * FROM clusters
                WHERE stock_id = ? AND timeframe_id = ?
            """
            cursor.execute(query, (
                int(stock_id) if hasattr(stock_id, '__int__') else stock_id,
                int(timeframe_id) if hasattr(timeframe_id, '__int__') else timeframe_id
            ))
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            
            #logger.info(f"Found {len(df)} clusters for stock_id={stock_id}, timeframe_id={timeframe_id}, config_id={config_id}")
            return df
        else:
            #logger.warning(f"No clusters found for stock_id={stock_id}, timeframe_id={timeframe_id}, config_id={config_id}")
            return pd.DataFrame(columns=columns)
    
    def get_configs_for_stock(self, stock_id):
        """
        Get all configurations for a stock.
        
        Args:
            stock_id: Stock ID
            
        Returns:
            List of config_id values
        """
        cursor = self.connection.cursor()
        query = """
            SELECT DISTINCT config_id FROM experiment_configs
            WHERE stock_id = ?
        """
        cursor.execute(query, (stock_id,))
        config_ids = [row[0] for row in cursor.fetchall()]
        
        if not config_ids:
            logger.warning(f"No configurations found for stock_id={stock_id}")
        else:
            l#ogger.info(f"Found {len(config_ids)} configurations for stock_id={stock_id}")
            
        return config_ids
    def get_configs_by_stock_and_timeframe(self, stock_id, timeframe_id):
        """
        Get all configurations for a stock and timeframe.
        
        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            
        Returns:
            List of config_id values
        """
        cursor = self.connection.cursor()
        query = """
            SELECT *  FROM experiment_configs
            WHERE stock_id = ? AND timeframe_id = ?
        """
        cursor.execute(query, (stock_id, timeframe_id))
        
        # convert to dataframe
        rows = cursor.fetchall()
        # get the column names
        columns = [description[0] for description in cursor.description]
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            #logger.info(f"Found {len(df)} configurations for stock_id={stock_id}, timeframe_id={timeframe_id}")
            return df
        
    def get_config_by_id(self, config_id):
        """
        Get configuration details by config_id.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            DataFrame with configuration details
        """
        cursor = self.connection.cursor()
        query = """
            SELECT * FROM experiment_configs
            WHERE config_id = ?
        """
        cursor.execute(query, (config_id,))
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        if rows:
            df = pd.DataFrame(rows, columns=columns)
            
            return df
        else:
            logger.warning(f"No configuration found for config_id={config_id}")
            return pd.DataFrame(columns=columns)
        
        
if __name__ == "__main__":
    # Example usage
    db = Database()
    stock_id = 1  # Example stock ID
    timeframe_id = 5  # Example timeframe ID
    config_id = 63  # Example config ID
    
    clusters = db.get_config_by_id(config_id)
    print(clusters['hold_period'])