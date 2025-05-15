# Database Structure and Relationships in Stock_AI_Predictor

## Overview

The Stock_AI_Predictor project utilizes a sophisticated database architecture implemented with SQLite, located in the `Data/Storage/data.db` file. This database serves as the central repository for all data components required for the AI-driven stock prediction system:

1. **Historical Stock Price Data**: Time-series data at multiple timeframes (15-min, 1-hour, daily)
2. **Technical Pattern Analysis**: Detection and classification of price patterns and clusters
3. **Sentiment Analysis**: Data from news articles and social media (Twitter)
4. **Prediction Generation**: System-generated investment recommendations and confidence metrics
5. **User Management**: Authentication, preferences, and notification delivery

The database is accessed by multiple system components including the prediction engine, backtesting framework, user interfaces, and data collection utilities. Both local SQLite and cloud-based SQLiteCloud configurations are supported for flexibility in deployment scenarios.

## Database Tables and Relationships

The database architecture consists of several interconnected tables organized into functional domains that together form a comprehensive system for stock market prediction and analysis.

### Core Tables

1. **`users`**
   - **Purpose**: Authentication and personalization layer
   - **Primary Key**: `UserID`
   - **Key Fields**: `Username`, `Password` (bcrypt hashed), `Email`, `Preferences` (JSON)
   - **Relationships**: One-to-many with `notifications`
   - **Usage**: User login, email notification preferences, personalized prediction delivery

2. **`stock_data`**
   - **Purpose**: Stores historical OHLC price data at multiple timeframes
   - **Composite Primary Key**: `(StockEntryID, StockID, TimeFrame)`
   - **Key Fields**: `Timestamp`, `OpenPrice`, `ClosePrice`, `HighPrice`, `LowPrice`, `Volume`
   - **Indexes**: Optimized for time-range queries through stock ID and timestamp
   - **Time Granularity**: Multiple timeframes (15-min, hourly, daily) stored with `TimeFrame` field
   - **Relationships**: Referenced by `patterns` and `clusters` tables
   - **Usage**: Primary data source for pattern recognition, technical analysis, and backtesting

3. **`patterns`**
   - **Purpose**: Stores price patterns identified in historical data for prediction
   - **Composite Primary Key**: `(PatternID, StockID)`
   - **Foreign Keys**: `ClusterID` references `clusters(ClusterID)`, `StockID` references stocks
   - **Key Fields**: `PricePoints` (normalized series), `TimeSpan`, `MarketCondition`, `Outcome`, `Label`, `MaxGain`, `MaxDrawdown`
   - **Pattern Types**: Classified as "Buy", "Sell", or "Neutral" in the `Label` field
   - **Performance Metrics**: Tracked through `MaxGain`, `MaxDrawdown`, and `Outcome` fields
   - **Storage Format**: Price points stored as comma-separated normalized values
   - **Usage**: Pattern matching for prediction generation and backtesting

4. **`clusters`**
   - **Purpose**: Groups similar patterns for more reliable prediction models
   - **Composite Primary Key**: `(ClusterID, StockID)`
   - **Key Fields**: `AVGPricePoints`, `MarketCondition`, `ProbabilityScore`, `Pattern_Count`, `MaxGain`, `MaxDrawdown`, `Label`
   - **Statistical Properties**: `ProbabilityScore` indicates historical success rate
   - **Risk Metrics**: `RewardRiskRatio` and `ProfitFactor` (calculated fields)
   - **Relationships**: One-to-many with `patterns` table
   - **Usage**: Higher-level pattern grouping for more statistically significant predictions

5. **`stock_sentiment`**
   - **Purpose**: Aggregates sentiment data by stock and date
   - **Primary Key**: Composite of `(stock_id, date)`
   - **Key Fields**: `news_sentiment_score`, `twitter_sentiment_score`, `combined_sentiment_score`, `sentiment_label`
   - **Count Metrics**: `article_count`, `tweet_count` for volume indicators
   - **Relationships**: Referenced by `predictions` table
   - **Usage**: Provides sentiment context to enhance pattern-based predictions

6. **`live_articles`**
   - **Purpose**: Stores news articles with sentiment analysis
   - **Primary Key**: `ID`
   - **Key Fields**: `Title`, `Summary`, `Source_Name`, `Date`, `Overall_Sentiment_Score`, `Overall_Sentiment_Label`
   - **Stock Relevance**: `Ticker_Sentiment` stores JSON with stock-specific sentiment scores
   - **Content Storage**: Full text in `Summary` with metadata in other fields
   - **Relationships**: Data aggregated into `stock_sentiment` table
   - **Usage**: Real-time and historical news sentiment analysis

7. **`tweets`**
   - **Purpose**: Stores tweets with engagement metrics and sentiment analysis
   - **Primary Key**: `ID` 
   - **Unique Constraint**: `tweet_id` field ensures no duplicate tweets
   - **Key Fields**: `tweet_text`, `created_at`, engagement metrics (`like_count`, etc.), `sentiment_score`, `sentiment_magnitude`, `weighted_sentiment`
   - **Author Metrics**: `author_followers`, `author_verified` for influence weighting
   - **Relationships**: Referenced by `stock_sentiment` through `ticker_id`
   - **Usage**: Social media sentiment analysis for prediction enhancement

8. **`predictions`**
   - **Purpose**: Stores system-generated trading predictions and recommendations
   - **Primary Key**: `PredictionID`
   - **Foreign Keys**: `StockID`, `PatternID`, `NewsID`, `TweetID`
   - **Key Fields**: `PredictionDate`, `PredictedOutcome` (JSON), `ConfidenceLevel`
   - **JSON Structure**: `PredictedOutcome` contains detailed prediction data including price targets, confidence intervals, sentiment factors, and specific pattern details
   - **Relationships**: One-to-many with `notifications` table
   - **Usage**: Core output of the prediction engine, used for notifications and performance tracking

9. **`notifications`**
   - **Purpose**: Manages user alerts for predictions and system events
   - **Primary Key**: `NotificationID`
   - **Foreign Keys**: `UserID`, `PredictionID`
   - **Key Fields**: `SentTime`, `NotificationType`, `Status`
   - **Notification Types**: Email alerts, in-app notifications, SMS (as configured)
   - **Status Tracking**: Delivery status monitoring through `Status` field
   - **Relationships**: Many-to-one with both `users` and `predictions`
   - **Usage**: Delivers actionable insights to users based on their preferences

## Detailed Table Relationships and Data Flow

### Stock Data to Pattern Analysis Pipeline

1. **Raw Data Collection → Storage → Analysis**
   - Raw stock price data is collected from external sources (Yahoo Finance, MT5)
   - Data is stored in the `stock_data` table with multiple timeframes
   - The `pip_pattern_miner.py` module processes this data to identify patterns

2. **Pattern Detection → Clustering → Probability Scoring**
   - Individual price patterns are identified and stored in the `patterns` table
   - Similar patterns are grouped into clusters in the `clusters` table
   - Each cluster receives a probability score based on historical performance
   - Risk metrics (MaxGain, MaxDrawdown, RewardRiskRatio) are calculated for each pattern and cluster

3. **Pattern Feature Extraction**
   - The `PricePoints` field in both tables stores normalized price movement as comma-separated values
   - `MarketCondition` classifies the overall trend (Bullish, Bearish, Neutral)
   - Performance metrics track historical success rates for prediction confidence

### Sentiment Analysis Pipeline

1. **Data Collection → Sentiment Extraction → Storage**
   - News articles are collected via AlphaVantage API (`alphavantage_api.py`)
   - Tweets are collected via Twitter API (`TwitterAPI_Sentiment.py`)
   - Raw data is stored in `live_articles` and `tweets` tables with sentiment scores

2. **Sentiment Aggregation → Stock Impact Analysis**
   - Sentiment data is aggregated by date and stock in the `stock_sentiment` table
   - The system calculates:
     - Combined sentiment scores weighted by source reliability
     - Article and tweet counts as volume indicators
     - Bullish/bearish ratios for directional strength

3. **Time Series Correlation**
   - Sentiment scores are analyzed for leading/lagging indicators
   - Correlation with price movements helps determine predictive power
   - Weighted sentiment features enhance the prediction model

### Prediction Generation Framework

1. **Pattern Recognition + Sentiment Analysis → Combined Model**
   - Current price action is matched against historical patterns in `clusters`
   - Sentiment data from `stock_sentiment` provides market context
   - Reinforcement learning model (`RL/Models`) may enhance predictions

2. **Confidence Calculation → Prediction Storage**
   - Prediction confidence is calculated based on:
     - Pattern probability score from historical performance
     - Sentiment strength and directional alignment
     - Volume indicators from news and social media
   - Complete prediction data is stored as JSON in `predictions.PredictedOutcome`

3. **Notification Workflow**
   - High-confidence predictions trigger notifications based on user preferences
   - `notifications` table tracks delivery status and user interaction
   - Email notifications use the `Interface/send_email.py` module

### User Interaction System

1. **Authentication → Personalization → Notification**
   - Users authenticate through the `users` table credentials
   - User preferences determine notification thresholds and delivery methods
   - Personalized predictions are delivered based on stock interests

2. **Feedback Loop**
   - User interactions with predictions are tracked
   - Performance analysis compares predictions with actual outcomes
   - System adjusts weightings based on successful prediction patterns

## Database Access Patterns and Code Integration

The database is accessed by various components throughout the system. Here's a detailed breakdown of how different modules interact with the database:

### Core Prediction Engine

- **`Core/engine_v2.py`**: 
  - Main prediction engine that integrates all data sources
  - Accesses `patterns`, `clusters`, and `stock_data` for technical analysis
  - Retrieves sentiment data for enhanced predictions
  - Sample usage:
    ```python
    def generate_prediction(self, stock_id, current_date):
        # Load relevant clusters with probability scores
        clusters = self.db.get_clusters_by_stock_id(stock_id)
        
        # Get recent sentiment data
        sentiment = self.fetch_recent_sentiment(stock_id, current_date)
        
        # Pattern matching against current price action
        matching_cluster = self.find_matching_pattern(stock_id, current_date)
        
        # Generate combined prediction
        prediction = self.combine_technical_and_sentiment(matching_cluster, sentiment)
        
        # Store prediction in database
        prediction_id = self.db.store_prediction_data(stock_id, prediction)
        
        return prediction, prediction_id
    ```

- **`Core/sentiment_enhanced_prediction.py`**:
  - Integrates sentiment analysis with pattern recognition
  - Performs time-series analysis of sentiment vs. price movements
  - Accesses `stock_sentiment`, `live_articles`, and `tweets` tables

### Data Collection and Processing

- **`Sentiment/API/TwitterAPI_Sentiment.py`**:
  - Retrieves tweets for specific stocks and calculates sentiment scores
  - Uses natural language processing for sentiment analysis
  - Stores results in the `tweets` table and updates `stock_sentiment`
  - Implementation details:
    ```python
    def collect_and_store_tweets(self, ticker_id, days_back=3):
        # Get appropriate search queries
        search_queries = self.get_search_queries_by_ticker(ticker_id)
        
        # Collect tweets for each query
        all_tweets = []
        for query in search_queries:
            tweets = self.search_tweets(query, days=days_back)
            all_tweets.extend(tweets)
            
        # Analyze sentiment and store in database
        for tweet in all_tweets:
            sentiment = self.analyze_sentiment(tweet['tweet_text'])
            self.db.store_tweets(
                ticker_id, 
                tweet['id'],
                tweet['text'],
                # Additional fields...
                sentiment['label'],
                sentiment['score'],
                sentiment['magnitude'],
                sentiment['weighted_score']
            )
    ```

- **`Sentiment/API/alphavantage_api.py`**:
  - Collects news articles related to specific stocks
  - Analyzes sentiment and relevance of each article
  - Stores data in the `live_articles` table

- **`Pattern/pip_pattern_miner.py`**:
  - Analyzes stock price data to identify significant patterns
  - Implements clustering algorithms to group similar patterns
  - Updates the `patterns` and `clusters` tables
  - Key functions:
    ```python
    def store_patterns_and_clusters(self, stock_id):
        # Store individual patterns
        self.db.store_pattern_data(stock_id, self)
        
        # Store cluster centers
        self.db.store_cluster_data(stock_id, self)
        
        # Bind patterns to clusters
        self.db.bind_pattern_cluster(stock_id, self)
        
        # Update probability scores
        self.db.update_all_cluster_probability_score(stock_id, self)
    ```

### User Interface and Notification

- **`Interface/gui.py`**:
  - Provides graphical interface for viewing predictions and historical data
  - Accesses database for user authentication and data visualization
  - Displays predicted vs. actual outcomes for performance tracking

- **`Interface/send_email.py`**:
  - Delivers prediction notifications to users via email
  - Tracks notification status in the `notifications` table
  - Implementation:
    ```python
    def send_prediction_notification(self, user_id, prediction_id):
        # Get user email
        user = self.db.connection.execute(
            "SELECT Email FROM users WHERE UserID = ?", 
            (user_id,)
        ).fetchone()
        
        # Get prediction details
        prediction = self.db.connection.execute(
            "SELECT PredictedOutcome FROM predictions WHERE PredictionID = ?",
            (prediction_id,)
        ).fetchone()
        
        # Send email and update notification status
        if self.send_email(user['Email'], self.format_prediction(prediction['PredictedOutcome'])):
            self.db.store_notification_data(
                user_id, prediction_id, datetime.now(), "Email", "Sent"
            )
    ```

### Backtesting Framework

- **`Experements/Backtesting/*.py`**:
  - Various scripts that validate prediction models against historical data
  - Access `stock_data`, `patterns`, and `clusters` for simulation
  - Track performance metrics in dedicated backtesting tables

- **`Experements/Backtesting/database_schema.py`**:
  - Defines additional tables specifically for backtesting:
    - `performance_metrics`: Stores results of backtesting runs
    - `experiment_configs`: Tracks different parameter configurations
    - `backtest_trades`: Records individual trades from simulation runs

## Performance Optimization and Technical Considerations

### Database Structure Optimizations

1. **Indexing Strategy**
   - The `stock_data` table uses a composite primary key `(StockEntryID, StockID, TimeFrame)` to enable efficient filtering
   - Additional indexes on `Timestamp` field accelerate time-range queries
   - The `patterns` and `clusters` tables use compound keys for efficient pattern matching

2. **Data Storage Formats**
   - Price patterns are stored as comma-separated normalized values in text fields
   - Complex data (like article sentiment details) is stored as JSON for flexibility
   - Binary data (like user password hashes) uses specialized SQLite data types

3. **Query Optimization**
   - Common queries are optimized with appropriate indexes
   - Filtering is performed at the database level when possible
   - Complex calculations are performed in application code rather than SQL

### Transaction Management

1. **Write Operations**
   - Batch inserts are used for high-volume data like `stock_data` and `tweets`
   - Transactions ensure data consistency for related table updates
   - Example implementation:
     ```python
     def store_cluster_data(self, stock_ID, pip_pattern_miner):
         try:
             # Begin transaction
             self.connection.execute("BEGIN TRANSACTION")
             
             # Insert multiple clusters
             for i, cluster in enumerate(pip_pattern_miner._cluster_centers):
                 # Process and insert cluster data
                 # ...
             
             # Commit all changes at once
             self.connection.commit()
         except Exception as e:
             self.connection.rollback()
             print(f"Error storing cluster data: {e}")
     ```

2. **Read Operations**
   - Prepared statements prevent SQL injection and improve performance
   - Result caching for frequently accessed data
   - Connection pooling for multi-threaded operations

### Backup and Recovery

1. **Database Backup Strategy**
   - The SQLite WAL (Write-Ahead Logging) mode is used for crash recovery
   - Regular backup files are created using SQLite's backup API
   - Journal files (`data.db-shm`, `data.db-wal`) provide transaction safety

2. **Migration and Schema Updates**
   - Schema changes follow a version-controlled migration pattern
   - The `database_schema.py` module can update existing tables without data loss

### Local vs. Cloud Database

1. **Local SQLite Configuration**
   - Used for development and standalone deployments
   - Implemented in `Data/Database/db.py`
   - Direct file access with thread safety considerations

2. **SQLiteCloud Configuration**
   - Used for production and multi-user scenarios
   - Implemented in `Data/Database/db_cloud.py`
   - Network-based access with additional authentication

## Data Modeling and Analytics Capabilities

### Time Series Analysis

1. **Stock Data Time Series**
   - The `stock_data` table preserves time-order for proper analysis
   - Multiple timeframes enable multi-timeframe analysis
   - Data alignment ensures consistent pattern detection

2. **Sentiment Time Correlation**
   - The `stock_sentiment` table aggregates sentiment by date
   - Time-lagged correlation analysis between sentiment and price movement
   - Lead/lag indicators help determine predictive power

### Machine Learning Integration

1. **Feature Engineering**
   - Pattern features extracted from `patterns` and `clusters` tables
   - Sentiment features from `live_articles` and `tweets`
   - Combined features feed into prediction models

2. **Model Training Data**
   - Historical pattern outcomes provide supervised learning labels
   - Backtesting tables store model performance metrics
   - Reinforcement learning environment uses database for state representation

### Analytical Queries

1. **Performance Analysis**
   - Success rate tracking for different pattern types
   - Sentiment correlation with market movements
   - Prediction accuracy over time

2. **Insight Generation**
   - Profitable pattern identification
   - Sentiment trend analysis
   - Market condition classification

## Database Schema SQL Definitions

Below are the SQL creation statements for the core tables in the database. These definitions show the structure and relationships between tables.

### User Management Tables

```sql
CREATE TABLE users (
    UserID INTEGER PRIMARY KEY,
    Username VARCHAR(50) NOT NULL,
    Password VARCHAR(255) NOT NULL,
    Email VARCHAR(100) NOT NULL,
    Preferences TEXT
)

CREATE TABLE notifications (
    NotificationID INTEGER PRIMARY KEY AUTOINCREMENT,
    UserID INTEGER,
    PredictionID INTEGER,
    SentTime DATETIME DEFAULT CURRENT_TIMESTAMP,
    NotificationType VARCHAR(50),
    Status VARCHAR(20)
)
```

### Stock Data and Technical Analysis Tables

```sql
CREATE TABLE stock_data (
    StockEntryID INTEGER,
    StockID INTEGER,
    StockSymbol TEXT,
    Timestamp TEXT,
    TimeFrame INTEGER,
    OpenPrice REAL,
    ClosePrice REAL,
    HighPrice REAL,
    LowPrice REAL,
    Volume REAL,
    primary key (StockEntryID, StockID, TimeFrame)
)

CREATE TABLE patterns (
    PatternID INTEGER,
    StockID INTEGER,
    ClusterID INTEGER,
    PricePoints TEXT,
    TimeSpan VARCHAR(50),
    MarketCondition VARCHAR(20),
    Outcome REAL,
    Label VARCHAR(50),
    MaxGain REAL,
    MaxDrawdown REAL,
    primary key (PatternID, StockID)
)

CREATE TABLE clusters (
    ClusterID INTEGER,
    StockID INTEGER,
    AVGPricePoints TEXT,
    MarketCondition VARCHAR(20),
    Outcome REAL,
    Label VARCHAR(50),
    ProbabilityScore REAL,
    Pattern_Count INTEGER,
    MaxGain REAL,
    MaxDrawdown REAL,
    primary key (ClusterID, StockID)
)
```

### Sentiment Analysis Tables

```sql
CREATE TABLE stock_sentiment (
    stock_id INTEGER,
    date TEXT,
    news_sentiment_score REAL,
    twitter_sentiment_score REAL,
    combined_sentiment_score REAL,
    sentiment_label TEXT,
    article_count INTEGER,
    tweet_count INTEGER,
    primary key (stock_id, date)
)

CREATE TABLE live_articles (
    ID INTEGER primary key autoincrement,
    Date TEXT,
    Authors TEXT,
    Source_Domain TEXT,
    Source_Name TEXT,
    Title TEXT,
    Summary TEXT,
    Url TEXT,
    Topics TEXT,
    Ticker_Sentiment TEXT,
    Overall_Sentiment_Label TEXT,
    Overall_Sentiment_Score REAL,
    Event_Type TEXT,
    Sentiment_Label TEXT,
    Sentiment_Score REAL,
    Fetch_Timestamp TEXT
)

CREATE TABLE tweets (
    ID INTEGER primary key autoincrement,
    ticker_id INTEGER,
    tweet_id TEXT constraint tweets_pk unique,
    tweet_text TEXT,
    created_at TEXT,
    retweet_count INTEGER,
    reply_count INTEGER,
    like_count INTEGER,
    quote_count INTEGER,
    bookmark_count INTEGER,
    lang TEXT,
    is_reply BOOLEAN,
    is_quote BOOLEAN,
    is_retweet BOOLEAN,
    url TEXT,
    search_term TEXT,
    author_username TEXT,
    author_name TEXT,
    author_verified BOOLEAN,
    author_blue_verified BOOLEAN,
    author_followers INTEGER,
    author_following INTEGER,
    sentiment_label TEXT,
    sentiment_score REAL,
    sentiment_magnitude REAL,
    weighted_sentiment REAL,
    collected_at TEXT
)
```

### Prediction and Evaluation Tables

```sql
CREATE TABLE predictions (
    PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
    StockID INTEGER,
    PatternID INTEGER,
    NewsID INTEGER,
    TweetID INTEGER,
    PredictionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    PredictedOutcome TEXT CHECK(json_valid(PredictedOutcome)),
    ConfidenceLevel FLOAT
)

CREATE TABLE performance_metrics (
    metric_id INTEGER PRIMARY KEY,
    stock_id INTEGER,
    timeframe_id INTEGER,
    config_id INTEGER,
    start_date TEXT,
    end_date TEXT,
    total_trades INTEGER,
    win_count INTEGER,
    loss_count INTEGER,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    max_consecutive_wins INTEGER,
    max_consecutive_losses INTEGER,
    profit_factor REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    recognition_technique TEXT,
    total_return_pct REAL,
    annualized_return_pct REAL,
    volatility REAL,
    calmar_ratio REAL,
    avg_trade_duration REAL
)
```

## Data Flow Diagram

```
                ┌─────────────────────────────────┐
                │        External Data Sources     │
                │  (Yahoo Finance, Twitter, News)  │
                └─────────────────┬───────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Collection Layer                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Stock Price    │    │  News Articles  │    │  Twitter Data   │  │
│  │  Collection     │    │  Collection     │    │  Collection     │  │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Database Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ stock_data  │  │ patterns &  │  │   live_    │  │   tweets    │  │
│  │   table     │  │  clusters   │  │  articles  │  │    table    │  │
│  └─────┬───────┘  └─────┬───────┘  └─────┬──────┘  └─────┬───────┘  │
└────────┼────────────────┼────────────────┼───────────────┼──────────┘
         │                │                │               │
         │                │                ▼               │
         │                │         ┌─────────────┐       │
         │                │         │    stock_   │       │
         │                │         │  sentiment  │       │
         │                │         └─────┬───────┘       │
         │                │               │               │
         ▼                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Analysis Layer                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Pattern Mining  │    │   Sentiment     │    │  ML/RL Model    │  │
│  │ (Technical)     │    │   Analysis      │    │  Training       │  │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘  │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Prediction Layer                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   Integrated Prediction Engine               │    │
│  │                                                              │    │
│  │  • Pattern Matching        • Confidence Calculation          │    │
│  │  • Sentiment Integration   • Risk Assessment                 │    │
│  │  • ML/RL Enhancement       • Position Sizing                 │    │
│  └────────────────────────────────┬──────────────────────────────┘  │
└─────────────────────────────────┬─┴────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Application Layer                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  User           │    │  Notification   │    │  Performance    │  │
│  │  Interface      │◄───┤  System         │    │  Tracking       │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Security and Data Privacy Considerations

1. **Authentication Security**
   - Passwords are hashed using the bcrypt algorithm with salting
   - Access control is implemented at both database and application levels
   - The Database class sanitizes SQL queries to prevent injection attacks

2. **Data Protection**
   - Sensitive financial prediction data is protected with access controls
   - User preferences and notification settings are stored securely
   - External API keys are stored in environment variables, not in the database

3. **Audit and Logging**
   - Key operations are logged for troubleshooting and security auditing
   - Notification table acts as an audit trail for prediction deliveries
   - Performance metrics track system accuracy over time

## Conclusion

The database architecture in the Stock_AI_Predictor project is designed with several key principles:

1. **Modular Design**: Tables are organized by functional domain with clear relationships
2. **Data Integration**: Various data sources (price, news, social media) are integrated seamlessly
3. **Performance Optimization**: Indexing and query optimization for efficient access patterns
4. **Extensibility**: Structure allows for adding new data sources and prediction techniques
5. **Feedback Loop**: Historical predictions are tracked to improve system accuracy over time

This comprehensive database design enables sophisticated stock market analysis by combining technical pattern recognition with sentiment analysis and machine learning, providing a solid foundation for generating reliable trading signals and investment recommendations.
