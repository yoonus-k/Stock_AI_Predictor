# Parameter Testing Optimization Guide

## Problem: Long Execution Time for Parameter Testing

Running parameter testing on 1-minute timeframe data can take days to complete on a local machine due to:
1. Large dataset volume with 1-minute candles
2. Combinatorial explosion of parameter sets
3. Limited CPU and memory resources

## Solution Options

### Option 1: Google Colab with SQLite Database (Recommended)

Use Google Colab's high-performance environment to run parameter tests and save results back to your database.

#### Benefits:
- Free GPU/TPU acceleration
- Up to 25GB RAM (vs typical local 8-16GB)
- Optimized for parallel execution
- No setup required beyond a Google account

#### Setup:
1. Upload the notebook and database to Google Drive
2. Run the provided `colab_parameter_testing.ipynb` notebook
3. Download results when complete

```bash
# Execute this locally to prepare your database
python -m Data.Database.setup_cloud_db --create-env
```

#### Running Tests in Colab:

You have multiple options to run tests in Colab:

1. **Test a specific stock and timeframe**:
   ```python
   # Run parameter testing for GOLD on 1-minute timeframe
   results, report = tester.run_parameter_testing_for_stock_timeframe_parallel(
       "GOLD", "M1", 
       start_date="2023-01-01",  
       end_date="2025-01-01"
   )
   print(report)
   ```

2. **Test all timeframes for a specific stock**:
   ```python
   # Test all timeframes for GOLD
   tester.run_all_tests(
       stock_identifier="GOLD",
       test_all_params=True
   )
   ```

3. **Test all stocks and all timeframes**:
   ```python
   # Test everything (this may take a very long time)
   tester.run_all_tests(
       stock_identifier=None,  # None = all stocks
       test_all_params=True
   )
   ```

### Option 2: SQLite Cloud Database

Use a cloud-hosted SQLite database that can be accessed from multiple environments.

#### Benefits:
- Access your database from anywhere (local machine, Colab, cloud)
- Built-in backup and disaster recovery
- Concurrent access from multiple machines
- Free tier available for smaller databases

#### Setup:
1. Create an account at [SQLiteCloud](https://sqlitecloud.io/)
2. Get your connection string
3. Run setup script to sync your local database to the cloud:

```bash
python -m Data.Database.setup_cloud_db --sync
```

### Option 3: Divide and Conquer (Multiple Machines)

Split parameter testing across multiple machines or environments.

#### Benefits:
- Utilize all available hardware
- No cloud dependencies
- Good for team collaboration

#### Setup:
1. Divide parameter ranges into chunks
2. Run each chunk on a different machine
3. Merge results back into main database

```python
# Example: Machine 1 - first half of parameters
PARAM_RANGES_1 = {
    'n_pips': [3, 4, 5],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}

# Example: Machine 2 - second half of parameters
PARAM_RANGES_2 = {
    'n_pips': [6, 7, 8],
    'lookback': [12, 24, 36, 48],
    'hold_period': [3, 6, 12, 24],
}
```

### Option 4: Configure Multithreaded Execution

Optimize your existing code for better performance on your local machine.

#### Benefits:
- No external dependencies
- Works with existing setup
- Incremental improvement

#### Setup:
1. Use the unified parameter tester script with optimal thread configuration:

```bash
# Run tests for GOLD on M1 timeframe with 6 threads
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --timeframe M1 --test-all --threads 6

# Run tests for all timeframes of GOLD
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --test-all

# Run tests for all stocks and all timeframes
python -m Experements.ParamTesting.run_parameter_tests --test-all
```

## Detailed Instructions for Google Colab Setup

1. **Prepare your local database**:
   ```bash
   # Navigate to your project directory
   cd c:\Users\yoonus\Documents\GitHub\Stock_AI_Predictor
   
   # Optimize database for transfer
   sqlite3 Data/Storage/data.db "VACUUM;"
   ```

2. **Upload your database to Google Drive**:
   - Create a folder in Google Drive: `Stock_AI_Predictor/Data/Storage/`
   - Upload your `data.db` file to this folder

3. **Open the Colab notebook**:
   - Upload `NoteBook/colab_parameter_testing.ipynb` to Google Drive
   - Open it with Google Colab
   - Follow the steps in the notebook

4. **Run parameter testing**:
   - Choose one of the testing options in the notebook
   - For 1-minute timeframe data, you can specify date ranges to limit processing time
   - Results will be automatically saved to your database

5. **Save and download results**:
   - The notebook includes cells to save results back to Google Drive
   - You can also download the updated database directly to your local machine

## Command-Line Usage for All Environments

A unified command-line script is provided that works with both local and Colab environments:

```bash
# Basic usage
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --timeframe M1 --test-all

# Use Colab implementation if available
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --timeframe M1 --test-all --use-colab

# Compare hold period strategies
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --compare

# Run a quick test with default parameters
python -m Experements.ParamTesting.run_parameter_tests --stock GOLD --timeframe M1 --quick-test
```

## Best Practices for Database Management

1. **Regular backups**: Always back up your database before and after testing:
   ```bash
   cp Data/Storage/data.db Data/Storage/data_backup_$(date +%Y%m%d).db
   ```

2. **Version control**: Keep track of database changes with descriptive filenames:
   ```
   data_pre_M1_test.db
   data_post_M1_test.db
   ```

3. **Database maintenance**: Regularly optimize your database:
   ```bash
   sqlite3 Data/Storage/data.db "VACUUM; ANALYZE;"
   ```

4. **Incremental testing**: Test one timeframe at a time and save results before moving to the next.

5. **Progress monitoring**: Set up periodic checkpoints to track testing progress.

## Conclusion

By leveraging Google Colab's computational resources and implementing the provided optimizations, you can significantly reduce the execution time for parameter testing, especially for 1-minute timeframe data, while ensuring reliable storage of your results in the database.
