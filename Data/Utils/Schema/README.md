# Database Schema Extractor

This utility extracts the complete schema from the project's SQLite database and creates well-formatted documentation.

## Features

- Extracts table definitions, columns, data types, constraints, and indices
- Generates sample data for each table
- Outputs schema in multiple formats:
  - Markdown documentation with tables
  - JSON format for programmatic access
- Includes table row counts and relationships

## Usage

Simply run the script:

```bash
python db_schema_extractor.py
```

## Output Files

The script generates the following files in the `Data/Utils/Schema` directory:

- `db_schema.md` - Comprehensive Markdown documentation of the database schema
- `db_schema.json` - JSON representation of the schema for programmatic use

## Schema Information Extracted

For each table, the utility extracts:

- Column definitions (name, type, constraints)
- Primary keys
- Foreign key relationships
- Indices and unique constraints
- Sample data (first 5 rows)
- Row counts
- Original SQL creation statements

## Example

The Markdown output will look like:

```markdown
# Database Schema Documentation

Generated on: 2025-05-10 14:30:45

## Tables Overview

| Table Name | Row Count | Description |
|------------|-----------|-------------|
| stock_data | 125000    |             |
| patterns   | 3450      |             |
| clusters   | 42        |             |
...

## stock_data

### Columns

| Column Name | Data Type | Not Null | Default Value | Primary Key |
|-------------|-----------|----------|---------------|-------------|
| StockEntryID | INTEGER | ✓ |  | ✓ |
| StockID | INTEGER | ✓ |  | ✓ |
...
```
