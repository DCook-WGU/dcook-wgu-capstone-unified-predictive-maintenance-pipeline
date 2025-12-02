# Creating Kafka Consumer Postgres User Role

## Create the user
```
    CREATE USER kafka_ingest WITH PASSWORD 'F9tX3qL8vW2pR7mC';
```
## Grant access to the database
```
    GRANT CONNECT ON DATABASE dcook_capstone_postgres_db TO kafka_ingest;
```

## Grant schema usage
```
    GRANT USAGE ON SCHEMA ingest TO kafka_ingest;
```

## Auto-apply permissions for new tables
```
    GRANT INSERT ON ALL TABLES IN SCHEMA ingest TO kafka_ingest;
```

## Auto-apply permissions for new tables
```
    ALTER DEFAULT PRIVILEGES IN SCHEMA ingest
    GRANT INSERT ON TABLES TO kafka_ingest;
```

