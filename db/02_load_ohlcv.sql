COPY ohlcv(date, open, high, low, close, volume, symbol)
FROM '/docker-entrypoint-initdb.d/data/ohlcv.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"');
