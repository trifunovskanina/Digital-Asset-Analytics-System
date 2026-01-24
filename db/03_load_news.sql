COPY news(date, content, link)
FROM '/docker-entrypoint-initdb.d/data/news.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"');
