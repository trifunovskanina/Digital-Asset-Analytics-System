CREATE TABLE ohlcv (
    id SERIAL NOT NULL,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,

    CHECK (high >= low),
    CHECK (volume >= 0),

    PRIMARY KEY (id)
);

CREATE TABLE news (
    id SERIAL NOT NULL,
    date TEXT NOT NULL,
    content TEXT NOT NULL,
    link TEXT NOT NULL,

    PRIMARY KEY (id)
);