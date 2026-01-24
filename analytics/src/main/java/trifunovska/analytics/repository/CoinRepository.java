package trifunovska.analytics.repository;

import trifunovska.analytics.model.Coin;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

@Repository
public class CoinRepository {

    private final JdbcTemplate jdbcTemplate;

    public CoinRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    private static class CoinRowMapper implements RowMapper<Coin> {
        @Override
        public Coin mapRow(ResultSet rs, int rowNum) throws SQLException {
            Coin row = new Coin();
            row.setDate(rs.getString("date"));
            row.setOpen(rs.getDouble("open"));
            row.setHigh(rs.getDouble("high"));
            row.setLow(rs.getDouble("low"));
            row.setClose(rs.getDouble("close"));
            row.setVolume(rs.getDouble("volume"));
            row.setSymbol(rs.getString("symbol"));
            return row;
        }
    }

    public List<Coin> findAll() {
        String sql = "SELECT \"date\", open, high, low, close, volume, symbol FROM ohlcv ORDER BY \"date\"";
        return jdbcTemplate.query(sql, new CoinRowMapper());
    }

    public List<Coin> findSymbol(String symbol, String fromDate, String toDate, Integer limit) {
        StringBuilder sql = new StringBuilder(
                "SELECT \"date\", open, high, low, close, volume, symbol FROM ohlcv WHERE 1=1"
        );

        List<Object> params = new ArrayList<>();

        if (symbol != null && !symbol.isBlank()) {
            sql.append(" AND LOWER(symbol) = LOWER(?)");
            params.add(symbol);
        }
        if (fromDate != null && !fromDate.isBlank()) {
            sql.append(" AND \"date\" >= ?");
            params.add(fromDate);
        }
        if (toDate != null && !toDate.isBlank()) {
            sql.append(" AND \"date\" <= ?");
            params.add(toDate);
        }
        if (limit != null && limit > 0) {
            sql.append(" ORDER BY \"date\" LIMIT ?");
            params.add(limit);
        } else
            sql.append(" ORDER BY \"date\" LIMIT 15");

        return jdbcTemplate.query(sql.toString(), params.toArray(), new CoinRowMapper());
    }

    public List<Coin> findBySymbol(String symbol, int limit) {
        String sql = "SELECT \"date\", open, high, low, close, volume, symbol " +
                "FROM ohlcv WHERE symbol = ? ORDER BY \"date\" LIMIT ?";
        return jdbcTemplate.query(sql, new Object[]{symbol, limit}, new CoinRowMapper());
    }
}
