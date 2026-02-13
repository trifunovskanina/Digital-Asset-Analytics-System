package com.trifunovska.analytics.repository;

import com.trifunovska.analytics.model.Ohlcv;
import org.springframework.data.jpa.repository.*;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface OhlcvRepository extends JpaRepository<Ohlcv, Integer> {
    @Query(nativeQuery = true,
            value = """
                SELECT *
                FROM ohlcv
                WHERE (:symbol IS NULL OR :symbol = '' OR symbol = :symbol) 
                    AND (:from_date IS NULL OR :from_date = '' OR \"date\" >= :from_date)
                    AND (:to_date IS NULL OR :to_date = '' OR \"date\" <= :to_date)
                ORDER BY \"date\" DESC
                LIMIT :limit
    """)
    List<Ohlcv> findSymbol(@Param("symbol") String symbol,
                           @Param("from_date") String fromDate,
                           @Param("to_date") String toDate,
                           @Param("limit") Integer limit);
}