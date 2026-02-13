package com.trifunovska.analytics.service;

import com.trifunovska.analytics.model.Ohlcv;
import com.trifunovska.analytics.repository.OhlcvRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class OhlcvService {
    private final OhlcvRepository ohlcvRepository;

    public OhlcvService(OhlcvRepository ohlcvRepository) {
        this.ohlcvRepository = ohlcvRepository;
    }

    public List<Ohlcv> findSymbol(String symbol, String fromDate, String toDate, Integer limit) {
        return ohlcvRepository.findSymbol(symbol, fromDate, toDate, limit);
    }
}