package trifunovska.analytics.service;

import trifunovska.analytics.model.Coin;
import trifunovska.analytics.repository.CoinRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CoinService {
    private final CoinRepository coinRepository;

    public CoinService(CoinRepository coinRepository) {
        this.coinRepository = coinRepository;
    }

    public List<Coin> findSymbol(String symbol, String fromDate, String toDate, Integer limit) {
        return coinRepository.findSymbol(symbol, fromDate, toDate, limit);
    }

    public List<Coin> findAll() {
        return coinRepository.findAll();
    }

    public List<Coin> findBySymbol(String symbol, int limit) {
        return coinRepository.findBySymbol(symbol, limit);
    }
}