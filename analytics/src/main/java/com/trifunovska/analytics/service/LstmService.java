package com.trifunovska.analytics.service;

import org.jspecify.annotations.Nullable;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
public class LstmService {

    private final RestTemplate restTemplate = new RestTemplate();

    public Map<String, Object> runForecast(String symbol, int days) {

        Map<String, Object> request = Map.of(
                "symbol", symbol,
                "n_future_days", days
        );

        return restTemplate.postForObject(
                "http://lstm:8000/predict",   // docker service name
                request,  // convert request to json
                Map.class  // convert response to map
        );
    }
}
