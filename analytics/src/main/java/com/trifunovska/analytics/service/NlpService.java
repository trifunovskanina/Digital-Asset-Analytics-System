package com.trifunovska.analytics.service;

import jakarta.annotation.Nullable;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class NlpService {

    private final RestTemplate restTemplate = new RestTemplate();

    public List<Map<String, Object>> predictSentiment(String fromDate, String toDate, Integer limit) {

        Map<String, Object> request = new HashMap<>();

        if (fromDate != null) request.put("fromDate", fromDate);
        if (toDate != null) request.put("toDate", toDate);
        if (limit != null) request.put("limit", limit);

        return restTemplate.postForObject(
                "http://nlp:8001/sentiment",
                request,
                List.class
        );
    }
}
