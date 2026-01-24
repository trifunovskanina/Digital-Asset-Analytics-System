package trifunovska.analytics.service;

import org.jspecify.annotations.Nullable;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class NlpService {

    private final RestTemplate restTemplate = new RestTemplate();

    public @Nullable List predictSentiment(String fromDate, String toDate, Integer limit) {

        Map<String, Object> request = new HashMap<>();

        if (fromDate != null) request.put("fromDate", fromDate);
        if (toDate != null) request.put("toDate", toDate);
        if (limit != null) request.put("limit", limit);

        return restTemplate.postForObject(
                "http://nlp:8002/sentiment",
                request,
                List.class
        );
    }
//    [
//    {
//        "date": "2024-01-01",
//            "content": "Some news text",
//            "link": "https://...",
//            "sentiment_label": "POSITIVE",
//            "sentiment_score": 0.91
//    },
//    {
//        "date": "2024-01-02",
//            "content": "Other news",
//            "link": "https://...",
//            "sentiment_label": "NEGATIVE",
//            "sentiment_score": 0.87
//    }
//]
}
