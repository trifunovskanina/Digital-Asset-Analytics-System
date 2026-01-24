package trifunovska.analytics.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

@Service
public class TechAnalysisService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper mapper = new ObjectMapper();

    public List<Map<String, Object>> runAnalysis(String symbol, String timeframe, int limit, String fromDate, String toDate) throws Exception {

        Map<String, Object> request = Map.of(
                "symbol", symbol,
                "timeframe", timeframe,
                "limit", limit
        );

        Map response =
                restTemplate.postForObject(
                        "http://tech:8001/tech-analysis",
                        request,  // convert request to json
                        Map.class // convert response to map
                );

        assert response != null;
        if (!response.containsKey("data"))
            throw new RuntimeException("Invalid response from FastAPI tech-analysis");

        List<Map<String, Object>> data =
                mapper.convertValue(response.get("data"),
                        new TypeReference<>() {});

        LocalDate from = (fromDate != null && !fromDate.isBlank())
                ? LocalDate.parse(fromDate)
                : null;

        LocalDate to = (toDate != null && !toDate.isBlank())
                ? LocalDate.parse(toDate)
                : null;

        if (from == null && to == null)
            return data;

        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        return data.stream()
                .filter(row -> {
                    Object dateObj = row.get("date");
                    if (dateObj == null) return false;

                    LocalDate rowDate;
                    try {
                        rowDate = LocalDateTime.parse(dateObj.toString(), fmt).toLocalDate();
                    } catch (Exception e) {
                        return false;
                    }

                    if (from != null && rowDate.isBefore(from)) return false;
                    if (to != null && rowDate.isAfter(to)) return false;

                    return true;
                })
                .toList();
    }
}
