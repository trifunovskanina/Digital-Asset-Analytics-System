package trifunovska.analytics.web;

import trifunovska.analytics.service.NlpService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;
import java.util.Map;

@Controller
public class NlpController {

    private final NlpService service;

    public NlpController(NlpService service) {
        this.service = service;
    }

    @GetMapping("/nlp")
    public String show(
            @RequestParam(required = false) String fromDate,
            @RequestParam(required = false) String toDate,
            @RequestParam(required = false, defaultValue = "30") Integer limit,
            Model model
    ) {

        List<Map<String, Object>> rows = service.predictSentiment(fromDate, toDate, limit);

        assert rows != null;
        long positiveCount = rows.stream()
                        .filter(row -> "POSITIVE".equals(row.get("sentiment_label")))
                                .count();
        long negativeCount = rows.stream()
                        .filter(row -> "NEGATIVE".equals(row.get("sentiment_label")))
                                .count();

        model.addAttribute("rows", rows);

        model.addAttribute("positiveCount", positiveCount);
        model.addAttribute("negativeCount", negativeCount);

        // so the page doesn't break
        model.addAttribute("fromDate", fromDate);
        model.addAttribute("toDate", toDate);
        model.addAttribute("limit", limit);

        return "nlp";
    }
}
