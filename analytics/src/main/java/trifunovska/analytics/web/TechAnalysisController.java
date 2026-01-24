package trifunovska.analytics.web;

import trifunovska.analytics.service.TechAnalysisService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class TechAnalysisController {

    private final TechAnalysisService techService;

    public TechAnalysisController(TechAnalysisService techService) {
        this.techService = techService;
    }

    @GetMapping("/tech-analysis-form")
    public String showForm() {
        return "tech-analysis-form";
    }

    @GetMapping("/tech-analysis")
    public String showTechnicalAnalysis(
            @RequestParam(defaultValue = "bitcoin") String symbol,
            @RequestParam(defaultValue = "1d") String timeframe,
            @RequestParam(defaultValue = "200") int limit,
            @RequestParam(required = false) String fromDate,
            @RequestParam(required = false) String toDate,
            Model model) throws Exception {

        var data = techService.runAnalysis(symbol.toLowerCase(), timeframe, limit, fromDate, toDate);

        model.addAttribute("symbol", symbol);
        model.addAttribute("timeframe", timeframe);
        model.addAttribute("data", data);
        model.addAttribute("fromDate", fromDate);
        model.addAttribute("toDate", toDate);

        return "tech-analysis";
    }

}
