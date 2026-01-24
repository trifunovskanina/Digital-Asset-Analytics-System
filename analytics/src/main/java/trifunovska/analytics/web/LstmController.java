package trifunovska.analytics.web;

import trifunovska.analytics.service.LstmService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class LstmController {

    private final LstmService lstmService;

    public LstmController(LstmService lstmService) {
        this.lstmService = lstmService;
    }

    @GetMapping("/lstm-form")
    public String showForm() {
        return "lstm-form";
    }

    @GetMapping("/lstm")
    public String showForecast(
            @RequestParam(defaultValue = "bitcoin") String symbol,
            @RequestParam(defaultValue = "10") int days,
            Model model) throws Exception {

        var result = lstmService.runForecast(symbol.toLowerCase(), days);

        model.addAttribute("symbol", symbol);
        model.addAttribute("days", days);
        model.addAttribute("result", result);

        return "lstm";
    }
}
