package com.trifunovska.analytics.web;

import com.trifunovska.analytics.model.Ohlcv;
import com.trifunovska.analytics.service.OhlcvService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.accept.DefaultApiVersionStrategy;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;
import java.util.stream.Collectors;

@Controller
public class OhlcvController {
    private final OhlcvService ohlcvService;

    public OhlcvController(OhlcvService ohlcvService) {
        this.ohlcvService = ohlcvService;
    }

    @GetMapping
    public String showTable(
            @RequestParam(required = false) String symbol,
            @RequestParam(required = false) String fromDate,
            @RequestParam(required = false) String toDate,
            @RequestParam(defaultValue = "20") Integer limit,
            Model model) {

        List<Ohlcv> rows = ohlcvService.findSymbol(symbol, fromDate, toDate, limit);
        model.addAttribute("rows", rows);
        model.addAttribute("symbol", symbol);
        model.addAttribute("fromDate", fromDate);
        model.addAttribute("toDate", toDate);
        model.addAttribute("limit", limit);

        return "index";
    }

    @GetMapping("/coin/{symbol}")
    public String showCoin(
            @PathVariable String symbol,
            @RequestParam(required = false, defaultValue = "60") Integer limit,
            Model model) {

        List<Ohlcv> rows = ohlcvService.findSymbol(symbol, null, null, limit);

        List<String> labels = rows.stream()
                .map(Ohlcv::getDate)
                .collect(Collectors.toList());

        List<Double> closes = rows.stream()
                .map(Ohlcv::getClose)
                .collect(Collectors.toList());

        model.addAttribute("symbol", symbol.toUpperCase());
        model.addAttribute("rows", rows);
        model.addAttribute("labels", labels);
        model.addAttribute("closes", closes);
        model.addAttribute("limit", limit);

        return "coin";
    }
}
