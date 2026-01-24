package trifunovska.analytics.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Coin {
    private String symbol;
    private String date;
    private Double open;
    private Double close;
    private Double high;
    private Double low;
    private Double volume;
}

