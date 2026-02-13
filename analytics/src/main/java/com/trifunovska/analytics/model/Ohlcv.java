package com.trifunovska.analytics.model;

import jakarta.persistence.*;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "ohlcv")
public class Ohlcv {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(name = "symbol", nullable = false)
    private String symbol;

    @Column(name = "date", nullable = false)
    private String date;

    @Column(name = "open", nullable = false)
    private Double open;

    @Column(name = "high", nullable = false)
    private Double high;

    @Column(name = "low", nullable = false)
    private Double low;

    @Column(name = "close", nullable = false)
    private Double close;

    @Column(name = "volume", nullable = false)
    private Double volume;
}