# Monopoly Pricing Under a Heterogeneous Boycott

This repository contains a research note and a companion model implementation.  The note develops a monopoly pricing model in which a boycott affects consumers differently depending on their willingness to pay.  Because the boycott disproportionately removes low‑valuation, price‑sensitive buyers, the residual demand becomes less elastic.  As a result, the optimal monopoly price can rise even though overall quantity falls. In Process at this moment.

## Contents

- **`Boycott_Intro.pdf`** – the research note. It outlines the baseline monopoly setup, introduces a valuation‑dependent survival probability for consumers during a boycott, derives residual demand, and shows how the boycott shifts optimal price and quantity.
- **`model.py`** – a Python script that implements the model. It defines a demand distribution and survival function `s(v, β)`, computes residual demand and profit, and solves for the profit‑maximizing price and quantity at different boycott intensities β.  (Rename if you use a different filename.)
