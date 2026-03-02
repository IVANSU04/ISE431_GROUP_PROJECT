# ISE431_GROUP_PROJECT

## Overview

This project simulates two mutually exclusive life-cycle housing strategies in Hong Kong from age 23 to 85:

- Option A: Buy a flat and hold property
- Option B: Rent and invest the equivalent capital

The model outputs annual cash-flow and net-worth trajectories and compares Future Worth (FW) at age 85.

## How to Run

```bash
python finance_model.py
python validate_model.py
```

Generated files are saved under `outputs/`.

## Modeling Rules (Implemented)

- Mortgage uses effective monthly compounding.
- Salary follows geometric growth until retirement (age 65).
- Rent increases with geometric step adjustment every 2 years.
- Buyer costs include down payment, mortgage, stamp duty, management fee, government rates, and periodic renovation.
- Reverse mortgage starts at age 65 with annual draw; outstanding lien accrues interest and is deducted from property value.
- Fair-comparison principle is enforced pre-retirement:
	- `renter total deployment = rent + investment`
	- `buyer equivalent deployment = buyer required annual housing/deposit deployment`
	- The two are made equal each year from age 23 to 64.

## Scenario Set

- `base`: investment return 5%
- `property_stagnant`: property appreciation 0%
- `invest_high`: investment return 7%
- `invest_low`: investment return 3%
- `rate_up`: mortgage rate up to 5%
- `rent_high`: rent growth 5%
- `rent_low`: rent growth 1%

## Parameter Evidence Checklist (for Report Writing)

Use this checklist in your report and fill each item with citation + date accessed:

- [ ] Mortgage rate basis (HIBOR/P-rate package, effective annual + monthly conversion)
- [ ] Investment return assumption (portfolio composition, long-run annualized return)
- [ ] Salary starting point and growth assumption for engineering graduates
- [ ] Property price baseline and historical appreciation reference
- [ ] Rent baseline and rent inflation reference
- [ ] General inflation (CPI) reference
- [ ] Management fee and government rates assumption source
- [ ] Reverse mortgage assumption source (programme terms/constraints)

## Notes

- The script is deterministic (no Monte Carlo).
- Output numbers are assumption-driven; sensitivity scenarios should be interpreted with parameter evidence in the report.