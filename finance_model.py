import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BaseParams:
    start_age: int = 23
    end_age: int = 85
    buy_age: int = 30
    retire_age: int = 65
    start_monthly_salary: float = 25000.0
    salary_growth: float = 0.04
    property_price: float = 6_000_000.0
    down_payment_ratio: float = 0.30
    mortgage_years: int = 30
    mortgage_rate: float = 0.025
    property_appreciation: float = 0.03
    management_fee_monthly: float = 1500.0
    management_fee_inflation: float = 0.02
    renovation_cost: float = 300_000.0
    renovation_interval_years: int = 15
    reverse_mortgage_ratio: float = 0.50
    reverse_mortgage_years: int = 20
    reverse_mortgage_interest: float = 0.025
    rent_monthly: float = 14_000.0
    rent_growth: float = 0.03
    rent_adjust_freq_years: int = 2
    investment_return: float = 0.07
    conservative_return: float = 0.03
    general_inflation: float = 0.02
    savings_rate: float = 0.40
    # Stamp Duty for first-time buyer (Ad Valorem Stamp Duty)
    # HK rates: 1.5% for <= 2M, 2.25% for 2-3M, 3% for 3-4M, 3.75% for 4-6M, 4.25% for 6-20M
    stamp_duty_rate: float = 0.0375  # 3.75% for 6M property
    # Government Rates: ~5% of rateable value per quarter, approx 1.5-2% of rent annually
    government_rates_ratio: float = 0.05  # 5% of estimated rental value annually


@dataclass
class ScenarioParams:
    name: str
    property_appreciation: float
    investment_return: float  # Renter's investment return (stocks)
    mortgage_rate: float
    rent_growth: float
    buyer_investment_return: float = None  # Buyer's investment return, defaults to conservative_return if None
    
    def __post_init__(self):
        # If buyer_investment_return not specified, use a more conservative rate
        if self.buyer_investment_return is None:
            self.buyer_investment_return = 0.04


KEY_AGES = [30, 40, 50, 60, 65, 85]


def monthly_rate_from_annual(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    r = monthly_rate_from_annual(annual_rate)
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def build_salary_series(params: BaseParams) -> Dict[int, float]:
    salary = {}
    for age in range(params.start_age, params.end_age + 1):
        years = age - params.start_age
        annual = params.start_monthly_salary * 12 * ((1 + params.salary_growth) ** years)
        salary[age] = annual
    return salary


def build_rent_series(params: BaseParams, rent_growth: float) -> Dict[int, float]:
    rent = {}
    current_rent = params.rent_monthly
    for age in range(params.start_age, params.end_age + 1):
        rent[age] = current_rent * 12
        if (age - params.start_age + 1) % params.rent_adjust_freq_years == 0:
            current_rent *= (1 + rent_growth) ** params.rent_adjust_freq_years
    return rent


def build_management_fee_series(params: BaseParams) -> Dict[int, float]:
    """Management fee starts from buy_age, not start_age."""
    fees = {}
    for age in range(params.start_age, params.end_age + 1):
        if age < params.buy_age:
            fees[age] = 0.0
        else:
            years_since_purchase = age - params.buy_age
            fees[age] = params.management_fee_monthly * 12 * ((1 + params.management_fee_inflation) ** years_since_purchase)
    return fees


def build_property_value_series(params: BaseParams, appreciation: float) -> Dict[int, float]:
    values = {}
    current_value = params.property_price
    for age in range(params.buy_age, params.end_age + 1):
        values[age] = current_value
        current_value *= (1 + appreciation)
    return values


def build_renovation_schedule(params: BaseParams) -> Dict[int, float]:
    schedule = {}
    age = params.buy_age
    while age <= params.end_age:
        if age != params.buy_age:
            years_since = age - params.buy_age
            schedule[age] = params.renovation_cost * ((1 + params.general_inflation) ** years_since)
        age += params.renovation_interval_years
    return schedule


def simulate_buy(params: BaseParams, scenario: ScenarioParams) -> pd.DataFrame:
    salary = build_salary_series(params)
    management_fee = build_management_fee_series(params)
    property_values = build_property_value_series(params, scenario.property_appreciation)
    renovation_schedule = build_renovation_schedule(params)
    down_payment = params.property_price * params.down_payment_ratio
    loan_amount = params.property_price - down_payment
    monthly_payment = mortgage_payment(loan_amount, scenario.mortgage_rate, params.mortgage_years)
    annual_mortgage_payment = monthly_payment * 12

    # Calculate required annual savings to afford down payment
    saving_years = params.buy_age - params.start_age
    required_annual_savings = down_payment / saving_years

    # Stamp duty paid at purchase
    stamp_duty = params.property_price * params.stamp_duty_rate

    # Government rates based on estimated rental value (at purchase time)
    base_rental_value = params.rent_monthly * 12

    reverse_total = 0.0
    reverse_annual = 0.0
    if params.retire_age in property_values:
        reverse_total = property_values[params.retire_age] * params.reverse_mortgage_ratio
        reverse_annual = reverse_total / params.reverse_mortgage_years

    records = []
    investment_balance = 0.0
    reverse_outstanding = 0.0

    for age in range(params.start_age, params.end_age + 1):
        income = salary[age] if age < params.retire_age else 0.0
        savings = 0.0
        mortgage = 0.0
        fees = 0.0
        renovation = 0.0
        reverse_income = 0.0
        stamp_duty_cost = 0.0
        govt_rates = 0.0
        rent_payment = 0.0

        if age < params.buy_age:
            # During saving period, buyer lives with parents (no rent) and saves aggressively
            rent_payment = 0.0
            savings = required_annual_savings  # Fixed amount to meet down payment target
        elif age == params.buy_age:
            # Pay stamp duty and use savings for down payment
            stamp_duty_cost = stamp_duty
            mortgage = annual_mortgage_payment
            fees = management_fee[age]
            # Government rates (based on rental value at purchase, then inflation adjusted)
            years_since_purchase = age - params.buy_age
            govt_rates = base_rental_value * ((1 + params.general_inflation) ** years_since_purchase) * params.government_rates_ratio
        elif params.buy_age < age < params.retire_age:
            mortgage = annual_mortgage_payment if age < params.buy_age + params.mortgage_years else 0.0
            fees = management_fee[age]
            renovation = renovation_schedule.get(age, 0.0)
            years_since_purchase = age - params.buy_age
            govt_rates = base_rental_value * ((1 + params.general_inflation) ** years_since_purchase) * params.government_rates_ratio
        else:
            # Retirement period
            fees = management_fee[age]
            years_since_purchase = age - params.buy_age
            govt_rates = base_rental_value * ((1 + params.general_inflation) ** years_since_purchase) * params.government_rates_ratio
            renovation = renovation_schedule.get(age, 0.0)
            if params.retire_age <= age < params.retire_age + params.reverse_mortgage_years:
                reverse_income = reverse_annual

        # Calculate total housing cost for this year
        housing_cost = mortgage + fees + renovation + stamp_duty_cost + govt_rates + rent_payment
        net_cashflow = income + reverse_income - savings - housing_cost

        # Investment logic - buyer uses blended investment return from scenario
        investment_balance = investment_balance * (1 + scenario.buyer_investment_return)

        # Add savings during accumulation phase
        investment_balance += savings

        # Deduct down payment when purchasing
        if age == params.buy_age:
            investment_balance -= down_payment

        # Funding shortfall is deducted from investment balance
        if net_cashflow < 0:
            investment_balance += net_cashflow

        # Reverse mortgage debt accrues with interest after retirement
        if age >= params.retire_age:
            reverse_outstanding = (reverse_outstanding + reverse_income) * (1 + params.reverse_mortgage_interest)

        property_value = property_values.get(age, 0.0)
        # After retirement, subtract reverse mortgage lien from property value
        if age >= params.retire_age:
            reverse_lien = min(reverse_outstanding, property_value)
            net_worth = investment_balance + property_value - reverse_lien
        else:
            reverse_lien = 0.0
            net_worth = investment_balance + property_value

        records.append(
            {
                "age": age,
                "income": income,
                "savings": savings,
                "rent_payment": rent_payment,
                "mortgage_payment": mortgage,
                "management_fee": fees,
                "govt_rates": govt_rates,
                "stamp_duty": stamp_duty_cost,
                "renovation": renovation,
                "reverse_income": reverse_income,
                "reverse_lien": reverse_lien,
                "net_cashflow": net_cashflow,
                "investment_balance": investment_balance,
                "property_value": property_value,
                "net_worth": net_worth,
            }
        )

    return pd.DataFrame(records)


def simulate_rent(params: BaseParams, scenario: ScenarioParams, mortgage_payment_annual: float, management_fee_annual: float) -> pd.DataFrame:
    salary = build_salary_series(params)
    rent = build_rent_series(params, scenario.rent_growth)

    down_payment = params.property_price * params.down_payment_ratio
    saving_years = params.buy_age - params.start_age
    annual_down_payment_saving = down_payment / saving_years

    # Stamp duty that buyer pays at purchase - renter invests this instead
    stamp_duty = params.property_price * params.stamp_duty_rate

    # Renovation schedule for fair comparison with buyer
    renovation_schedule = build_renovation_schedule(params)

    # Base rental value for calculating buyer's govt rates
    base_rental_value = params.rent_monthly * 12

    records = []
    investment_balance = 0.0

    for age in range(params.start_age, params.end_age + 1):
        income = salary[age] if age < params.retire_age else 0.0
        rent_payment = rent[age]  # Renter always pays rent
        savings = 0.0
        extra_invest = 0.0
        buyer_total_deployment = 0.0

        if age < params.buy_age:
            # Fair comparison: renter deployment (rent + invest) equals buyer's down-payment saving
            buyer_total_deployment = annual_down_payment_saving
            extra_invest = buyer_total_deployment - rent_payment
        elif age == params.buy_age:
            # Calculate buyer's total housing cost at this age (including stamp duty)
            years_since_purchase = 0
            buyer_govt_rates = base_rental_value * params.government_rates_ratio
            buyer_mgmt_fee = params.management_fee_monthly * 12
            buyer_renovation = renovation_schedule.get(age, 0.0)
            buyer_total_deployment = mortgage_payment_annual + buyer_mgmt_fee + buyer_govt_rates + stamp_duty + buyer_renovation
            # Renter invests/withdraws difference to keep total deployment equivalent
            extra_invest = buyer_total_deployment - rent_payment
        elif params.buy_age < age < params.retire_age:
            # Calculate buyer's total housing cost at this age
            years_since_purchase = age - params.buy_age
            buyer_govt_rates = base_rental_value * ((1 + params.general_inflation) ** years_since_purchase) * params.government_rates_ratio
            buyer_mgmt_fee = params.management_fee_monthly * 12 * ((1 + params.management_fee_inflation) ** years_since_purchase)
            buyer_renovation = renovation_schedule.get(age, 0.0)
            # Mortgage ends after mortgage_years
            buyer_mortgage = mortgage_payment_annual if age < params.buy_age + params.mortgage_years else 0.0
            buyer_total_deployment = buyer_mortgage + buyer_mgmt_fee + buyer_govt_rates + buyer_renovation
            # Renter invests/withdraws difference to keep total deployment equivalent
            extra_invest = buyer_total_deployment - rent_payment
        # else: retirement - no new investments, just pay rent from portfolio

        # Investment growth first
        investment_balance = investment_balance * (1 + scenario.investment_return)
        # Add new investments
        investment_balance += savings + extra_invest
        # Deduct rent from investment during retirement (no income)
        if age >= params.retire_age:
            investment_balance -= rent_payment

        net_cashflow = income - savings - rent_payment - extra_invest
        net_worth = investment_balance

        records.append(
            {
                "age": age,
                "income": income,
                "rent_payment": rent_payment,
                "savings": savings,
                "extra_invest": extra_invest,
                "buyer_equivalent_deployment": buyer_total_deployment,
                "renter_total_deployment": rent_payment + extra_invest,
                "net_cashflow": net_cashflow,
                "investment_balance": investment_balance,
                "net_worth": net_worth,
            }
        )

    return pd.DataFrame(records)


def summarize_key_ages(df_buy: pd.DataFrame, df_rent: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for age in KEY_AGES:
        buy_row = df_buy.loc[df_buy["age"] == age].iloc[0]
        rent_row = df_rent.loc[df_rent["age"] == age].iloc[0]
        summary.append(
            {
                "age": age,
                "buy_net_worth": buy_row["net_worth"],
                "rent_net_worth": rent_row["net_worth"],
            }
        )
    return pd.DataFrame(summary)


def build_scenarios(params: BaseParams) -> List[ScenarioParams]:
    return [
        ScenarioParams(
            name="base",
            property_appreciation=0.03,
            investment_return=0.05,
            mortgage_rate=0.025,
            rent_growth=0.03,
        ),
        ScenarioParams(
            name="property_stagnant",
            property_appreciation=0.0,
            investment_return=0.05,
            mortgage_rate=0.025,
            rent_growth=0.03,
        ),
        ScenarioParams(
            name="invest_high",
            property_appreciation=0.03,
            investment_return=0.07,
            mortgage_rate=0.025,
            rent_growth=0.03,
        ),
        ScenarioParams(
            name="invest_low",
            property_appreciation=0.03,
            investment_return=0.03,
            mortgage_rate=0.025,
            rent_growth=0.03,
        ),
        ScenarioParams(
            name="rate_up",
            property_appreciation=0.03,
            investment_return=0.07,
            mortgage_rate=0.05,
            rent_growth=0.03,
        ),
        ScenarioParams(
            name="rent_high",
            property_appreciation=0.03,
            investment_return=0.07,
            mortgage_rate=0.025,
            rent_growth=0.05,
        ),
        ScenarioParams(
            name="rent_low",
            property_appreciation=0.03,
            investment_return=0.07,
            mortgage_rate=0.025,
            rent_growth=0.01,
        ),
    ]


def plot_cashflow(df_buy: pd.DataFrame, df_rent: pd.DataFrame, title: str, output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df_buy["age"], df_buy["net_cashflow"], label="Buy")
    plt.plot(df_rent["age"], df_rent["net_cashflow"], label="Rent")
    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Net Cashflow (HKD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_net_worth(df_buy: pd.DataFrame, df_rent: pd.DataFrame, title: str, output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df_buy["age"], df_buy["net_worth"], label="Buy")
    plt.plot(df_rent["age"], df_rent["net_worth"], label="Rent")
    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Net Worth (HKD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sensitivity_spider(results: pd.DataFrame, output_path: str) -> None:
    categories = results["scenario"].tolist()
    buy_values = results["buy_fw"].tolist()
    rent_values = results["rent_fw"].tolist()

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    buy_values += buy_values[:1]
    rent_values += rent_values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, buy_values, label="Buy")
    ax.fill(angles, buy_values, alpha=0.15)
    ax.plot(angles, rent_values, label="Rent")
    ax.fill(angles, rent_values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Sensitivity Spider (Future Worth)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_scenario_bar(results: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 5))
    results_melt = results.melt(id_vars="scenario", value_vars=["buy_fw", "rent_fw"], var_name="strategy", value_name="future_worth")
    sns.barplot(data=results_melt, x="scenario", y="future_worth", hue="strategy")
    plt.title("Scenario Future Worth Comparison")
    plt.xlabel("Scenario")
    plt.ylabel("Future Worth (HKD)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_scenario(params: BaseParams, scenario: ScenarioParams) -> Dict[str, object]:
    down_payment = params.property_price * params.down_payment_ratio
    loan_amount = params.property_price - down_payment
    monthly_payment = mortgage_payment(loan_amount, scenario.mortgage_rate, params.mortgage_years)
    annual_mortgage_payment = monthly_payment * 12
    management_fee_annual = params.management_fee_monthly * 12

    df_buy = simulate_buy(params, scenario)
    df_rent = simulate_rent(params, scenario, annual_mortgage_payment, management_fee_annual)

    key_summary = summarize_key_ages(df_buy, df_rent)

    return {
        "df_buy": df_buy,
        "df_rent": df_rent,
        "key_summary": key_summary,
        "buy_fw": df_buy.iloc[-1]["net_worth"],
        "rent_fw": df_rent.iloc[-1]["net_worth"],
    }


def main() -> None:
    params = BaseParams()
    scenarios = build_scenarios(params)

    results_rows = []
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    for scenario in scenarios:
        result = run_scenario(params, scenario)
        df_buy = result["df_buy"]
        df_rent = result["df_rent"]

        df_buy.to_csv(f"{output_dir}/cashflow_buy_{scenario.name}.csv", index=False)
        df_rent.to_csv(f"{output_dir}/cashflow_rent_{scenario.name}.csv", index=False)
        result["key_summary"].to_csv(f"{output_dir}/key_ages_{scenario.name}.csv", index=False)

        plot_cashflow(
            df_buy,
            df_rent,
            f"Cashflow Comparison - {scenario.name}",
            f"{output_dir}/cashflow_{scenario.name}.png",
        )
        plot_net_worth(
            df_buy,
            df_rent,
            f"Net Worth Comparison - {scenario.name}",
            f"{output_dir}/networth_{scenario.name}.png",
        )

        results_rows.append(
            {
                "scenario": scenario.name,
                "buy_fw": result["buy_fw"],
                "rent_fw": result["rent_fw"],
            }
        )

    results = pd.DataFrame(results_rows)
    results.to_csv(f"{output_dir}/scenario_summary.csv", index=False)

    plot_sensitivity_spider(results, f"{output_dir}/sensitivity_spider.png")
    plot_scenario_bar(results, f"{output_dir}/scenario_bar.png")


if __name__ == "__main__":
    main()
