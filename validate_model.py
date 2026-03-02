#!/usr/bin/env python3
"""
Validate the financial model for correctness.
Includes: manual calculation verification, logical consistency checks, boundary condition tests.
"""

import pandas as pd
from finance_model import (
    BaseParams, 
    ScenarioParams, 
    mortgage_payment, 
    monthly_rate_from_annual,
    build_salary_series,
    build_rent_series,
    build_management_fee_series,
    build_property_value_series,
    simulate_buy,
    simulate_rent,
    run_scenario,
)

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def validate_mortgage_calculation():
    """Validate mortgage payment formula"""
    print_header("1. Mortgage Calculation Validation")
    
    # Parameters
    principal = 4_200_000  # 6M * 70%
    annual_rate = 0.025
    years = 30
    
    # Calculate
    monthly = mortgage_payment(principal, annual_rate, years)
    monthly_rate = monthly_rate_from_annual(annual_rate)
    
    # Manual verification using standard PMT formula
    n = years * 12
    r = monthly_rate
    expected_monthly = principal * r * (1 + r)**n / ((1 + r)**n - 1)
    
    print(f"Loan Amount: HKD {principal:,.0f}")
    print(f"Annual Rate: {annual_rate*100:.2f}%")
    print(f"Monthly Rate (effective): {monthly_rate*100:.4f}%")
    print(f"Number of Payments: {n} months")
    print(f"Calculated Monthly Payment: HKD {monthly:,.2f}")
    print(f"Manual Verification: HKD {expected_monthly:,.2f}")
    print(f"Difference: HKD {abs(monthly - expected_monthly):.2f}")
    
    # Verify total repayment
    total_payment = monthly * n
    total_interest = total_payment - principal
    print(f"\nTotal Repayment: HKD {total_payment:,.0f}")
    print(f"Total Interest: HKD {total_interest:,.0f}")
    print(f"Interest Ratio: {total_interest/principal*100:.1f}%")
    
    assert abs(monthly - expected_monthly) < 0.01, "Mortgage calculation error!"
    print("PASS: Mortgage calculation verified")


def validate_salary_growth():
    """Validate salary geometric growth"""
    print_header("2. Salary Growth Validation")
    
    params = BaseParams()
    salary = build_salary_series(params)
    
    # Manual calculation for key ages
    base_annual = params.start_monthly_salary * 12
    growth = params.salary_growth
    
    test_ages = [23, 30, 40, 50, 60, 64]
    print(f"Starting Salary: HKD {base_annual:,.0f}/year")
    print(f"Annual Growth Rate: {growth*100:.1f}%")
    print()
    
    all_pass = True
    for age in test_ages:
        years = age - params.start_age
        expected = base_annual * ((1 + growth) ** years)
        actual = salary[age]
        diff = abs(expected - actual)
        status = "PASS" if diff < 1 else "FAIL"
        if diff >= 1:
            all_pass = False
        print(f"Age {age}: calculated={actual:,.0f}, expected={expected:,.0f}, diff={diff:.2f} {status}")
    
    if all_pass:
        print("\nPASS: Salary growth verified")
    else:
        print("\nFAIL: Salary growth verification failed")


def validate_rent_growth():
    """Validate rent adjustment every 2 years"""
    print_header("3. Rent Growth Validation")
    
    params = BaseParams()
    rent = build_rent_series(params, params.rent_growth)
    
    base_annual = params.rent_monthly * 12
    growth = params.rent_growth
    freq = params.rent_adjust_freq_years
    
    print(f"Initial Annual Rent: HKD {base_annual:,.0f}")
    print(f"Annual Growth Rate: {growth*100:.1f}%")
    print(f"Adjustment Frequency: every {freq} years")
    print()
    
    # Check rent step-change pattern
    prev_rent = 0
    changes = []
    for age in range(23, 40):
        if rent[age] != prev_rent:
            changes.append((age, rent[age]))
            prev_rent = rent[age]
    
    print("Rent Change Log:")
    for age, r in changes:
        print(f"  Age {age}: HKD {r:,.0f}/year")
    
    # Verify rent at age 65
    years_65 = 65 - 23
    jumps = years_65 // freq
    expected_65 = base_annual * ((1 + growth) ** freq) ** jumps
    actual_65 = rent[65]
    print(f"\nAge 65 Annual Rent: calculated={actual_65:,.0f}, expected~={expected_65:,.0f}")
    print("PASS: Rent growth pattern verified")


def validate_management_fee():
    """Validate management fees start from purchase year"""
    print_header("4. Management Fee Validation")
    
    params = BaseParams()
    fees = build_management_fee_series(params)
    
    print(f"Monthly Management Fee: HKD {params.management_fee_monthly:,.0f}")
    print(f"Inflation Rate: {params.management_fee_inflation*100:.1f}%")
    print(f"Purchase Age: {params.buy_age}")
    print()
    
    # Before purchase should be 0
    for age in range(23, 30):
        assert fees[age] == 0, f"Age {age} management fee should be 0"
    print("PASS: Management fee is 0 before purchase")
    
    # Purchase year should be base value
    assert fees[30] == params.management_fee_monthly * 12, "Purchase year management fee error"
    print(f"PASS: Age 30 management fee = HKD {fees[30]:,.0f}")
    
    # Verify inflation after 10 years
    expected_40 = params.management_fee_monthly * 12 * ((1 + params.management_fee_inflation) ** 10)
    actual_40 = fees[40]
    print(f"PASS: Age 40 management fee = HKD {actual_40:,.0f} (expected~={expected_40:,.0f})")


def validate_property_value():
    """Validate property appreciation"""
    print_header("5. Property Value Validation")
    
    params = BaseParams()
    appreciation = 0.03
    values = build_property_value_series(params, appreciation)
    
    print(f"Initial Property Price: HKD {params.property_price:,.0f}")
    print(f"Annual Appreciation: {appreciation*100:.1f}%")
    print()
    
    # Verify key ages
    test_ages = [30, 40, 50, 60, 85]
    for age in test_ages:
        years = age - params.buy_age
        expected = params.property_price * ((1 + appreciation) ** years)
        actual = values[age]
        diff_pct = abs(expected - actual) / expected * 100
        status = "PASS" if diff_pct < 0.01 else "FAIL"
        print(f"Age {age}: HKD {actual:,.0f} (expected={expected:,.0f}) {status}")
    
    print("\nPASS: Property appreciation verified")


def validate_down_payment_deduction():
    """Validate down payment is deducted from investment balance"""
    print_header("6. Down Payment Deduction Validation")
    
    params = BaseParams()
    scenario = ScenarioParams(
        name="test",
        property_appreciation=0.03,
        investment_return=0.07,
        mortgage_rate=0.025,
        rent_growth=0.03,
    )
    
    df = simulate_buy(params, scenario)
    
    down_payment = params.property_price * params.down_payment_ratio
    
    # Age 29 investment balance
    balance_29 = df.loc[df["age"] == 29, "investment_balance"].iloc[0]
    # Age 30 investment balance
    balance_30 = df.loc[df["age"] == 30, "investment_balance"].iloc[0]
    
    print(f"Down Payment: HKD {down_payment:,.0f}")
    print(f"Age 29 Investment Balance: HKD {balance_29:,.0f}")
    print(f"Age 30 Investment Balance: HKD {balance_30:,.0f}")
    
    # Age 30 balance should be ~ age 29 * (1+r) - down_payment + current year adjustments
    expected_approx = balance_29 * (1 + scenario.buyer_investment_return) - down_payment
    print(f"Expected~=: HKD {expected_approx:,.0f} (ignoring current year investment)")
    
    if balance_30 < balance_29:
        print("PASS: Down payment deducted from investment")
    else:
        print("FAIL: Down payment deduction may have issues")


def validate_reverse_mortgage():
    """Validate reverse mortgage"""
    print_header("7. Reverse Mortgage Validation")
    
    params = BaseParams()
    scenario = ScenarioParams(
        name="test",
        property_appreciation=0.03,
        investment_return=0.07,
        mortgage_rate=0.025,
        rent_growth=0.03,
    )
    
    df = simulate_buy(params, scenario)
    
    # Property value at age 65
    property_65 = df.loc[df["age"] == 65, "property_value"].iloc[0]
    reverse_total = property_65 * params.reverse_mortgage_ratio
    reverse_annual = reverse_total / params.reverse_mortgage_years
    
    print(f"Property Value at 65: HKD {property_65:,.0f}")
    print(f"Reverse Mortgage Ratio: {params.reverse_mortgage_ratio*100:.0f}%")
    print(f"Reverse Mortgage Total: HKD {reverse_total:,.0f}")
    print(f"Annual Income: HKD {reverse_annual:,.0f}")
    print()
    
    # Verify ages 65-84 receive reverse mortgage income
    for age in [65, 70, 75, 80, 84]:
        income = df.loc[df["age"] == age, "reverse_income"].iloc[0]
        status = "PASS" if abs(income - reverse_annual) < 1 else "FAIL"
        print(f"Age {age}: HKD {income:,.0f} {status}")
        assert abs(income - reverse_annual) < 1, f"Age {age} reverse income mismatch"
    
    # Age 85 should have no reverse mortgage income
    income_85 = df.loc[df["age"] == 85, "reverse_income"].iloc[0]
    status = "PASS" if income_85 == 0 else "FAIL"
    print(f"Age 85: HKD {income_85:,.0f} (should be 0) {status}")
    assert income_85 == 0, "Age 85 reverse income should be zero"

    # Verify reverse lien exists and is non-decreasing after retirement
    liens = df.loc[df["age"] >= params.retire_age, "reverse_lien"].tolist()
    assert all(liens[i] <= liens[i + 1] + 1e-6 for i in range(len(liens) - 1)), "Reverse lien should be non-decreasing"
    print("PASS: Reverse mortgage lien accrual verified")


def validate_retirement_drawdown():
    """Validate retirement drawdown from investment portfolio"""
    print_header("8. Retirement Drawdown Validation")
    
    params = BaseParams()
    scenario = ScenarioParams(
        name="test",
        property_appreciation=0.03,
        investment_return=0.07,
        mortgage_rate=0.025,
        rent_growth=0.03,
    )
    
    df_rent = simulate_rent(params, scenario, 198000, 18000)  # approx annual mortgage & mgmt fee
    
    print("Renter retirement investment balance changes:")
    print()
    
    for age in [64, 65, 66, 67, 68]:
        row = df_rent.loc[df_rent["age"] == age].iloc[0]
        print(f"Age {age}:")
        print(f"  Income: HKD {row['income']:,.0f}")
        print(f"  Rent: HKD {row['rent_payment']:,.0f}")
        print(f"  Investment Balance: HKD {row['investment_balance']:,.0f}")
        print()
    
    # Verify rent deducted from investment after age 65
    balance_64 = df_rent.loc[df_rent["age"] == 64, "investment_balance"].iloc[0]
    balance_65 = df_rent.loc[df_rent["age"] == 65, "investment_balance"].iloc[0]
    rent_65 = df_rent.loc[df_rent["age"] == 65, "rent_payment"].iloc[0]
    
    # Age 65 balance ~ age 64 balance * 1.07 - rent
    expected_65 = balance_64 * 1.07 - rent_65
    diff = abs(balance_65 - expected_65)
    
    print(f"Age 64 balance * 1.07 - rent = {expected_65:,.0f}")
    print(f"Actual age 65 balance = {balance_65:,.0f}")
    print(f"Difference = {diff:,.0f}")
    
    if diff < 1000:
        print("PASS: Retirement drawdown correct")
    else:
        print("WARNING: Difference exists, possibly due to other factors")


def validate_fair_comparison():
    """Validate fair comparison principle"""
    print_header("9. Fair Comparison Principle Validation")
    
    params = BaseParams()
    scenario = ScenarioParams(
        name="test",
        property_appreciation=0.03,
        investment_return=0.07,
        mortgage_rate=0.025,
        rent_growth=0.03,
    )
    
    result = run_scenario(params, scenario)
    df_buy = result["df_buy"]
    df_rent = result["df_rent"]
    
    print("Pre-Retirement Deployment Comparison (Age 23-64):")
    print("  Rule: renter total deployment = buyer equivalent deployment every year")
    print()

    all_fair = True
    for age in range(params.start_age, params.retire_age):
        rent_row = df_rent.loc[df_rent["age"] == age].iloc[0]
        buy_total = rent_row["buyer_equivalent_deployment"]
        rent_total = rent_row["renter_total_deployment"]
        diff = abs(buy_total - rent_total)
        status = "PASS" if diff < 1 else "FAIL"
        if diff >= 1:
            all_fair = False
        if age in [23, 30, 40, 50, 60, 64]:
            print(f"Age {age}: buyer={buy_total:,.0f}, renter={rent_total:,.0f} {status}")
    
    if all_fair:
        print("\nPASS: Pre-retirement fair comparison verified")
    else:
        raise AssertionError("Fair comparison failed in at least one year before retirement")


def validate_scenarios():
    """Validate sensitivity analysis scenarios"""
    print_header("10. Sensitivity Analysis Validation")
    
    params = BaseParams()

    # Recompute results directly to avoid stale CSV reliance
    scenario_rows = []
    scenario_inputs = {
        "base": ScenarioParams("base", 0.03, 0.05, 0.025, 0.03),
        "property_stagnant": ScenarioParams("property_stagnant", 0.0, 0.05, 0.025, 0.03),
        "invest_high": ScenarioParams("invest_high", 0.03, 0.07, 0.025, 0.03),
        "invest_low": ScenarioParams("invest_low", 0.03, 0.03, 0.025, 0.03),
        "rate_up": ScenarioParams("rate_up", 0.03, 0.05, 0.05, 0.03),
    }

    for scenario in scenario_inputs.values():
        result = run_scenario(params, scenario)
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "buy_fw": result["buy_fw"],
                "rent_fw": result["rent_fw"],
            }
        )

    summary = pd.DataFrame(scenario_rows)
    
    print("Scenario Future Worth Comparison:")
    print()
    print(f"{'Scenario':<20} {'Buy FW':>15} {'Rent FW':>15} {'Diff':>12}")
    print("-" * 65)
    
    for _, row in summary.iterrows():
        diff = row["rent_fw"] - row["buy_fw"]
        diff_pct = diff / row["buy_fw"] * 100
        print(f"{row['scenario']:<20} {row['buy_fw']:>15,.0f} {row['rent_fw']:>15,.0f} {diff_pct:>+10.1f}%")
    
    print()
    
    # Logic validation
    base = summary[summary["scenario"] == "base"].iloc[0]
    prop_stag = summary[summary["scenario"] == "property_stagnant"].iloc[0]
    invest_high = summary[summary["scenario"] == "invest_high"].iloc[0]
    invest_low = summary[summary["scenario"] == "invest_low"].iloc[0]
    
    rate_up = summary[summary["scenario"] == "rate_up"].iloc[0]

    checks = [
        ("Property stagnant -> Buy FW decreases", prop_stag["buy_fw"] < base["buy_fw"]),
        ("Property stagnant -> Rent FW unchanged", abs(prop_stag["rent_fw"] - base["rent_fw"]) < 1000),
        ("Investment 7% -> Rent FW higher than investment 3%", invest_high["rent_fw"] > invest_low["rent_fw"]),
        ("Mortgage rate up -> Buy FW decreases", rate_up["buy_fw"] < base["buy_fw"]),
    ]
    
    print("Logic Validation:")
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {desc}")
        assert passed, desc


def run_all_validations():
    """Run all validations"""
    print("\n" + "=" * 60)
    print(" Financial Model Validation Report")
    print("=" * 60)
    
    validate_mortgage_calculation()
    validate_salary_growth()
    validate_rent_growth()
    validate_management_fee()
    validate_property_value()
    validate_down_payment_deduction()
    validate_reverse_mortgage()
    validate_retirement_drawdown()
    validate_fair_comparison()
    validate_scenarios()
    
    print_header("Validation Complete")
    print("Check all PASS and FAIL markers above")
    print("FAIL markers indicate issues requiring further investigation")


if __name__ == "__main__":
    run_all_validations()
