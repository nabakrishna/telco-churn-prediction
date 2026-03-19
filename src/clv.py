import numpy as np


AVERAGE_RETENTION_COST = 50
AVERAGE_ACQUISITION_COST = 300
DISCOUNT_RATE = 0.10


def calculate_clv(monthly_charges, tenure_months, churn_probability, contract_type):
    contract_multipliers = {
        "Month-to-month": 1.0,
        "One year": 1.2,
        "Two year": 1.5,
    }
    loyalty_multiplier = contract_multipliers.get(contract_type, 1.0)

    expected_remaining_lifetime_months = max(1, (1 - churn_probability) * 24 * loyalty_multiplier)
    gross_clv = monthly_charges * expected_remaining_lifetime_months
    discounted_clv = gross_clv / (1 + DISCOUNT_RATE / 12) ** expected_remaining_lifetime_months

    return round(discounted_clv, 2)


def calculate_revenue_at_risk(monthly_charges, churn_probability, avg_months_lost=18):
    return round(monthly_charges * avg_months_lost * churn_probability, 2)


def calculate_retention_roi(revenue_at_risk, retention_cost=AVERAGE_RETENTION_COST):
    net_benefit = revenue_at_risk - retention_cost
    roi_percent = (net_benefit / retention_cost) * 100 if retention_cost > 0 else 0
    worthwhile = net_benefit > 0
    return {
        "revenue_at_risk": revenue_at_risk,
        "retention_cost": retention_cost,
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 1),
        "worthwhile": worthwhile,
    }


def get_clv_tier(clv_value):
    if clv_value >= 2000:
        return "Platinum", "#a78bfa"
    elif clv_value >= 1000:
        return "Gold", "#fbbf24"
    elif clv_value >= 500:
        return "Silver", "#94a3b8"
    else:
        return "Bronze", "#b45309"