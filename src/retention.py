def generate_retention_strategies(user_inputs, churn_probability, shap_top_features):
    strategies = []

    contract = user_inputs.get("Contract", "")
    tenure = user_inputs.get("tenure", 0)
    internet = user_inputs.get("InternetService", "")
    tech_support = user_inputs.get("TechSupport", "")
    online_security = user_inputs.get("OnlineSecurity", "")
    payment = user_inputs.get("PaymentMethod", "")
    monthly_charges = user_inputs.get("MonthlyCharges", 0)
    senior = user_inputs.get("SeniorCitizen", 0)

    if contract == "Month-to-month":
        strategies.append({
            "priority": "HIGH",
            "action": "Offer Annual Contract Upgrade",
            "detail": "Provide a 15–20% discount if the customer switches to a 1 or 2-year contract.",
            "impact": "Reduces churn probability by ~25–35%",
            "icon": "📋",
            "color": "#ef4444",
        })

    if tenure < 12:
        strategies.append({
            "priority": "HIGH",
            "action": "Early Loyalty Reward",
            "detail": "Send a personalised 'thank you' offer with 2 months free service or a bill credit.",
            "impact": "Increases 12-month retention by ~18%",
            "icon": "🎁",
            "color": "#ef4444",
        })

    if internet == "Fiber optic" and tech_support == "No":
        strategies.append({
            "priority": "HIGH",
            "action": "Complimentary Tech Support Trial",
            "detail": "Offer 3 months of free Tech Support — Fiber optic customers without support churn at 2x the rate.",
            "impact": "Reduces dissatisfaction churn by ~20%",
            "icon": "🛠️",
            "color": "#ef4444",
        })

    if online_security == "No" and internet != "No":
        strategies.append({
            "priority": "MEDIUM",
            "action": "Bundle Online Security",
            "detail": "Promote a discounted security bundle. Customers with security add-ons have 30% lower churn.",
            "impact": "Increases perceived value & stickiness",
            "icon": "🔒",
            "color": "#f59e0b",
        })

    if payment == "Electronic check":
        strategies.append({
            "priority": "MEDIUM",
            "action": "Switch to Auto-Pay",
            "detail": "Offer a $5/month bill credit to migrate from electronic check to automatic bank transfer.",
            "impact": "Electronic check customers churn at 2x the rate of auto-pay customers",
            "icon": "💳",
            "color": "#f59e0b",
        })

    if monthly_charges > 80:
        strategies.append({
            "priority": "MEDIUM",
            "action": "Personalised Discount Offer",
            "detail": f"Current charges are ${monthly_charges:.0f}/mo. Offer a 10% loyalty discount to reduce perceived cost.",
            "impact": "Price sensitivity is a top churn driver for high-bill customers",
            "icon": "💰",
            "color": "#f59e0b",
        })

    if senior == 1:
        strategies.append({
            "priority": "LOW",
            "action": "Senior Customer Care Programme",
            "detail": "Enrol in dedicated senior support line with simplified billing and priority call handling.",
            "impact": "Improves NPS and reduces service-related churn",
            "icon": "🤝",
            "color": "#22c55e",
        })

    if not strategies:
        strategies.append({
            "priority": "LOW",
            "action": "Proactive Check-in Call",
            "detail": "Schedule a courtesy call to ensure satisfaction and offer any available loyalty rewards.",
            "impact": "Low-risk customers can still benefit from proactive engagement",
            "icon": "📞",
            "color": "#22c55e",
        })

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    strategies.sort(key=lambda x: priority_order[x["priority"]])
    return strategies


def get_churn_risk_label(churn_probability):
    if churn_probability >= 0.75:
        return "Critical", "#ef4444"
    elif churn_probability >= 0.50:
        return "High", "#f97316"
    elif churn_probability >= 0.30:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#22c55e"
