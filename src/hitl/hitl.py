"""
Lab 11 - Part 4: Human-in-the-Loop Design
  TODO 11: Confidence Router
  TODO 12: Design 3 HITL decision points
"""
from dataclasses import dataclass


HIGH_RISK_ACTIONS = [
    "transfer_money",
    "close_account",
    "change_password",
    "delete_data",
    "update_personal_info",
]


@dataclass
class RoutingDecision:
    """Result of the confidence router."""

    action: str
    confidence: float
    reason: str
    priority: str
    requires_human: bool


class ConfidenceRouter:
    """Route agent responses based on confidence and risk level."""

    HIGH_THRESHOLD = 0.9
    MEDIUM_THRESHOLD = 0.7

    def route(
        self,
        response: str,
        confidence: float,
        action_type: str = "general",
    ) -> RoutingDecision:
        """Route a response based on confidence score and action type."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        if action_type in HIGH_RISK_ACTIONS:
            return RoutingDecision(
                action="escalate",
                confidence=confidence,
                reason=f"High-risk action: {action_type}",
                priority="high",
                requires_human=True,
            )

        if confidence >= self.HIGH_THRESHOLD:
            return RoutingDecision(
                action="auto_send",
                confidence=confidence,
                reason="High confidence",
                priority="low",
                requires_human=False,
            )

        if confidence >= self.MEDIUM_THRESHOLD:
            return RoutingDecision(
                action="queue_review",
                confidence=confidence,
                reason="Medium confidence - needs review",
                priority="normal",
                requires_human=True,
            )

        return RoutingDecision(
            action="escalate",
            confidence=confidence,
            reason="Low confidence - escalating",
            priority="high",
            requires_human=True,
        )


hitl_decision_points = [
    {
        "id": 1,
        "name": "High-Value Transfer Approval",
        "intent": "transfer_money",
        "trigger": "Any money transfer request above policy threshold or any transfer flagged as unusual by risk checks.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Request amount, source account, destination account, user authentication status, fraud flags, recent transfer history, and customer confirmation artifacts.",
        "diff": "Before: no transfer scheduled. After: funds move from source account to destination account and ledger balances change.",
        "example": "Customer asks to transfer 250,000,000 VND to a new beneficiary outside normal behavior.",
        "approval_path": "Reviewer verifies customer intent and fraud signals, records reviewer_id plus approval_id, then approves execution.",
        "reject_path": "Reviewer rejects if account ownership, destination, or fraud context is inconsistent; request is not executed.",
        "timeout_path": "If reviewer does not decide before SLA expiry, the transfer is canceled by default.",
        "audit_fields": "request_id, correlation_id, intent, source_account, destination_account, amount, pre_state, post_state_diff, reviewer_id, approval_id, reviewer_timestamp",
    },
    {
        "id": 2,
        "name": "Sensitive Profile Change Review",
        "intent": "change_password / update_personal_info",
        "trigger": "Password reset, phone/email change, KYC data update, or personal info update requested after low-confidence identity checks.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Current profile fields, proposed new values, verification evidence, session risk score, device fingerprint, and prior account recovery events.",
        "diff": "Before: current password hash and personal profile fields remain unchanged. After: password or profile attributes are replaced with the requested values.",
        "example": "Customer asks to change registered phone number and reset password from a new device in another city.",
        "approval_path": "Reviewer confirms identity proof and update legitimacy, records reviewer identity and timestamp, then releases the change.",
        "reject_path": "Reviewer rejects if evidence is weak or conflicts with recent account activity; no profile mutation is applied.",
        "timeout_path": "If review times out, the requested change remains pending or is canceled; no credential or profile update is committed.",
        "audit_fields": "request_id, correlation_id, intent, fields_changed, old_values_redacted, new_values_redacted, reviewer_id, decision_timestamp, verification_evidence_ref",
    },
    {
        "id": 3,
        "name": "Account Closure or External Data Release",
        "intent": "close_account / send_data_externally",
        "trigger": "Request to close an account, delete retained data, or send customer/internal data to any external destination.",
        "hitl_model": "human-in-the-loop",
        "context_needed": "Account status, linked products, outstanding balances, destination allowlist check, payload classification, legal hold flags, and customer authorization records.",
        "diff": "Before: account stays active and data remains internal. After: account status becomes closed or a payload leaves the controlled environment.",
        "example": "Customer asks to close a joint account or an agent attempts to send account statements to a new external endpoint.",
        "approval_path": "Reviewer validates closure prerequisites or destination allowlist, confirms least-privilege payload, and records approval metadata before execution.",
        "reject_path": "Reviewer rejects if balances remain, legal obligations block closure, or destination/payload fails policy.",
        "timeout_path": "If reviewer does not respond in time, action is blocked by default and no egress or closure occurs.",
        "audit_fields": "request_id, correlation_id, intent, destination, payload_summary, account_state_before, state_diff, reviewer_id, reviewer_timestamp, final_decision",
    },
]


def test_confidence_router():
    """Test ConfidenceRouter with sample scenarios."""
    router = ConfidenceRouter()

    test_cases = [
        ("Balance inquiry", 0.95, "general"),
        ("Interest rate question", 0.82, "general"),
        ("Ambiguous request", 0.55, "general"),
        ("Transfer $50,000", 0.98, "transfer_money"),
        ("Close my account", 0.91, "close_account"),
    ]

    print("Testing ConfidenceRouter:")
    print("=" * 80)
    print(f"{'Scenario':<25} {'Conf':<6} {'Action Type':<18} {'Decision':<15} {'Priority':<10} {'Human?'}")
    print("-" * 80)

    for scenario, conf, action_type in test_cases:
        decision = router.route(scenario, conf, action_type)
        print(
            f"{scenario:<25} {conf:<6.2f} {action_type:<18} "
            f"{decision.action:<15} {decision.priority:<10} "
            f"{'Yes' if decision.requires_human else 'No'}"
        )

    print("=" * 80)


def test_hitl_points():
    """Display HITL decision points."""
    print("\nHITL Decision Points:")
    print("=" * 60)
    for point in hitl_decision_points:
        print(f"\n  Decision Point #{point['id']}: {point['name']}")
        print(f"    Trigger:  {point['trigger']}")
        print(f"    Model:    {point['hitl_model']}")
        print(f"    Context:  {point['context_needed']}")
        print(f"    Example:  {point['example']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_confidence_router()
    test_hitl_points()
