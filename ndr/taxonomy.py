"""
NDR intent taxonomy and the intent -> operational action policy layer.

Design note
-----------
Intent (what the customer *said*) is deliberately kept separate from action
(what ops should *do*). Two reasons:

1.  Business policy changes far more often than language does. When ops decides
    that "out of station" should hold at hub instead of rescheduling, that is a
    one-line change here and does not require retraining the model.
2.  Several distinct intents collapse onto the same action (REFUSED and
    CANCEL_REQUEST both end in RTO). Training a model to predict the *action*
    directly throws away the signal that makes those cases separable, and makes
    the model much harder to debug.
"""

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #

ACTIONS = {
    "RESCHEDULE_NEXT_ATTEMPT": "Reschedule delivery for the next working day",
    "RESCHEDULE_ON_DATE": "Schedule delivery for the customer-specified date/slot",
    "HOLD_AT_HUB": "Hold shipment at hub until customer is back in town",
    "UPDATE_ADDRESS": "Trigger address-correction flow, then re-attempt",
    "UPDATE_CONTACT": "Update the contact number on the shipment, then re-attempt",
    "DELIVER_TO_ALTERNATE": "Authorise handover to the named alternate recipient",
    "VERIFY_AND_CLOSE": "Verify POD against the claim, then close the NDR",
    "CONTACT_CUSTOMER": "Call the customer before the next attempt",
    "SHARE_PAYMENT_LINK": "Send prepaid payment link, convert COD to prepaid",
    "ARRANGE_SELF_PICKUP": "Share hub address/timings for customer pickup",
    "INITIATE_RTO": "Return shipment to origin",
    "CANCEL_ORDER": "Cancel the order, then RTO the shipment",
    "ESCALATE_OPS": "Route to the human operations queue",
    "ESCALATE_3PL": "Raise a dispute/service ticket with the logistics partner",
}


# --------------------------------------------------------------------------- #
# Intents
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Intent:
    code: str
    description: str
    primary_action: str
    # Action to use when the customer gave a concrete date/time in the message.
    action_if_date: Optional[str] = None
    # Actions that should never fire without a human in the loop, regardless of
    # model confidence, because they are destructive and hard to reverse.
    irreversible: bool = False
    # Extra ticket raised alongside the primary action.
    secondary_action: Optional[str] = None
    notes: str = ""


INTENTS = [
    Intent(
        code="CUSTOMER_UNAVAILABLE",
        description="Customer was not at home / was at work when the agent arrived",
        primary_action="RESCHEDULE_NEXT_ATTEMPT",
        action_if_date="RESCHEDULE_ON_DATE",
        notes="Highest-volume NDR intent in Indian e-commerce.",
    ),
    Intent(
        code="RESCHEDULE_REQUEST",
        description="Customer explicitly asks for a different date or time slot",
        primary_action="RESCHEDULE_ON_DATE",
        action_if_date="RESCHEDULE_ON_DATE",
    ),
    Intent(
        code="OUT_OF_STATION",
        description="Customer is travelling / out of city for several days",
        primary_action="HOLD_AT_HUB",
        action_if_date="RESCHEDULE_ON_DATE",
        notes="Re-attempting immediately guarantees a second failure. Hold instead.",
    ),
    Intent(
        code="ADDRESS_INCORRECT",
        description="Address on the shipment is wrong, incomplete or mis-pinned",
        primary_action="UPDATE_ADDRESS",
        secondary_action="CONTACT_CUSTOMER",
    ),
    Intent(
        code="ADDRESS_CHANGE",
        description="Address is correct but customer wants it delivered elsewhere",
        primary_action="UPDATE_ADDRESS",
        notes="Serviceability + COD-limit re-check required before re-attempt.",
    ),
    Intent(
        code="CONTACT_UPDATE",
        description="Phone number on the shipment is wrong or unreachable",
        primary_action="UPDATE_CONTACT",
        secondary_action="CONTACT_CUSTOMER",
        notes="Distinct from ADDRESS_INCORRECT: the fix is a different field.",
    ),
    Intent(
        code="COURIER_NO_CONTACT",
        description="Agent did not call, or called and refused to come to the door",
        primary_action="CONTACT_CUSTOMER",
        secondary_action="ESCALATE_3PL",
    ),
    Intent(
        code="FAKE_ATTEMPT_DISPUTE",
        description="Customer asserts no delivery attempt was ever made",
        primary_action="ESCALATE_3PL",
        secondary_action="CONTACT_CUSTOMER",
        notes=(
            "Operationally the most valuable class to isolate. A falsely-marked "
            "NDR is a 3PL quality failure, not a customer problem — auto-"
            "rescheduling it hides the root cause and inflates RTO."
        ),
    ),
    Intent(
        code="PAYMENT_ISSUE",
        description="COD cash not arranged, or customer wants to pay online",
        primary_action="SHARE_PAYMENT_LINK",
        secondary_action="CONTACT_CUSTOMER",
    ),
    Intent(
        code="REFUSED_NOT_NEEDED",
        description="Customer no longer wants the product / ordered by mistake / too late",
        primary_action="INITIATE_RTO",
        irreversible=True,
    ),
    Intent(
        code="CANCEL_REQUEST",
        description="Customer explicitly asks to cancel the order",
        primary_action="CANCEL_ORDER",
        irreversible=True,
    ),
    Intent(
        code="ALTERNATE_RECIPIENT",
        description="Customer authorises handover to a guard, neighbour or relative",
        primary_action="DELIVER_TO_ALTERNATE",
        notes="A genuine resolution path, not a failure — needs its own class.",
    ),
    Intent(
        code="ALREADY_RECEIVED",
        description="Customer says the parcel is already delivered / duplicate NDR",
        primary_action="VERIFY_AND_CLOSE",
        secondary_action="ESCALATE_3PL",
        notes="Often a mis-scan or a proxy delivery to a neighbour/guard.",
    ),
    Intent(
        code="SELF_PICKUP",
        description="Customer offers to collect the parcel from the hub",
        primary_action="ARRANGE_SELF_PICKUP",
    ),
    Intent(
        code="NEEDS_HUMAN_REVIEW",
        description=(
            "Reply cannot be safely automated: unintelligible, empty, off-topic, "
            "self-contradictory, multi-intent with conflicting actions, a legal "
            "threat, or a message showing distress"
        ),
        primary_action="ESCALATE_OPS",
        irreversible=True,
        notes=(
            "Reachable two ways: predicted directly by the classifier, OR "
            "assigned to any prediction below ABSTAIN_THRESHOLD. The second "
            "path is what makes the system degrade safely instead of guessing."
        ),
    ),
]

INTENT_BY_CODE = {i.code: i for i in INTENTS}
INTENT_CODES = [i.code for i in INTENTS]

# Intents the system is never allowed to execute without human sign-off, even at
# 0.99 confidence. Getting these wrong destroys a sale and cannot be undone.
REQUIRES_HUMAN_APPROVAL = {i.code for i in INTENTS if i.irreversible}


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

# Below this, we do not trust the label enough to act on it at all.
ABSTAIN_THRESHOLD = 0.45
# Between ABSTAIN and AUTO, we act but flag for review.
AUTO_EXECUTE_THRESHOLD = 0.75


@dataclass
class Decision:
    intent: str
    confidence: float
    action: str
    action_description: str
    mode: str                       # AUTO_EXECUTE | REVIEW | HUMAN_REQUIRED
    reason: str
    secondary_action: Optional[str] = None
    entities: dict = field(default_factory=dict)
    alternatives: list = field(default_factory=list)


def decide(intent_code: str, confidence: float, entities: dict,
           alternatives: list | None = None) -> Decision:
    """Map a predicted intent + extracted entities onto an operational action."""
    intent = INTENT_BY_CODE.get(intent_code, INTENT_BY_CODE["NEEDS_HUMAN_REVIEW"])

    # 1. Low confidence -> never guess, hand to a human.
    if confidence < ABSTAIN_THRESHOLD:
        return Decision(
            intent="NEEDS_HUMAN_REVIEW",
            confidence=confidence,
            action="ESCALATE_OPS",
            action_description=ACTIONS["ESCALATE_OPS"],
            mode="HUMAN_REQUIRED",
            reason=f"Confidence {confidence:.2f} below abstain threshold "
                   f"{ABSTAIN_THRESHOLD}; routed for human review.",
            entities=entities,
            alternatives=alternatives or [],
        )

    # 2. Entity-conditioned action refinement.
    action = intent.primary_action
    reason = f"Intent '{intent.code}' -> {action}."
    if intent.action_if_date and (entities.get("date") or entities.get("time")):
        action = intent.action_if_date
        when = " ".join(filter(None, [entities.get("date"), entities.get("time")]))
        reason = f"Intent '{intent.code}' with explicit slot ({when}) -> {action}."

    # 3. Irreversible actions are gated on a human regardless of confidence.
    if intent.code in REQUIRES_HUMAN_APPROVAL:
        mode = "HUMAN_REQUIRED"
        reason += " Irreversible or non-automatable — requires ops confirmation."
    elif confidence >= AUTO_EXECUTE_THRESHOLD:
        mode = "AUTO_EXECUTE"
    else:
        mode = "REVIEW"
        reason += f" Confidence {confidence:.2f} below auto-execute threshold; queued for review."

    return Decision(
        intent=intent.code,
        confidence=confidence,
        action=action,
        action_description=ACTIONS[action],
        mode=mode,
        reason=reason,
        secondary_action=intent.secondary_action,
        entities=entities,
        alternatives=alternatives or [],
    )
