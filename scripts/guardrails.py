# guardrails.py
"""
Very naive guardrails to filter out harmful queries (input filtering)
and to detect or adjust misleading/hallucinated outputs (output filtering).
In practice, you'd want something more robust, possibly a policy-based approach.
"""

# Simple keywords for demonstration
HARMFUL_KEYWORDS = ["kill", "attack", "terrorist", "bomb", "hate", "abuse"]
PROHIBITED_TOPICS = ["violent", "illegal"]

def reject_harmful_query(query):
    """
    Reject queries containing harmful keywords or topics.
    Return True if the query is rejected, otherwise False.
    """
    lower_query = query.lower()
    for kw in HARMFUL_KEYWORDS + PROHIBITED_TOPICS:
        if kw in lower_query:
            return True
    return False

def filter_misleading_output(text):
    """
    Naive approach to detect if the text is purely hallucinated or contradictory.
    Real solutions might use specialized classifiers or logic.
    
    For demonstration, we just check if the text contains suspicious patterns 
    like "undefined" or "NaN" or "???", etc.
    """
    suspicious_markers = ["???", "undefined", "NaN", "no data"]
    lower_text = text.lower()
    for marker in suspicious_markers:
        if marker in lower_text:
            # Return a sanitized message or an empty string
            return "Output flagged as potentially misleading."
    return text

if __name__ == "__main__":
    # Demo
    queries = [
        "What is the revenue for 2023?",
        "I want to bomb the building"
    ]
    for q in queries:
        if reject_harmful_query(q):
            print(f"Rejected query: {q}")
        else:
            print(f"Accepted query: {q}")
    
    outputs = [
        "The company's revenue is ??? for 2023",
        "Revenue: 50000"
    ]
    for o in outputs:
        print("Filtered output:", filter_misleading_output(o))
