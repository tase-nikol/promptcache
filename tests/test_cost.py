from promptcache.cost import SimplePricing, CostTracker

def test_cost_tracker():
    pricing = SimplePricing(input_per_1m=1.0, output_per_1m=2.0)
    ct = CostTracker(pricing=pricing)
    c = ct.compute("any", 1000, 500)
    assert c is not None
    assert c > 0
