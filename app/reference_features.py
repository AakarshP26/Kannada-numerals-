"""
Reference Features for Quality Assessment
Pre-computed ideal feature profiles for each Kannada digit
"""
import numpy as np

# Ideal feature characteristics for each digit (based on proper handwriting)
# These are normalized reference values for quality comparison
IDEAL_FEATURES = {
    0: {  # ೦ - should have 1 large circular loop
        'num_loops': 1,
        'loop_area_ratio': 0.3,
        'circularity': 0.8,
        'num_endpoints': 0,
        'aspect_ratio': 1.0,
        'tips': {
            'loop': "Make sure to close the loop completely",
            'circularity': "Try to make it more circular",
            'aspect': "Keep it round, not too tall or wide"
        }
    },
    1: {  # ೧ - vertical stroke with hook
        'num_loops': 0,
        'num_endpoints': 2,
        'num_junctions': 0,
        'aspect_ratio': 0.4,
        'tips': {
            'loop': "This digit should not have closed loops",
            'endpoints': "Should have 2 stroke endings",
            'aspect': "Should be taller than wide"
        }
    },
    2: {  # ೨ - curved stroke
        'num_loops': 0,
        'num_endpoints': 2,
        'num_junctions': 0,
        'tips': {
            'loop': "This digit should not have closed loops",
            'endpoints': "Should have 2 stroke endings"
        }
    },
    3: {  # ೩ - curved with potential small loop
        'num_loops': 0,
        'num_endpoints': 2,
        'tips': {
            'endpoints': "Should have 2 stroke endings"
        }
    },
    4: {  # ೪ 
        'num_loops': 0,
        'num_endpoints': 2,
        'tips': {
            'endpoints': "Should have 2 stroke endings"
        }
    },
    5: {  # ೫
        'num_loops': 0,
        'num_endpoints': 2,
        'tips': {
            'endpoints': "Should have 2 stroke endings"
        }
    },
    6: {  # ೬ - has loop at bottom
        'num_loops': 1,
        'loop_area_ratio': 0.15,
        'num_endpoints': 1,
        'tips': {
            'loop': "Include the small loop at the bottom",
            'endpoints': "Should have 1 open stroke ending"
        }
    },
    7: {  # ೭
        'num_loops': 0,
        'num_endpoints': 2,
        'tips': {
            'endpoints': "Should have 2 stroke endings"
        }
    },
    8: {  # ೮ - two loops
        'num_loops': 2,
        'loop_area_ratio': 0.25,
        'num_endpoints': 0,
        'tips': {
            'loop': "Should have two connected loops",
            'endpoints': "Make sure both loops are closed"
        }
    },
    9: {  # ೯ - loop at top
        'num_loops': 1,
        'loop_area_ratio': 0.15,
        'num_endpoints': 1,
        'tips': {
            'loop': "Include the loop at the top",
            'endpoints': "Should have 1 open stroke ending at bottom"
        }
    }
}


def compute_quality_score(digit, features):
    """
    Compute quality score (0-100) comparing drawn features to ideal.
    
    Args:
        digit: The target digit (0-9)
        features: Dict with extracted feature values
        
    Returns:
        score (0-100), feedback list
    """
    ideal = IDEAL_FEATURES.get(digit, {})
    feedback = []
    scores = []
    
    # Check loop count
    if 'num_loops' in ideal:
        expected_loops = ideal['num_loops']
        actual_loops = features.get('loops', 0)
        
        if expected_loops == 0 and actual_loops > 0:
            scores.append(50)
            feedback.append("❌ This digit shouldn't have closed loops")
        elif expected_loops > 0 and actual_loops == 0:
            scores.append(40)
            feedback.append("❌ " + ideal.get('tips', {}).get('loop', "Missing loop"))
        elif expected_loops == actual_loops:
            scores.append(100)
            feedback.append("✅ Loop structure is correct!")
        else:
            scores.append(70)
            feedback.append(f"⚠️ Expected {expected_loops} loop(s), found {actual_loops}")
    
    # Check endpoints
    if 'num_endpoints' in ideal:
        expected_ep = ideal['num_endpoints']
        actual_ep = features.get('endpoints', 0)
        
        diff = abs(expected_ep - actual_ep)
        if diff == 0:
            scores.append(100)
            feedback.append("✅ Stroke endings are correct!")
        elif diff <= 1:
            scores.append(80)
            feedback.append(f"⚠️ Expected {expected_ep} endpoints, found {actual_ep}")
        else:
            scores.append(60)
            feedback.append(f"❌ Expected {expected_ep} endpoints, found {actual_ep}")
    
    # Check aspect ratio if relevant
    if 'aspect_ratio' in ideal:
        expected_ar = ideal['aspect_ratio']
        actual_ar = features.get('aspect_ratio', 1.0)
        
        if abs(expected_ar - actual_ar) < 0.3:
            scores.append(100)
            feedback.append("✅ Proportions look good!")
        else:
            scores.append(70)
            feedback.append("⚠️ " + ideal.get('tips', {}).get('aspect', "Check proportions"))
    
    # Calculate overall score
    if scores:
        overall = int(np.mean(scores))
    else:
        overall = 75  # Default if no specific checks
    
    # Add encouragement based on score
    if overall >= 90:
        feedback.insert(0, "🌟 Excellent work!")
    elif overall >= 75:
        feedback.insert(0, "👍 Good job! Keep practicing.")
    elif overall >= 60:
        feedback.insert(0, "💪 Nice try! Here's how to improve:")
    else:
        feedback.insert(0, "🎯 Let's work on this together:")
    
    return overall, feedback


def get_digit_description(digit):
    """Get learning description for a digit."""
    descriptions = {
        0: "೦ (Zero): A circular loop. Make sure it's closed!",
        1: "೧ (One): A vertical stroke with a small hook.",
        2: "೨ (Two): A curved stroke, open at both ends.",
        3: "೩ (Three): Curved shape, similar to English 3.",
        4: "೪ (Four): Angular shape with curves.",
        5: "೫ (Five): Curved stroke pattern.",
        6: "೬ (Six): Has a small loop at the bottom.",
        7: "೭ (Seven): Curved stroke, no loops.",
        8: "೮ (Eight): Two connected loops, like figure-8.",
        9: "೯ (Nine): Has a loop at the top, tail at bottom.",
    }
    return descriptions.get(digit, f"Kannada digit {digit}")
