import re

def __parse_hex_float(hex_str: str) -> float:
    
    hex_str = hex_str.strip()
    
    #hex float pattern: 0x[mantissa]p[exponent]
    pattern = r'^0[xX]([0-9a-fA-F]*\.?[0-9a-fA-F]+)[pP]([+-]?\d+)$'
    match = re.match(pattern, hex_str)
    
    if not match:
        raise ValueError(f"Invalid hex format: {hex_str}")
    
    mantissa_str, exponent_str = match.groups()
    
    if '.' in mantissa_str:
        integer_part, fractional_part = mantissa_str.split('.')
    else:
        integer_part = mantissa_str
        fractional_part = ''
    
    if integer_part:
        mantissa_value = float(int(integer_part, 16))
    else:
        mantissa_value = 0.0
    
    if fractional_part:
        frac_value = 0.0
        for i, digit in enumerate(fractional_part, 1):
            frac_value += int(digit, 16) / (16 ** i)
        mantissa_value += frac_value
    
    exponent = int(exponent_str)
    
    result = mantissa_value * (2 ** exponent)
    
    return result


def _replace_hex_floats_in_code(code: str) -> str:

    hex_float_pattern = r'0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]+[pP][+-]?\d+'
    
    def replace_match(match):
        hex_str = match.group(0)
        try:
            decimal_value = _parse_hex_float(hex_str)
            return f"{decimal_value:.17g}"
        except Exception as e:
            print(f"Warning: Could not convert hex float {hex_str}: {e}")
            return hex_str
    
    return re.sub(hex_float_pattern, replace_match, code)


if __name__ == "__main__":
    # Test the parser with known values
    test_cases = [
        ("0x1.62e42fefa39efp-1", 0.6931471805599453, "ln(2)"),
        ("0x1.5bf0a8b145769p+1", 2.718281828459045, "e"),
        ("0x1.71547652b82fep+0", 1.4426950408889634, "log₂(e)"),
        ("0x1.921fb54442d18p+1", 3.141592653589793, "π"),
        ("0x1p0", 1.0, "1.0"),
        ("0x1p1", 2.0, "2.0"),
        ("0x1p-1", 0.5, "0.5"),
        ("0x1.8p1", 3.0, "3.0"),
        ("0xap0", 10.0, "10.0"),
    ]
    
    print("Testing hex float parser:")
    print("-" * 70)
    for hex_str, expected, description in test_cases:
        result = _parse_hex_float(hex_str)
        match = "✓" if abs(result - expected) < 1e-15 else "✗"
        print(f"{match} {hex_str:25s} = {result:.17g} ({description})")
        if abs(result - expected) >= 1e-15:
            print(f"  Expected: {expected:.17g}, Difference: {abs(result - expected)}")
    
    print("\n" + "=" * 70)
    print("Testing code replacement:")
    print("-" * 70)
    
    test_code = """
import torch

def kernel():
    ln2 = 0x1.62e42fefa39efp-1
    e = 0x1.5bf0a8b145769p+1
    log2e = 0x1.71547652b82fep+0
    pi = 0x1.921fb54442d18p+1
    return ln2 * e * pi
"""
    
    print("Original code:")
    print(test_code)
    print("\nCleaned code:")
    print(replace_hex_floats_in_code(test_code))

