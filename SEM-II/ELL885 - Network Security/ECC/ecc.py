from PIL import Image
import random

# Parameters
p = 257
a = 0
b = -4
G = (2, 2)
n = 240

def mod_inverse(a, p):
    a = a % p
    if a == 0:
        raise ValueError("Inverse does not exist")
    lm, hm = 1, 0
    low, high = a, p
    while low > 1:
        r = high // low
        nm = hm - lm * r
        new = high - low * r
        hm, lm = lm, nm
        high, low = low, new
    return lm % p

def point_add(P, Q):
    if P == 'O':
        return Q
    if Q == 'O':
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 != y2:
        return 'O'
    if P != Q:
        denom = x2 - x1
        if denom == 0:
            return 'O'
        lam = ((y2 - y1) * mod_inverse(denom, p)) % p
    else:
        if y1 == 0:
            return 'O'
        lam = ((3 * x1 * x1 + a) * mod_inverse(2 * y1, p)) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def scalar_mult(k, P):
    result = 'O'
    addend = P
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1
    return result

def compute_curve_points():
    points = []
    for x in range(p):
        y_sq = (x ** 3 + a * x + b) % p
        for y in range(p):
            if (y * y) % p == y_sq:
                points.append((x, y))
    return points

# Precompute
curve_points = compute_curve_points()
if len(curve_points) < 256:
    raise RuntimeError("Not enough points to map all 8-bit values!")

# Key generation
private_key = random.randint(1, n - 1)
public_key = scalar_mult(private_key, G)

def encrypt_image(input_path, output_path):
    img = Image.open(input_path)
    width, height = img.size
    pixels = img.convert('RGB').load()

    cipher_data = []
    for y in range(height):
        for x in range(width):
            for val in pixels[x, y]:  # R, G, B
                Pm = curve_points[val]
                k = random.randint(1, n - 1)
                C1 = scalar_mult(k, G)
                kQ = scalar_mult(k, public_key)
                C2 = point_add(Pm, kQ)
                if C1 == 'O': C1 = (0, 0)
                if C2 == 'O': C2 = (0, 0)
                cipher_data.extend([C1[0], C1[1], C2[0], C2[1]])

    # Cipher image: width is 4x original width
    cipher_img = Image.new('RGB', (width * 4, height))
    cipher_pixels = cipher_img.load()
    idx = 0
    for y in range(height):
        for x in range(width * 4):
            if idx + 2 < len(cipher_data):
                r = cipher_data[idx] % 256
                g = cipher_data[idx + 1] % 256
                b = cipher_data[idx + 2] % 256
                cipher_pixels[x, y] = (r, g, b)
                idx += 3
            else:
                cipher_pixels[x, y] = (0, 0, 0)
    cipher_img.save(output_path)
    print(f"Encrypted image saved at {output_path}")

# Example usage
encrypt_image('./images/plain_image.png', './images/ciphertext.png')
