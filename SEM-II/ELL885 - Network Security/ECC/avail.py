from PIL import Image

# =============================================
# Elliptic Curve Parameters (Must match encryption parameters)
# =============================================
p = 257  # Prime modulus
a = 0    # Curve parameter a
b = (-4) % p  # Ensure b is in field (mod p)
G = (2, 2)  # Base point
n = 240     # Order of G (example value)

# =============================================
# ECC Arithmetic Functions
# =============================================
def mod_inverse(a, p):
    if a == 0:
        raise ZeroDivisionError("Modular inverse does not exist for 0")
    lm, hm = 1, 0
    low, high = a % p, p
    while low > 1:
        ratio = high // low
        nm = hm - lm * ratio
        new = high - low * ratio
        hm, lm = nm, new
        high, low = low, new
    return lm % p

def is_on_curve(P):
    if P == 'O':
        return True
    x, y = P
    return (y ** 2 - (x ** 3 + a * x + b)) % p == 0

def point_add(P, Q):
    if P == 'O':
        return Q
    if Q == 'O':
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 != y2:
        return 'O'
    try:
        if P != Q:
            lam = ((y2 - y1) * mod_inverse(x2 - x1, p)) % p
        else:
            lam = ((3 * x1**2 + a) * mod_inverse(2 * y1, p)) % p
    except ZeroDivisionError:
        return 'O'
    x3 = (lam**2 - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def scalar_mult(k, P):
    if k == 0 or P == 'O':
        return 'O'
    Q = 'O'
    current = P
    while k > 0:
        if k % 2 == 1:
            Q = point_add(Q, current)
        current = point_add(current, current)
        k = k // 2
    return Q

# =============================================
# Precompute Valid Points on the Curve
# =============================================
def compute_curve_points():
    points = []
    for x in range(p):
        y_sq = (pow(x, 3, p) + a * x + b) % p
        for y in range(p):
            if (y * y) % p == y_sq:
                points.append((x, y))
    return points

curve_points = compute_curve_points()

# =============================================
# Key (Load your private key used in encryption)
# =============================================
private_key = 101  # Example key from encryption code

# =============================================
# Image Decryption
# =============================================
def decrypt_image(input_path, output_path):
    try:
        cipher_img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
        return

    cipher_width, cipher_height = cipher_img.size
    original_width = cipher_width // 4
    original_height = cipher_height
    cipher_pix = cipher_img.load()

    decrypted_img = Image.new('RGB', (original_width, original_height))
    decrypted_pix = decrypted_img.load()

    point_to_index = {point: idx for idx, point in enumerate(curve_points)}

    for y in range(original_height):
        for x in range(original_width):
            cipher_x = x * 4
            pixels = [
                cipher_pix[cipher_x, y],
                cipher_pix[cipher_x + 1, y],
                cipher_pix[cipher_x + 2, y],
                cipher_pix[cipher_x + 3, y]
            ]
            elements = []
            for pxl in pixels:
                elements.extend(pxl)

            r, g, b = [], [], []
            for channel in range(3):
                offset = channel * 4
                C1x = elements[offset]
                C1y = elements[offset + 1]
                C2x = elements[offset + 2]
                C2y = elements[offset + 3]

                C1 = (C1x % p, C1y % p)
                C2 = (C2x % p, C2y % p)

                if not is_on_curve(C1) or not is_on_curve(C2):
                    decrypted_val = 0
                else:
                    S = scalar_mult(private_key, C1)
                    if S == 'O':
                        decrypted_val = 0
                    else:
                        S_neg = (S[0], (-S[1]) % p)
                        Pm = point_add(C2, S_neg)
                        decrypted_val = point_to_index.get(Pm, 0) % 256

                if channel == 0:
                    r.append(decrypted_val)
                elif channel == 1:
                    g.append(decrypted_val)
                else:
                    b.append(decrypted_val)

            decrypted_pix[x, y] = (
                sum(r) % 256 if r else 0,
                sum(g) % 256 if g else 0,
                sum(b) % 256 if b else 0
            )

    decrypted_img.save(output_path)
    decrypted_img.show()

# =============================================
# Example Usage
# =============================================
decrypt_image('./images/ciphertext.png', './images/decrypted.png')
print("Decryption complete. Decrypted image saved as decrypted.png")
