import numpy as np
from PIL import Image
import os

# ECC curve implementation
class ECC_Curve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p

    def is_on_curve(self, point):
        if point is None:
            return True
        x, y = point
        # Ensure calculations are done modulo p at each step to prevent overflow
        return (pow(y, 2, self.p) - (pow(x, 3, self.p) + (self.a * x) % self.p + self.b) % self.p) % self.p == 0

    def inverse_mod(self, k):
        # Handle k=0 case if necessary, though for point addition denominators should not be 0 mod p
        if k % self.p == 0:
             raise ValueError("Inverse does not exist for 0 mod p")
        return pow(k, self.p - 2, self.p)

    def point_add(self, p1, p2):
        if p1 is None: return p2
        if p2 is None: return p1

        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2 and y1 != y2:
            return None # Point at infinity

        if x1 == x2: # Point doubling
            # Ensure 2*y1 is not 0 mod p
            if (2 * y1) % self.p == 0:
                 return None # Point at infinity if y1 = 0 mod p
            m = (3 * pow(x1, 2, self.p) + self.a) * self.inverse_mod(2 * y1)
        else: # Point addition of distinct points
            # Ensure x2 - x1 is not 0 mod p
            denominator = (x2 - x1) % self.p
            if denominator == 0:
                 # This case should ideally not happen if x1 != x2, unless p is very small
                 raise ValueError("Denominator is 0 mod p during point addition")
            m = (y2 - y1) * self.inverse_mod(denominator)

        m %= self.p
        x3 = (pow(m, 2, self.p) - x1 - x2) % self.p
        y3 = (m * (x1 - x3) - y1) % self.p
        return (x3 % self.p, y3 % self.p) # Ensure results are within the field

    def scalar_mult(self, k, point):
        result = None
        addend = point
        # Ensure k is positive
        k = k % self.n if hasattr(self, 'n') and self.n is not None else k
        if k == 0:
            return None
        if k < 0:
             # Handle negative scalar multiplication if needed (not standard for simple encoding)
             raise ValueError("Negative scalar multiplication not implemented")

        while k > 0:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1
        return result

# ECC Encryptor
class ECC_Encryptor:
    def __init__(self, curve, G, n):
        self.curve = curve
        self.G = G
        self.n = n # Order of the base point G
        # Ensure G is on the curve and n is the order of G
        if not self.curve.is_on_curve(self.G):
            raise ValueError("Generator point G is not on the curve.")
        # Basic check for n being the order (scalar_mult(n, G) should be point at infinity)
        # A more rigorous check would be needed for true security
        if self.curve.scalar_mult(n, self.G) is not None:
             # This check is simplified; a proper order check is more complex
             print("Warning: n might not be the correct order of G.")


        self.private_key = np.random.randint(1, n)
        self.public_key = self.curve.scalar_mult(self.private_key, G)
        if self.public_key is None:
             raise ValueError("Generated public key is the point at infinity. Regenerate keys.")


    def encrypt_point(self, M_point, recipient_public_key):
        # Ensure M_point is on the curve
        if not self.curve.is_on_curve(M_point):
            # This shouldn't happen if image_to_points works correctly with the map
            raise ValueError(f"Plaintext point {M_point} is not on the curve.")

        k = np.random.randint(1, self.n)
        C1 = self.curve.scalar_mult(k, self.G)
        # Ensure scalar multiplication doesn't result in the point at infinity unexpectedly
        if C1 is None:
             raise ValueError("C1 is the point at infinity during encryption.")

        C2 = self.curve.point_add(M_point, self.curve.scalar_mult(k, recipient_public_key))
        if C2 is None:
             raise ValueError("C2 is the point at infinity during encryption.")

        return (C1, C2)

# ECC Decryptor
class ECC_Decryptor:
    def __init__(self, curve, private_key):
        self.curve = curve
        self.private_key = private_key

    def decrypt_point(self, C1, C2):
        # Ensure C1 and C2 are on the curve (basic check)
        if not self.curve.is_on_curve(C1):
             print(f"Warning: C1 point {C1} not on curve during decryption.")
        if not self.curve.is_on_curve(C2):
             print(f"Warning: C2 point {C2} not on curve during decryption.")

        S = self.curve.scalar_mult(self.private_key, C1)
        # If S is the point at infinity, decryption will fail or result in point at infinity
        if S is None:
             print("Warning: S is the point at infinity during decryption.")
             return None # Or handle as an error

        S_inv = (S[0], (-S[1]) % self.curve.p)
        return self.curve.point_add(C2, S_inv)

# Utility functions
def generate_point_map(curve):
    val = 0
    point_map = {}
    required_points = 256 # We need a mapping for 0-255
    points_found = []

    # Iterate through possible x and y values
    for x in range(curve.p):
        rhs = (pow(x, 3, curve.p) + (curve.a * x) % curve.p + curve.b) % curve.p
        for y in range(curve.p):
            if (pow(y, 2, curve.p)) % curve.p == rhs:
                point = (x, y)
                # Add points to a temporary list first
                points_found.append(point)

    # Sort the points to ensure a consistent mapping
    # Sorting by x then y is a common approach
    points_found.sort(key=lambda point: (point[0], point[1]))

    # Check if we found enough points
    if len(points_found) < required_points:
        raise ValueError(f"Not enough points found on the curve for modulus {curve.p}. Needed {required_points}, found {len(points_found)}. Consider a larger prime p.")

    # Create the map for the first required_points
    for i in range(required_points):
        point_map[i] = points_found[i]

    return point_map

def inverse_map(point_map):
    return {v: k for k, v in point_map.items()}

def image_to_points(image_array, point_map):
    # Ensure all pixel values are valid keys in the point_map
    for val in image_array.flatten():
        if val not in point_map:
             # This check should ideally pass after fixing generate_point_map
             raise ValueError(f"Pixel value {val} not found in point map.")

    return [point_map[val] for val in image_array.flatten()]

def points_to_image(points, inverse_point_map, shape):
    # Ensure all points are valid keys in the inverse_point_map
    for p in points:
         if p not in inverse_point_map:
              # This indicates an issue with decryption or the inverse map
              print(f"Warning: Decrypted point {p} not found in inverse point map. It might not be a valid point from the original mapping.")
              # You might want a strategy to handle this, e.g., map to 0 or a default value
              # For this example, we'll let the KeyError happen if it does, but the warning is useful.
              # A safer approach might be:
              # vals = [inverse_point_map.get(p, 0) for p in points] # Map unknown points to 0


    vals = [inverse_point_map[p] for p in points]
    return np.array(vals).reshape(shape).astype(np.uint8)

# Setup shared ECC params
# Increased modulus p to ensure enough points
a, b, p = 2, 3, 263 # Increased modulus p
# A valid point on the curve y^2 = x^3 + 2x + 3 mod 263
G = (3, 6)
# The number of points on the curve is 268. Using this as a placeholder for n.
# In a real application, n should be the order of G, which is a divisor of the total number of points.
n = 268 # Approximate number of points (minus point at infinity)
# Note: Using the total number of points as n is generally incorrect. n should be the order of the generator G.
# Finding the order of a point is computationally hard in general, which is related to ECC security.
# For a simple demonstration, using a value close to the number of points might work, but it's not cryptographically sound.
# Let's proceed with these parameters for the purpose of fixing the KeyError, acknowledging the cryptographic weakness.

curve = ECC_Curve(a, b, p)
try:
    point_map = generate_point_map(curve)
    inverse_point = inverse_map(point_map)
    print(f"Successfully generated point map with {len(point_map)} entries.")
except ValueError as e:
    print(f"Error initializing point map: {e}")
    # Exit or handle the error appropriately if not enough points are found
    exit()


# Generate keys
# We need to ensure n is the order of G for secure key generation.
# With our current simplified approach, this is a potential weakness.
# Using the total number of points (minus 1) or a large prime divisor of it is common.
# For p=263, a=2, b=3, the number of points is 268. 268 = 2^2 * 67.
# Prime divisors are 2, 67. The largest prime divisor is 67. This is still too small for security.
# A real curve would have a very large prime order n.
# Let's keep n=268 for now, acknowledging the limitation.
encryptor = ECC_Encryptor(curve, G, n)
decryptor = ECC_Decryptor(curve, encryptor.private_key)

# Create dummy image files for demonstration if they don't exist
if not os.path.exists('./images'):
    os.makedirs('./images')
if not os.path.exists('./images/plain_image.png'):
    # Create a simple grayscale image (e.g., a gradient)
    img = np.linspace(0, 255, 32*32, dtype=np.uint8).reshape((32, 32))
    Image.fromarray(img, mode='L').save('./images/plain_image.png')
    print("Created dummy plain_image.png")


def encrypt_image(input_path, output_path):
    try:
        image = Image.open(input_path).convert("L").resize((32, 32))
        img_array = np.array(image)
        plaintext_points = image_to_points(img_array, point_map)
        ciphertext = [encryptor.encrypt_point(P, encryptor.public_key) for P in plaintext_points]

        # Flatten ciphertext as integers to save (C1x, C1y, C2x, C2y)
        cipher_flat = [item for C1, C2 in ciphertext for item in (*C1, *C2)]
        # Use a larger dtype if p is large, e.g., np.uint16 or np.int64
        # With p=263, uint16 is sufficient as values are <= 262
        np.save(output_path, np.array(cipher_flat, dtype=np.uint16))
        print(f"Encrypted and saved to {output_path}.npy")
    except FileNotFoundError:
        print(f"Error: Input image not found at {input_path}")
    except Exception as e:
        print(f"An error occurred during encryption: {e}")


def decrypt_image(input_path, output_path):
    try:
        # Ensure the .npy extension is handled
        npy_input_path = input_path
        if not npy_input_path.endswith('.npy'):
            npy_input_path += '.npy'

        flat = np.load(npy_input_path)
        # Ensure the flat array has a shape divisible by 4
        if flat.shape[0] % 4 != 0:
             raise ValueError("Ciphertext data has an unexpected size.")

        chunks = flat.reshape(-1, 4)
        # Ensure the values are within the expected range for the modulus
        if np.max(chunks) >= curve.p or np.min(chunks) < 0:
             print(f"Warning: Loaded ciphertext contains values outside the expected range [0, {curve.p-1}].")

        ciphertext = [((int(c[0]), int(c[1])), (int(c[2]), int(c[3]))) for c in chunks]
        decrypted_points = [decryptor.decrypt_point(C1, C2) for C1, C2 in ciphertext]

        # Filter out any None results from decryption (e.g., if S was point at infinity)
        valid_decrypted_points = [p for p in decrypted_points if p is not None]

        if len(valid_decrypted_points) != len(decrypted_points):
             print(f"Warning: {len(decrypted_points) - len(valid_decrypted_points)} points failed to decrypt properly.")

        # Need to ensure the number of decrypted points matches the original image size
        original_size = 32 * 32
        if len(valid_decrypted_points) != original_size:
             raise ValueError(f"Number of decrypted points ({len(valid_decrypted_points)}) does not match expected image size ({original_size}).")

        recovered_image = points_to_image(valid_decrypted_points, inverse_point, (32, 32))
        Image.fromarray(recovered_image).save(output_path)
        print(f"Decrypted and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Encrypted file not found at {npy_input_path}")
    except Exception as e:
        print(f"An error occurred during decryption: {e}")


# Execute the encryption and decryption
encrypt_image('./images/plain_image.png', './images/ciphertext.png')
decrypt_image('./images/ciphertext.png', './images/decrypted.png')