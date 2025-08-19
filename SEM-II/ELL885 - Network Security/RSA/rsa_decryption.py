from PIL import Image

# RSA Parameters
p = 61   # Prime 1
q = 53   # Prime 2
n = p * q  # Modulus (3233)
d = 2753  # Private exponent

def rsa_decrypt(cipher_int):
    return pow(cipher_int, d, n)

def process_cipher_image(image_path):
    cipher_img = Image.open(image_path)
    width, height = cipher_img.size
    print(f"Ciphertext image size: {width}x{height}")
    orig_width = width // 2
    orig_height = height
    cipher_bytes = bytearray(cipher_img.tobytes())
    return cipher_bytes, orig_width, orig_height

def decrypt_image(cipher_bytes):
    cipher_ints = []
    for i in range(0, len(cipher_bytes), 2):
        cipher_int = (cipher_bytes[i] << 8) | cipher_bytes[i+1]
        cipher_ints.append(cipher_int)
    plain_bytes = bytearray()
    for c in cipher_ints:
        decrypted_byte = rsa_decrypt(c)
        plain_bytes.append(decrypted_byte)
    return plain_bytes

def create_recovered_image(plain_bytes, width, height):
    recovered_img = Image.frombytes('L', (width, height), bytes(plain_bytes))
    return recovered_img

if __name__ == "__main__":
    input_cipher = "./images/encrypted_image.png"
    output_recovered = "./images/recovered_image.png"
    cipher_bytes, orig_width, orig_height = process_cipher_image(input_cipher)
    plain_bytes = decrypt_image(cipher_bytes)
    recovered_img = create_recovered_image(plain_bytes, orig_width, orig_height)
    recovered_img.save(output_recovered)
    print(f"Recovered image saved as {output_recovered}")
    print(f"Recovered dimensions: {orig_width}x{orig_height}")
    # Image.open(input_cipher).show(title="Encrypted Image")
    # recovered_img.show(title="Recovered Image")