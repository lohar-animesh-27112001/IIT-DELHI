from PIL import Image

# Key Generation
p = 61  # Prime 1
q = 53  # Prime 2
n = p * q  # Modulus (3233)
phi = (p - 1) * (q - 1)  # Euler's totient (3120)
e = 17  # Public exponent (must be coprime with phi)
d = 2753  # Private exponent (modular inverse of e mod phi)

def rsa_encrypt(plain_byte):
    return pow(plain_byte, e, n)

def process_image(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    width, height = img_gray.size
    print(f"Original image size: {width}x{height}")
    img_bytes = bytearray(img_gray.tobytes())
    return img_bytes, width, height

def encrypt_image(img_bytes):
    cipher_ints = []
    for byte in img_bytes:
        cipher_ints.append(rsa_encrypt(byte))
    return cipher_ints

def create_cipher_image(cipher_ints, orig_width, orig_height):
    cipher_bytes = bytearray()
    for c in cipher_ints:
        cipher_bytes.extend(c.to_bytes(2, byteorder='big'))
    new_width = orig_width * 2
    new_height = orig_height
    cipher_img = Image.frombytes('L', (new_width, new_height), bytes(cipher_bytes))
    return cipher_img

if __name__ == "__main__":
    input_image = "./images/plain_image.png"
    output_image = "./images/encrypted_image.png"
    img_bytes, width, height = process_image(input_image)
    cipher_ints = encrypt_image(img_bytes)
    cipher_img = create_cipher_image(cipher_ints, width, height)
    cipher_img.save(output_image)
    print(f"Encrypted image saved as {output_image}")
    print(f"Ciphertext dimensions: {cipher_img.size[0]}x{cipher_img.size[1]}")