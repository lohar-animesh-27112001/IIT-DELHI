from PIL import Image
import numpy as np
import hashlib

from aes_encryption import AESEncryptor


class AESDecryptor:
    SBOX = AESEncryptor.SBOX
    INVERSE_SBOX = [
        0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
        0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
        0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
        0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
        0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
        0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
        0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
        0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
        0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
        0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
        0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
        0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
        0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
        0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
        0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
        0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
        0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
        0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
        0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
        0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
        0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
        0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
        0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
        0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
        0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
        0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
        0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
        0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
        0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
        0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
        0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
    ]
    
    RCON = AESEncryptor.RCON

    def __init__(self, key):
        self.key = self._pad_key(key)
        self.nk = len(self.key) // 4
        self.nr = {4:10, 6:12, 8:14}[self.nk]
        self.round_keys = self._key_expansion()

    def _pad_key(self, key):
        hash_obj = hashlib.sha256(key.encode()).digest()
        lengths = {16:16, 24:24, 32:32}
        return hash_obj[:lengths.get(len(key), 32)]

    def _key_expansion(self):
        key_words = [self.key[i:i+4] for i in range(0, len(self.key), 4)]
        for i in range(self.nk, 4 * (self.nr + 1)):
            temp = key_words[i - 1]
            if i % self.nk == 0:
                temp = self._sub_word(self._rot_word(temp))
                temp = bytes([temp[0] ^ self.RCON[i // self.nk - 1]] + list(temp[1:]))
            elif self.nk > 6 and i % self.nk == 4:
                temp = self._sub_word(temp)
            new_word = bytes([a ^ b for a, b in zip(key_words[i - self.nk], temp)])
            key_words.append(new_word)
        return [key_words[i * 4:(i + 1) * 4] for i in range(len(key_words) // 4)]

    def _sub_word(self, word):
        return bytes([self.SBOX[b] for b in word])

    def _rot_word(self, word):
        return word[1:] + word[0:1]

    def _bytes_to_state(self, data):
        return [list(data[i:i + 4]) for i in range(0, 16, 4)]

    def _state_to_bytes(self, state):
        return bytes(sum(state, []))

    def _inverse_sub_bytes(self, state):
        for row in state:
            for b in row:
                if b < 0 or b >= len(self.INVERSE_SBOX):
                    print(f"Invalid byte value {b} found in state: {state}")
                    raise IndexError(f"Byte value {b} is out of range for INVERSE_SBOX.")
        return [[self.INVERSE_SBOX[b] for b in row] for row in state]

    def _inverse_shift_rows(self, state):
        return [
            state[0],
            state[1][-1:] + state[1][:-1],
            state[2][-2:] + state[2][:-2],
            state[3][-3:] + state[3][:-3]
        ]

    def _inverse_mix_columns(self, state):
        inv_mixed = []
        for col in range(4):
            s = [
                state[0][col],
                state[1][col],
                state[2][col],
                state[3][col]
            ]
            inv_mixed.append([
                self._gf_mul(0x0e, s[0]) ^ self._gf_mul(0x0b, s[1]) ^ self._gf_mul(0x0d, s[2]) ^ self._gf_mul(0x09, s[3]),
                self._gf_mul(0x09, s[0]) ^ self._gf_mul(0x0e, s[1]) ^ self._gf_mul(0x0b, s[2]) ^ self._gf_mul(0x0d, s[3]),
                self._gf_mul(0x0d, s[0]) ^ self._gf_mul(0x09, s[1]) ^ self._gf_mul(0x0e, s[2]) ^ self._gf_mul(0x0b, s[3]),
                self._gf_mul(0x0b, s[0]) ^ self._gf_mul(0x0d, s[1]) ^ self._gf_mul(0x09, s[2]) ^ self._gf_mul(0x0e, s[3])
            ])
        return [[row[i] for row in inv_mixed] for i in range(4)]

    def _gf_mul(self, a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11b
            b >>= 1
        return p

    def _add_round_key(self, state, round_key):
        return [[b ^ k for b, k in zip(row, key)]
                for row, key in zip(state, round_key)]

    def _decrypt_block(self, block):
        if len(block) != 16:
            raise ValueError(f"Expected block of length 16, got {len(block)}")
        state = self._bytes_to_state(block)
        state = self._add_round_key(state, self.round_keys[self.nr])
        for round in range(self.nr - 1, 0, -1):
            state = self._inverse_shift_rows(state)
            state = self._inverse_sub_bytes(state)
            state = self._add_round_key(state, self.round_keys[round])
            state = self._inverse_mix_columns(state)
        state = self._inverse_shift_rows(state)
        state = self._inverse_sub_bytes(state)
        state = self._add_round_key(state, self.round_keys[0])
        return self._state_to_bytes(state)

    def _unpad_data(self, data):
        pad_len = data[-1]
        return data[:-pad_len] if pad_len <= 16 else data

    def decrypt_image(self, image_path, output_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        encrypted_data = img_array.tobytes()
        decrypted = bytearray()
        for i in range(0, len(encrypted_data), 16):
            block = encrypted_data[i:i + 16]
            while len(block) < 16:
                block += b'\x00'
            decrypted += self._decrypt_block(block)
        decrypted = self._unpad_data(decrypted)
        decrypted_img = Image.frombytes(img.mode, img.size, bytes(decrypted))
        decrypted_img.save(output_path)
        return decrypted_img

if __name__ == "__main__":
    key = "my_secret_key"
    decryptor = AESDecryptor(key)

    encrypted_img = "images/encrypted_image.png"
    decrypted_output_image = "images/decrypted_image.png"
    decrypted_img = decryptor.decrypt_image(encrypted_img, decrypted_output_image)
    decrypted_img.show()