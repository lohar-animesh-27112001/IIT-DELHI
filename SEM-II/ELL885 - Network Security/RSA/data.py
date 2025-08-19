from PIL import Image
import numpy as np
import os

os.makedirs('images', exist_ok=True)
plaintext_array = np.zeros((128, 128, 3), dtype=np.uint8)
for i in range(plaintext_array.shape[0]):
    for j in range(plaintext_array.shape[1]):
        if (i // 16 + j // 16) % 2 == 0:
            plaintext_array[i, j] = [255, 255, 255]
        else:
            plaintext_array[i, j] = [0, 0, 0]
plaintext_image = Image.fromarray(plaintext_array)
plaintext_image.save('images/plain_image.png')
print(" 'plain_image.png' created successfully inside 'images/' folder!")
plaintext_image.show()