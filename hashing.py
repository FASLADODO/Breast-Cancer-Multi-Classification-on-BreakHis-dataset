from PIL import Image
import imagehash
hash = imagehash.whash(Image.open('SOB_B_A-14-22549AB-200-001.png'))
print(hash)
hash1 = hash = imagehash.whash(Image.open('1.png'))
print(hash1)