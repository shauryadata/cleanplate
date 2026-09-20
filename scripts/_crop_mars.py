from PIL import Image
im = Image.open("datasets/backgrounds/mars_curiosity_360_pano.jpg")
im.crop((380, 125, 1680, 600)).save("datasets/backgrounds/mars_clean.jpg", quality=92)
print("   mars_clean.jpg regenerated from the panorama", im.size, "-> (1300, 475)")
