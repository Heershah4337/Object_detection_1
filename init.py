#loading dataset
from PIL import Image
im = Image.open("C:/Users/ADMIN/Desktop/Im4.jpg")
width, height = im.size
print (width, height)
new_width  = 28
new_height = 28
im = im.resize((new_width, new_height), Image.ANTIALIAS)
im.save("C:/Users/ADMIN/Desktop/Im_4.jpg")
width, height = im.size
print (width, height)

#resize_image("C:/Users/ADMIN/Desktop/Test/Data/Test/Im4.jpg","C:/Users/ADMIN/Desktop/Test/Data/Test/Im_4.jpg")
def loadImages(path):
    # return array of images

     imagesList = listdir(path)
     loadedImages = []
     for image in imagesList:
         img = PImage.open(path + image)
         loadedImages.append(img)

         return loadedImages

#resizing the image
#def resize_image(path,new_path):




