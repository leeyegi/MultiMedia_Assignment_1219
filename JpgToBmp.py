from PIL import Image
#jpg to bmp하는 모듈


#이미지 불러오기
file_in = "./test.jpg"
img = Image.open(file_in)

#저장할 이미지 파일이름
file_out = "./test.bmp"

print (len(img.split()))

#image가 RGBA이면 split후 A만 빼고 merge
if len(img.split()) == 4:
    r, g, b, a = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.save(file_out)

else:
    img.save(file_out)