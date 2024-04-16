from PIL import Image

# 读取图像
image = Image.open("input_images/_DSC8679.JPG")

# 将图像转换为全黑
black_image = Image.new("RGB", image.size, (0, 0, 0))
white_image = Image.new("RGB", image.size, (255, 255, 255))

# 保存全黑图像
# black_image.save("test_mask.png")
white_image.save("test_mask.png")
