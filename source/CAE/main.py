
import matplotlib.pyplot as plt
from maxPooling import *
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    img_path = 'source/CAE/1.jpg'
    transforms = transforms.Compose([transforms.Resize((472, 840)), transforms.ToTensor()])
    img_pil = Image.open(img_path)
    img = transforms(img_pil).unsqueeze(0)
    print(img.shape)
    model = MaxPooling()
    out = model(img)
    # mostrar imagen
    out = out.squeeze(0).permute(1, 2, 0).detach().numpy()
    plt.imshow(out)
    plt.show()
    print(out.shape)