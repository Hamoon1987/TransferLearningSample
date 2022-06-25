import torch
import glob
import os
from model import fetch_model
from PIL import Image
from torchvision import transforms
list_of_files = glob.glob('save_model/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = fetch_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


try:
    os.makedirs('./test')
except:
    assert(os.path.isdir('./test'))
img_name = os.listdir("./test")
checkpoint = torch.load(latest_file)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
model.eval()
img = Image.open("./test/" + img_name[-1])
transform = transforms.Compose([transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
img = transform(img)
img = img.unsqueeze(0)
with torch.no_grad():
    output = model(img)
_, pred = torch.max(output, 1)
classes = ["ant", "bee"]
print(classes[pred.item()])