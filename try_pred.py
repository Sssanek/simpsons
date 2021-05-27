import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image
import numpy as np


def predict(model, test_loader):
    with torch.no_grad():
        logits = []
        for inputs in test_loader:
            inputs = inputs.to('cpu')
            model.eval()
            outputs = model(inputs[None, ...]).cpu()
            logits.append(outputs)
    probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


d = {'abraham_grampa_simpson': 'Абрахам Симпсон (Дед)',
     'agnes_skinner': 'Агнес Скиннер',
     'apu_nahasapeemapetilon': 'Апу Нахасапимапетилон',
     'barney_gumble': 'Барни Гамбл',
     'bart_simpson': 'Барт Симпсон',
     'carl_carlson': 'Карл Карлсон',
     'charles_montgomery_burns': 'Чарльз Монтгомери Бёрнс (Мистер Бёрнс)',
     'chief_wiggum': 'Клэнси Виггам (Шеф полиции)',
     'cletus_spuckler': 'Клетус Спаклер',
     'comic_book_guy': 'Продавец комиксов',
     'edna_krabappel': 'Эдна Крабаппл',
     'fat_tony': 'Жирный Тони',
     'gil': 'Гил Гундерсон',
     'groundskeeper_willie': 'Садовник Вилли',
     'homer_simpson': 'Гомер Симпсон',
     'kent_brockman': 'Кент Брокман',
     'krusty_the_clown': 'Клоун Красти',
     'lenny_leonard	': 'Ленни Леонард',
     'lionel_hutz': 'Лайнел Хатц',
     'lisa_simpson': 'Лиза Симпсон',
     'maggie_simpson': 'Мэгги Симпсон',
     'marge_simpson': 'Мардж Симпсон',
     'martin_prince': 'Мартин Принс',
     'mayor_quimby': 'Джо Куимби (Мэр Куимби)',
     'milhouse_van_houten': 'Милхаус Ван Хутен',
     'miss_hoover': 'Элизабет Гувер (Мисис Гувер)',
     'moe_szyslak': 'Мо Сизлак',
     'ned_flanders': 'Нед Фландерс',
     'nelson_muntz': 'Нельсон Манц',
     'otto_mann': 'Отто Манн',
     'patty_bouvier': 'Пэтти Бувье',
     'principal_skinner': 'Сеймур Скиннер (Директор Скиннер)',
     'professor_john_frink': 'Профессор Фринк',
     'rainier_wolfcastle': 'Райнер Вульфкасл',
     'ralph_wiggum': 'Ральф Виггам',
     'selma_bouvier': 'Сельма Бувье',
     'sideshow_bob': 'Сайдшоу Боб',
     'sideshow_mel': 'Сайдшоу Мел',
     'snake_jailbird': 'Змей Джейлбёрд',
     'troy_mcclure': 'Трой Макклюр',
     'waylon_smithers': 'Вэйлон Смитерс',
     'disco_stu': 'Диско Стю'
     }

labels = []
for k in d.keys():
    labels.append(k)
model_incep = torch.load('full_model.pth', map_location=torch.device('cpu'))
RESCALE_SIZE = 299
test_file = 'images.jfif'
label_encoder = LabelEncoder()
label_encoder.fit(labels)
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
im = Image.open(test_file)
x = im.convert('RGB')
x = transform(x)
test_loader = DataLoader(x, shuffle=False, batch_size=64)
probs = predict(model_incep, test_loader)
preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
print(d[preds[0]])
