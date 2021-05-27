from tkinter import *
from StyledWidgets import HoverButton
from tkinter import filedialog
from PIL import ImageTk, Image
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

def genral_pred(labels, model_incep):
    global pic_path
    if pic_path == 'no_path':
        name['text'] = 'Выберите картинку'
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        RESCALE_SIZE = 299
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        im = Image.open(pic_path)
        x = im.convert('RGB')
        x = transform(x)
        test_loader = DataLoader(x, shuffle=False, batch_size=64)
        probs = predict(model_incep, test_loader)
        preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
        print(d[preds[0]])
        name['text'] = 'Это ' + str(d[preds[0]])


def leave2(root2):
    root2.destroy()


def leave():
    root.destroy()


def show_img(path):
    global pic
    img = Image.open(path)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    pic['width'] = 300
    pic['height'] = 300
    pic['image'] = img
    pic.image = img


def choose_file():
    global pic_path
    old = pic_path
    pic_path = filedialog.askopenfilename(title='open')
    new = pic_path
    if old != new:
        show_img(pic_path)


def all():
    root2 = Toplevel(root)
    root2["bg"] = '#b734eb'
    # создание кнопки выхода
    btn_exit2 = HoverButton(
        root2,
        text='Назад',
        activebackground='#00ff00',
        font=("Courier", 22),
        command=lambda: leave2(root2)
    )
    btn_exit2.pack(anchor=NW, padx=20, pady=20)
    root2.resizable(width=False, height=False)
    root2.attributes('-fullscreen', True)
    root2.mainloop()


# создание основного окна
root = Tk()
root["bg"] = '#b734eb'
mainPart = Frame(root, bg='#b734eb')
# создание кнопки выхода
btn_exit = HoverButton(
    root,
    text='Выход',
    activebackground='#00ff00',
    font=("Courier", 22),
    command=leave
)
info = Label(
    mainPart,
    text='Выберите картинку одного из персонажей\n'
         'Симпсонов с вашего компьютера',
    bg='#b734eb',
    font=("Courier", 30),
)
choose = HoverButton(
    mainPart,
    text='Выбрать файл',
    activebackground='#4039fa',
    font=("Courier", 25),
    command=lambda: choose_file()
)
# здесь должно быть окно с картинкой
pic = Label(mainPart, width=60, height=30, bg='white')
name = Label(
    mainPart,
    text='Имя персонажа',
    bg='#b734eb',
    font=("Courier", 30)
)
btn_pred = HoverButton(
    mainPart,
    text='Предсказать!',
    activebackground='#4039fa',
    font=("Courier", 25),
    command=lambda: genral_pred(labels, model_incep)
)
btn_all = HoverButton(
    mainPart,
    text='Посмотреть всех\nперсонажей',
    activebackground='#4039fa',
    font=("Courier", 25),
    command=all
)
# расположение виджетов
btn_exit.pack(anchor=NW, padx=20, pady=20)
mainPart.pack()
info.grid(row=0, column=0)
choose.grid(row=1, column=0)
pic.grid(row=2, column=0)
name.grid(row=3, column=0)
btn_pred.grid(row=4, column=0)
btn_all.grid(row=5, column=0)
pic_path = 'no_path'
RESCALE_SIZE = 299
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
# свойства окна
root.resizable(width=False, height=False)
root.attributes('-fullscreen', True)
root.mainloop()
