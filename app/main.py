from tkinter import *
import os
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


# функция генерации самого предсказания с помощью нейронной сети
def predict(model, test_loader):
    # без рассчета градиентов
    with torch.no_grad():
        logits = []
        # переводим модель в режим предсказания и делаем его
        for inputs in test_loader:
            inputs = inputs.to('cpu')
            model.eval()
            outputs = model(inputs[None, ...]).cpu()
            logits.append(outputs)
    probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs


# функция обработки изображения и подачи в модель для генерации предсказания
def genral_pred(labels, model_incep):
    global pic_path
    if pic_path == 'no_path':
        name['text'] = 'Выберите картинку'
    else:
        # обработка изображения и его нормализация
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
        # занесение в загрузчик в модель
        test_loader = DataLoader(x, shuffle=False, batch_size=64)
        # генерация предсказания и вывод на экран
        probs = predict(model_incep, test_loader)
        preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
        name['text'] = 'Это ' + str(d[preds[0]])


# выход с окна рассмотрения всех персонажей
def leave2(root2):
    root2.destroy()


# выход с главного окна
def leave():
    root.destroy()


# функция вывода выбранного изображения на экран
def show_img(path):
    global pic
    img = Image.open(path)
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    pic['width'] = 300
    pic['height'] = 300
    pic['image'] = img
    pic.image = img


# функция генерации диалогового окна для выбора изображения
def choose_file():
    global pic_path
    old = pic_path
    pic_path = filedialog.askopenfilename(title='open')
    new = pic_path
    if old != new and new != '':
        show_img(pic_path)


# окно с рассморением всех персонажей
def all():
    global d
    root2 = Toplevel(root)
    root2["bg"] = '#b734eb'
    # создание фрейма с персонажами
    mainPart2 = Frame(root2, bg='#b734eb')
    # создание кнопки выхода
    btn_exit2 = HoverButton(
        root2,
        text='Назад',
        activebackground='#00ff00',
        font=("Courier", 22),
        command=lambda: leave2(root2)
    )
    info2 = Label(
        mainPart2,
        text='Здесь показаны все основные\nперсонажи мультсериала',
        bg='#b734eb',
        font=("Courier", 30),
    )
    # расположение на экране фрейма и кнопки выхода
    btn_exit2.pack(anchor=NW, padx=20, pady=20)
    mainPart2.pack()
    info2.pack()
    root2.resizable(width=False, height=False)
    # реализация меню персонажей с ползунком
    canvas_2 = Canvas(mainPart2, width=300, height=900)
    canvas_2.pack(side=LEFT, fill=BOTH, expand=1)
    scroll = Scrollbar(mainPart2, orient=VERTICAL, command=canvas_2.yview)
    scroll.pack(side=RIGHT, fill=Y)
    canvas_2.configure(yscrollcommand=scroll.set)
    canvas_2.bind('<Configure>', lambda e: canvas_2.configure(scrollregion = canvas_2.bbox("all")))
    dop_frame = Frame(canvas_2, )
    canvas_2.create_window((0, 0), window=dop_frame, anchor="nw")
    folder = 'characters'
    for address, dirs, files in os.walk(folder):
        for file in files:
            pic_new = Label(dop_frame)
            img = Image.open(address + '/' + file)
            img = img.resize((300, 300), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            pic_new['width'] = 300
            pic_new['height'] = 300
            pic_new['image'] = img
            pic_new.image = img
            pic_new.pack(padx=170, pady=5)
            Label(dop_frame, text=d[file.split('.')[0]], font=("Courier", 18)).pack()
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
# генерация виджетов
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
# окно с картинкой
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
# словарь (хеш-таблица), для соотношения названий классов и русскоязычных
# имен персонажей
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
     'lenny_leonard': 'Ленни Леонард',
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
# метки классов и загрузка модели
for k in d.keys():
    labels.append(k)
model_incep = torch.load('full_model.pth', map_location=torch.device('cpu'))
# свойства окна
root.resizable(width=False, height=False)
root.attributes('-fullscreen', True)
root.mainloop()
