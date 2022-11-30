import pandas as pd
from PySimpleGUI import PySimpleGUI as sg
from regressor import Regressor

def get_color(status):
    if(status == 'Successful'):
        return 'green'
    else:
        return 'red'

# Layout
sg.theme('DarkBlue')

pred_text = None

opcoes = {
    'Processador': ['AMD 4700S', 'AMD A10 9700', 'AMD A8 9600', 'AMD Athlon 3000G', 'AMD Ryzen 3 3200G', 'AMD Ryzen 3 4100', 'AMD Ryzen 5 2400G', 'AMD Ryzen 5 3600', 'AMD Ryzen 5 4500', 'AMD Ryzen 5 4600G', 'AMD Ryzen 5 5500', 'AMD Ryzen 5 5600G', 'AMD Ryzen 5 5600X', 'AMD Ryzen 5 PRO 4650G', 'AMD Ryzen 7 3700X', 'AMD Ryzen 7 3800XT', 'AMD Ryzen 7 5700G', 'AMD Ryzen 7 5700X', 'AMD Ryzen 7 5800X', 'AMD Ryzen 9 5900X', 'AMD Ryzen 9 5950X', 'Intel Core i3 10º Gen', 'Intel Core i3 12º Gen', 'Intel Core i3 3º Gen', 'Intel Core i5 10º Gen', 'Intel Core i5 11º Gen', 'Intel Core i5 12º Gen', 'Intel Core i5 1º Gen', 'Intel Core i5 2º Gen', 'Intel Core i5 3º Gen', 'Intel Core i5 8º Gen', 'Intel Core i7 10º Gen', 'Intel Core i7 11º Gen', 'Intel Core i7 12º Gen', 'Intel Core i7 1º Gen', 'Intel Core i7 2º Gen', 'Intel Core i7 3º Gen', 'Intel Core i9 10º Gen', 'Intel Core i9 11º Gen'],
    'Pro_score': [],
    'Placa_video': ['GeForce GT 1030', 'GeForce GT 210', 'GeForce GT 220', 'GeForce GT 610', 'GeForce GT 730', 'GeForce GTX 1050', 'GeForce GTX 1050 Ti', 'GeForce GTX 1060', 'GeForce GTX 1650', 'GeForce GTX 1660 Super', 'GeForce GTX 750 TI', 'GeForce RTX 2060', 'GeForce RTX 3050', 'GeForce RTX 3060', 'GeForce RTX 3060 TI', 'GeForce RTX 3070 TI', 'GeForce RTX 3080', 'GeForce RTX 3090', 'Intel HD Graphics 2000', 'Intel HD Graphics 2500','Quadro T1000', 'Radeon HD 6570', 'Radeon R5 220', 'Radeon R7', 'Radeon RX 550', 'Radeon RX 560', 'Radeon RX 6400', 'Radeon RX 6500 XT', 'Radeon RX 6600', 'Radeon RX 6650 XT', 'Radeon RX 6700 XT', 'Radeon RX 6800 XT', 'Radeon Vega 11', 'Radeon Vega 3', 'Radeon Vega 7', 'Radeon Vega 8'],
    'Video_score': [],
    'Memoria_video': [],
    'RAM': ['4', '8', '16', '32'],
    'SSD': ['0', '120', '128', '240', '256', '480', '512', '960'],
    'HD': ['0', '500', '1024', '2000', '3072'],
    'Fonte': ['200', '350', '400', '450', '500', '550', '600', '650', '750', '850']
}

layout = [
    [sg.Text('Selecione as configurações de computador entre as disponibilizadas pelo programa:')],
    [sg.Text('Aviso: em campos opcionais não preenchidos, serão considerados os valores padrões', text_color='red')],
    [sg.Text('Processador:', size=(17, 1)), sg.Combo(opcoes['Processador'], size=(25, 1), key='processador')],
    [sg.Text('Placa de Vídeo:', size=(17, 1)), sg.Combo(opcoes['Placa_video'], size=(25, 1), key='placa_video')],
    [sg.Text('RAM (GB):', size=(17, 1)), sg.Combo(opcoes['RAM'], size=(25, 1), key='ram')],
    [sg.Text('SSD (GB):', size=(17, 1)), sg.Combo(opcoes['SSD'], size=(25, 1), key='ssd'), sg.Text('Valor padrão (0)', size=(15, 1))],
    [sg.Text('HD (GB):', size=(17, 1)), sg.Combo(opcoes['HD'], size=(25, 1), key='hd'), sg.Text('Valor padrão (0)', size=(15, 1))],
    [sg.Text('Fonte (W):', size=(17, 1)), sg.Combo(opcoes['Fonte'], size=(25, 1), key='fonte')],
    [sg.Button('Enviar')],
    [sg.Text('Os valores são apenas aproximações e podem variar, em média, 13.15% do valor original.', text_color='red')],
    [sg.Text('Lembre-se que isso é um estudo.\nPara saber mais consulte o nosso artigo sobre os detalhes do experimento.', text_color='red')]
]

win = sg.Window('Prices predictor', layout, size=(600, 350))

pc = []

scores = pd.read_csv("processadores.csv")
scores = scores.dropna()
for i in scores['Scores']:
    opcoes['Pro_score'].append(i)
    
scores, memorias = pd.read_csv("placas_video.csv"), pd.read_csv("memoria_video.csv")
scores, memorias = scores.dropna(), memorias.dropna()
for i in range(len(scores['Score'])):
    opcoes['Video_score'].append(scores['Score'][i])
    opcoes['Memoria_video'].append(memorias['Memoria'][i])

for i in range(82):
    pc.append(0.0)
    
regressor = Regressor()

while True:
    events, values = win.read()
    if events == sg.WINDOW_CLOSED:
        break
    elif events == 'Enviar':
        success = True
        for i in range(82):
            pc[i] = 0.0
        if values['processador'] in opcoes['Processador']:
            index = opcoes['Processador'].index(values['processador'])
            index1 = opcoes['Processador'].index(values['processador']) + 7
            pc[0] = float(opcoes['Pro_score'][index])
            pc[index1] = 1.0
        else:
            sg.Popup('Processador inválido', text_color='red')
            success = False
            
        if values['placa_video'] in opcoes['Placa_video']:
            index = opcoes['Placa_video'].index(values['placa_video'])
            index1 = opcoes['Placa_video'].index(values['placa_video']) + 46
            pc[1] = float(opcoes['Video_score'][index])
            pc[2] = float(opcoes['Memoria_video'][index])
            pc[index1] = 1.0
        else:
            sg.Popup('Placa de vídeo inválida', text_color='red')
            sucess = False
        
        if values['ram'] in opcoes['RAM']:
            pc[3] = float(values['ram'])
        else:
            sg.Popup('RAM inválida', text_color='red')
            success = False
        
        if values['ssd'] in opcoes['SSD']:
            pc[4] = float(values['ssd'])
        else:
            pc[4] = 0.0
            
        if values['hd'] in opcoes['HD']:
            pc[5] = float(values['hd'])
        else:
            pc[5] = 0.0
        
        if values['fonte'] in opcoes['Fonte']:
            pc[6] = float(values['fonte'])
        else:
            sg.Popup('Fonte inválida', text_color='red')
            success = False
        if success:
            price = regressor.predict([pc])
            sg.Popup('Essa configuração de computador está avaliada em:\nR$ %.2f' % price, title='Predict')
        