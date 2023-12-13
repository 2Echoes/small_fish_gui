import PySimpleGUI as sg

layout = [
    [sg.InputText(default_text= 'Hello enter something please', key='answer')],
    [sg.Button('Ok')]      
          ]

window = sg.Window('Testing', layout=layout)
output = window.read()
print(output)