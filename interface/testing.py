import FreeSimpleGUI as sg

layout = [
    [sg.Radio(['A',], key='button', group_id=0, key='test1'),sg.Radio(['B'], key='button', group_id=0, key='test2'), sg.Radio(['C'], key='button', group_id=0, key='test3')],
    [sg.Radio(['D',], key='button', group_id=1, key='test4')],
    [sg.Button('Ok')]
    ]

window = sg.Window(title= 'test', layout=layout)
event, values = window.read()

print(values)
