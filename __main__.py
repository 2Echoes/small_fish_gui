import sys
import os
import PySimpleGUI as sg

def main():
    import small_fish_gui.pipeline.main


if __name__ == "__main__":
    try :
        sys.exit(main())
    except Exception as error :
        sg.popup("Sorry. Something went wrong...\nIf you have some time to spare could you please communicate the error you encountered (next window) on :\nhttps://github.com/2Echoes/small_fish/issues.")
        sg.popup_error_with_traceback(error)
        raise error
