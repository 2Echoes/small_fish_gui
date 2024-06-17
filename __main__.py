import sys, subprocess
import PySimpleGUI as sg

def main():
    import small_fish_gui.pipeline.main

def is_last_version() :
    latest_version = str(subprocess.run([sys.executable, '-m', 'pip', 'install', '{}==random'.format('small_fish_gui')], capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ','').split(',')[-1]

    current_version = str(subprocess.run([sys.executable, '-m', 'pip', 'show', '{}'.format('small_fish_gui')], capture_output=True, text=True))
    current_version = current_version[current_version.find('Version:')+8:]
    current_version = current_version[:current_version.find('\\n')].replace(' ','')

    return current_version == latest_version

if __name__ == "__main__":

    if not is_last_version() :
        print("A new version of Small Fish is available. To update close small fish and type :\npip install --upgrade small_fish_gui")

    try :
        sys.exit(main())
    except Exception as error :
        sg.popup("Sorry. Something went wrong...\nIf you have some time to spare could you please communicate the error you encountered (next window) on :\nhttps://github.com/2Echoes/small_fish/issues.")
        sg.popup_error_with_traceback(error)
        raise error
