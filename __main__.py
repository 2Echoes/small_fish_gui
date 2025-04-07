import sys, subprocess, traceback, os, re
from small_fish_gui import __version__

def main():
    import small_fish_gui.pipeline.main

def _get_version() :
    return __version__

AVAILABLE_ARGUMENTS = {
    ('-v','--v','--version') : "Prompt the software version.",
    ('--launch', '-l') : "Launch small fish gui, equivalent to no arguments.",
    ('-h', '--help', '--h') : "Prompt this help menu."
}

def is_last_version() :
    
    query = subprocess.run([sys.executable, '-m', 'pip', 'index', 'versions', 'small_fish_gui'], capture_output=True, text=True)
    all_version = query.stdout.split(',')
    latest_version = all_version[0]
    regex = r"\d+\.\d+\.\d+"
    latest = re.findall(regex, latest_version)
    
    current_version = _get_version()
    
    if len(latest) == 0 : 
        return current_version
    else :
        return current_version == latest[-1]

if __name__ == "__main__":

    if not is_last_version() :
        print("A new version of Small Fish is available. To update close small fish and type :\npip install --upgrade small_fish_gui")

    try :
        arguments = sys.argv

        if len(arguments) > 1 :
            if arguments[1] in ['-v','--v','--version'] :
                print(_get_version())
                quit()
            elif arguments[1] in ['--launch', '-l'] :
                pass
            elif arguments[1] in ['-h', '--help', '--h'] :
                for key, help in AVAILABLE_ARGUMENTS.items() :
                    print(f"{key} : {help}")
                quit()
            else :
                print(f"Incorrect argument : {arguments}, to launch small fish don't pass any argument or pick amongst {AVAILABLE_ARGUMENTS.keys()}")

        sys.exit(main())

    except Exception as error :
        with open("error_log.txt",'a') as error_log :
            error_log.writelines([
                f"version {_get_version()}",
                f"error : {error}",
                f"traceback :\n{traceback.format_exc()}",
            ])

        print(f"error_log saved at {os.getcwd()}/error_log.txt. Please consider reporting this by opening an issue on github.")