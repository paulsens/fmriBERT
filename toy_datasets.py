# main.py
import sys

if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if "-m" in opts:
        #-m "this is the description of the run" will be at the end of the command line call
        run_desc = args[-1]
    else:
        run_desc = None
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

    print("run_desc is "+str(run_desc))