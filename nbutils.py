
import logging

try:
    import IPython
    import IPython.display
    IPYTHON = IPython.get_ipython()
    IN_NOTEBOOK = IPYTHON.__class__.__name__ == 'ZMQInteractiveShell'
    if IN_NOTEBOOK:
        IPython.get_ipython().magic("matplotlib notebook")
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.rcParams['figure.figsize'] = (6.0, 6.0)
except:
    logging.info("Not running ipython.")
    IN_NOTEBOOK = False

def display(text, end="\n"):
    if IN_NOTEBOOK:
        IPython.display.display(text)
    else:
        print(text, end=end)

def display_markdown(markdown):
    if IN_NOTEBOOK:
        IPython.display.display(IPython.display.Markdown(markdown))
    else:
        print(markdown)

