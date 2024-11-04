import tkinter as tk

from napytau.cli.cli_arguments import CLIArguments
from napytau.core.logic_mockup import logic


def init(cli_arguments: CLIArguments):
    # Create the main window
    root = tk.Tk()
    root.geometry("800x500")  # Set window size
    root.title("TPSE Mockup")  # Set window title

    # Create a StringVar to associate with the button
    button_text = tk.StringVar()
    label_text = tk.StringVar()
    entry_text = tk.StringVar()

    def calculate():
        string_value = entry_text.get()

        if not string_value:
            return

        num = int(entry_text.get())
        result = logic(num)
        label_text.set(f"Result: {result}")

    # Create the button widget with all options
    button_text.set("Calculate!")
    button = tk.Button(root,
                       textvariable=button_text,
                       anchor=tk.CENTER,
                       compound=tk.CENTER,
                       bg="gray",
                       activebackground="lightgray",
                       height=3,
                       width=30,
                       bd=3,
                       font=("Arial", 16, "bold"),
                       cursor="hand2",
                       fg="black",
                       padx=15,
                       pady=15,
                       justify=tk.CENTER,
                       wraplength=250,
                       command=calculate,
                       )

    # Create the label and entry widgets
    label_text.set("Result:")
    label = tk.Label(root, textvariable=label_text, font=("Arial", 16, "bold"))
    entry = tk.Entry(root, textvariable=entry_text, width=50, font=("Arial", 16, "bold"))

    # Pack the button into the window
    label.pack(pady=30)
    entry.pack()
    button.pack(pady=20)  # Add some padding to the top

    # Run the main event loop
    root.mainloop()
